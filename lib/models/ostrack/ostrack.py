import math
import os
from typing import List
import torch
import torch.nn.functional as F
from torch import nn

from lib.models.layers.head import build_box_head
from lib.models.ostrack.hivit import hivit_base
from lib.utils.box_ops import box_xyxy_to_cxcywh


class ProTeusH(nn.Module):
    def __init__(self, transformer, box_head, head_type="CENTER"):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.head_type = head_type

        # ============================================================
        # [Phase 1] 纯净模式：所有 Phase 3 组件全部注释掉
        # ============================================================
        # self.predictor = MambaPredictor(dim=512)
        # self.observer = UOTObserver(dim=512)
        # self.synergy = BayesianSynergy(dim=512)
        # self.norm_fusion = nn.LayerNorm(512)
        # self.fusion_map = nn.Linear(512, 512 * 2)
        # self.fusion_alpha = nn.Parameter(torch.tensor(0.0))
        # ============================================================

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                prompt_history=None,
                **kwargs):

        # 1. 预处理
        if template.shape[3] != search.shape[3]:
            padding_width = search.shape[3] - template.shape[3]
            template_padded = F.pad(template, (0, padding_width, 0, 0))
        else:
            template_padded = template

        x_in = torch.cat([template_padded, search], dim=2)

        # 2. Backbone 前向传播 (标准 OSTrack 逻辑)
        # Phase 1 不需要复杂的 forward_tracking，直接跑就行
        results = self.backbone(x_in)
        f3 = results[-1]

        # [B, 512, H, W] -> [B, N, 512]
        f3_flat = f3.flatten(2).transpose(1, 2)

        # 3. 提取 Search 区域特征
        visual_feats = f3_flat[:, -self.feat_len_s:]

        # ============================================================
        # [Phase 1] 纯净模式：没有任何融合逻辑
        # ============================================================
        # 这里直接把纯视觉特征送入 Head

        out = self.forward_head(visual_feats)
        return out

    def forward_head(self, cat_feature):
        # [B, N, C] -> [B, C, N] -> [B, C, H, W]
        opt = (cat_feature.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat)
            return {'pred_boxes': bbox.view(bs, Nq, 4), 'score_map': score_map_ctr,
                    'size_map': size_map, 'offset_map': offset_map}
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    # 1. 构建 Backbone
    backbone = hivit_base()
    hidden_dim = 512

    # 2. 加载 ImageNet 预训练权重 (关键)
    if cfg.MODEL.PRETRAIN_FILE and training:
        # Phase 1 加载的是 hivit_base_224.pth
        ckpt_path = cfg.MODEL.PRETRAIN_FILE
        try:
            print(f">>> [Phase 1] Loading ImageNet weights from: {ckpt_path}")
            # weights_only=False 解决 PyTorch 版本问题
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'net' in checkpoint:
                state_dict = checkpoint['net']
            else:
                state_dict = checkpoint

                # --- ostrack.py 中的 build_ostrack 修正片段 ---

            # 适配 ImageNet 权重 key (核心：处理 8 个 block 的偏移)
            backbone_dict = backbone.state_dict()
            new_dict = {}
            for k, v in state_dict.items():
                k_clean = k.replace('module.', '')

                # 处理核心 Transformer blocks 的偏移映射
                if k_clean.startswith('blocks.'):
                    parts = k_clean.split('.')
                    idx = int(parts[1])
                    # 将官方 0-23 映射到 8-31
                    new_idx = idx + 8
                    k_mapped = k_clean.replace(f'blocks.{idx}', f'blocks.{new_idx}')

                    if k_mapped in backbone_dict and v.shape == backbone_dict[k_mapped].shape:
                        new_dict[k_mapped] = v

                # 处理其他非 block 权重（如 patch_embed, pos_embed）
                elif k_clean in backbone_dict and v.shape == backbone_dict[k_clean].shape:
                    new_dict[k_clean] = v

            msg = backbone.load_state_dict(new_dict, strict=False)
            print(f">>> [Phase 1] Loaded {len(new_dict)} keys into Backbone. (Mapped Stage 3)")
            # 此时 Missing keys 应该只剩下 blocks.0 到 blocks.7

        except Exception as e:
            print(f">>> [Error] Failed to load ImageNet weights: {e}")
            # Phase 1 如果加载不到预训练权重是致命的，这里抛出异常
            raise e

    # 3. 构建 Head & Model
    box_head = build_box_head(cfg, hidden_dim)
    model = ProTeusH(backbone, box_head, head_type=cfg.MODEL.HEAD.TYPE)

    if training:
        # Phase 1 全员解冻
        for p in model.parameters():
            p.requires_grad = True
        print(">>> [Phase 1] All parameters unfreezed for training.")

    return model