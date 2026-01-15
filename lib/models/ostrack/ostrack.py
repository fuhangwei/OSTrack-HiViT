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
        ckpt_path = cfg.MODEL.PRETRAIN_FILE
        try:
            print(f">>> [Phase 1] Loading weights from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            # --- 【核心修复】深度解析嵌套的 state_dict ---
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'net' in checkpoint:
                state_dict = checkpoint['net']
            elif 'module' in checkpoint and isinstance(checkpoint['module'], dict):
                # 针对你当前文件的特殊处理
                state_dict = checkpoint['module']
            else:
                state_dict = checkpoint

            # --- 修正后的自适应加载逻辑 ---
            backbone_dict = backbone.state_dict()
            new_dict = {}

            # 1. 自动检测权重类型
            max_ckpt_idx = -1
            for k in state_dict.keys():
                if 'blocks.' in k:
                    parts = k.split('blocks.')[1].split('.')
                    if parts[0].isdigit():
                        max_ckpt_idx = max(max_ckpt_idx, int(parts[0]))

            # 如果最大索引超过 24，说明是原生 HiViT 权重(32层)，不需要偏移
            # 如果最大索引是 23，说明是标准 ViT 权重(24层)，需要 +8 偏移
            offset = 8 if max_ckpt_idx < 25 else 0
            print(f">>> [Debug] Max block index in ckpt: {max_ckpt_idx}, using offset: {offset}")

            for k, v in state_dict.items():
                # 清理前缀
                k_clean = k.replace('module.', '').replace('backbone.', '').replace('model.', '').replace('visual.',
                                                                                                          '')

                if k_clean.startswith('blocks.'):
                    parts = k_clean.split('.')
                    try:
                        idx = int(parts[1])
                        new_idx = idx + offset  # 应用自适应偏移
                        k_mapped = k_clean.replace(f'blocks.{idx}', f'blocks.{new_idx}')

                        if k_mapped in backbone_dict and v.shape == backbone_dict[k_mapped].shape:
                            new_dict[k_mapped] = v
                    except:
                        continue
                elif k_clean in backbone_dict and v.shape == backbone_dict[k_clean].shape:
                    new_dict[k_clean] = v

            msg = backbone.load_state_dict(new_dict, strict=False)
            print(f">>> [Phase 1] Smart Load: {len(new_dict)} keys matched.")

            if len(new_dict) == 0:
                print(">>> [Critical Warning] Still 0 keys matched! Check key format manually.")

        except Exception as e:
            print(f">>> [Error] Failed to load weights: {e}")
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