import math
import os
import torch
import torch.nn.functional as F
from torch import nn

from lib.models.layers.head import build_box_head
from lib.models.ostrack.hivit import hivit_base
from lib.models.ostrack.mamba_predictor import MambaPredictor
from lib.models.ostrack.uot_observer import UOTObserver
from lib.models.ostrack.synergy_update import BayesianSynergy
from lib.utils.box_ops import box_xyxy_to_cxcywh


class ProTeusH(nn.Module):
    def __init__(self, transformer, box_head, head_type="CENTER"):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.head_type = head_type

        # Phase 3 核心组件
        self.predictor = MambaPredictor(dim=512)
        self.observer = UOTObserver(dim=512)
        self.synergy = BayesianSynergy(dim=512)

        # 🚀 [核心补齐] 空间对齐层：解决全图污染问题
        self.spatial_align = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        # 零初始化融合参数，保护视觉特征
        self.fusion_alpha = nn.Parameter(torch.tensor(0.0))

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, template, search, ce_template_mask=None, ce_keep_rate=None, prompt_history=None, **kwargs):
        B = template.shape[0]

        # 1. Anchor 锁死
        with torch.no_grad():
            z_patch, _ = self.backbone.patch_embed(template)
            p_anchor = torch.mean(z_patch.reshape(B, -1, 512), dim=1, keepdim=True).detach()

        # 2. 对齐训练/推理分布
        if prompt_history is None:
            prompt_history = p_anchor.repeat(1, 16, 1)
            if self.training:
                prompt_history = prompt_history + torch.randn_like(prompt_history) * 0.02
        p_prior = self.predictor(prompt_history).unsqueeze(1)

        # 3. Backbone Inference
        if template.shape[3] != search.shape[3]:
            padding_width = search.shape[3] - template.shape[3]
            template_padded = F.pad(template, (0, padding_width, 0, 0))
        else:
            template_padded = template
        x_in = torch.cat([template_padded, search], dim=2)

        # 🚀 [修复2] 删除不支持的 CE 参数，直接调用 forward
        results = self.backbone(x_in)
        f3 = results[-1]  # [B, 512, Hp, Wp]

        # 🚀 [修复1] 必须先展平并转置，才能正确执行 Token 切片
        # f3 shape: [B, 512, H, W] -> flatten -> [B, 512, N] -> transpose -> [B, N, 512]
        f3_flat = f3.flatten(2).transpose(1, 2)
        visual_feats = f3_flat[:, -self.feat_len_s:]  # 这才是真正的 Search Tokens [B, 256, 512]

        # 4. UOT + Synergy
        p_obs, confidence = self.observer(p_prior, visual_feats)
        p_next = self.synergy(p_anchor, p_prior, p_obs, confidence)

        # 4. UOT + Synergy
        p_obs, confidence = self.observer(p_prior, visual_feats)
        p_next = self.synergy(p_anchor, p_prior, p_obs, confidence)

        # ============================================================
        # 🚀【核心修复】改用乘法融合 (Channel Modulation)
        # ============================================================

        # 1. 空间对齐 (保留你的 Cross-Attention)
        # 这里的 aligned_temporal 拥有和 visual_feats 一样的空间维度 [B, N, C]
        aligned_temporal, _ = self.spatial_align(visual_feats, p_next, p_next)

        # 2. 计算门控系数 (Gating)
        # alpha 控制 Mamba 的介入程度，初始为 0
        alpha = torch.tanh(self.fusion_alpha)

        # 3. 生成通道调制图 (Modulation Map)
        # 我们希望 aligned_temporal 变成一个 [0, 1] 之间的调节系数
        # sigmoid 确保它是一个比例，而不是绝对值
        modulation = torch.sigmoid(aligned_temporal)

        # 4. 乘法融合 (The Magic)
        # 公式：Visual * (1 + alpha * Modulation)
        # 物理含义：
        # - 如果 Visual 是 0 (背景)，结果还是 0 (保护了背景纯净度)。
        # - 如果 alpha 是 0，结果就是 Visual (完美退化回 Phase 1)。
        # - Mamba 只能“增强”或“抑制”已有的视觉特征，不能凭空创造噪声。

        refined_feats = visual_feats * (1.0 + alpha * modulation)

        # ============================================================

        out = self.forward_head(refined_feats)
        out.update({'p_next': p_next, 'p_anchor': p_anchor, 'p_obs': p_obs})
        return out

    def forward_head(self, cat_feature):
        # 这里的 cat_feature 已经是切片后的 search tokens
        # 即使这里再次切片也没关系，因为长度正好是 feat_len_s
        enc_opt = cat_feature[:, -self.feat_len_s:]

        # [B, N, C] -> [B, C, N] -> [B, C, H, W]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
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

    backbone = hivit_base()
    box_head = build_box_head(cfg, 512)
    model = ProTeusH(backbone, box_head, head_type=cfg.MODEL.HEAD.TYPE)

    if cfg.MODEL.PRETRAIN_FILE and training:
        ckpt_path = cfg.MODEL.PRETRAIN_FILE
        print(f">>> [Phase 3] Loading weights from: {ckpt_path}")

        # 添加 weights_only=False
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint

        model_dict = model.state_dict()
        new_dict = {}
        load_count = 0

        for k, v in state_dict.items():
            k_clean = k.replace('module.', '')
            if k_clean in model_dict:
                if v.shape == model_dict[k_clean].shape:
                    new_dict[k_clean] = v
                    load_count += 1

        if load_count == 0:
            raise ValueError("!!! No weights loaded! Check your checkpoint path or keys!")

        msg = model.load_state_dict(new_dict, strict=False)
        print(f">>> [Phase 3] Successfully loaded {load_count} keys.")

        # 确认 Box Head 是否加载
        head_loaded = any("box_head" in k for k in new_dict.keys())
        if not head_loaded:
            raise ValueError("!!! Box Head weights NOT detected! Training will FAIL.")
        print(">>> [Phase 3] Box Head weights LOADED.")

    if training:
        mamba_path = os.path.join(pretrained_path, "mamba_phase2.pth")
        if os.path.exists(mamba_path):
            model.predictor.load_state_dict(torch.load(mamba_path, map_location='cpu', weights_only=False))
            print("[Phase 3] Loaded Mamba Pre-trained Weights.")

            # ❄️❄️❄️ 【必须新增】冻结 Mamba，贯彻 Anchor 策略 ❄️❄️❄️
            for p in model.predictor.parameters():
                p.requires_grad = False
            print(">>> [Phase 3 Strategy] Mamba Predictor is FROZEN (Acting as Anchor).")
            # ❄️❄️❄️ 结束 ❄️❄️❄️
        else:
            print("[Warning] Mamba weights not found! Using Random Init.")

        # -------------------------------------------------------------
        # 🚀【SOTA 必加】强制解冻 Backbone
        # -------------------------------------------------------------
        # 确保 optimizer 能注册到参数
        for n, p in model.backbone.named_parameters():
            p.requires_grad = True
        print(">>> [Phase 3 SOTA Strategy] FORCED Backbone requires_grad = True. Ready for Full Finetuning.")
        # -------------------------------------------------------------

    return model