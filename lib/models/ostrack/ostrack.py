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

        # 🟢 [SOTA 新增] 模长对齐层 (防止 Phase 2/3 分布失配)
        self.norm_fusion = nn.LayerNorm(512)

        # 🟢 [新增代码] 使用门控通道调制 (FiLM 机制)
        # 将 p_next (B, 1, 512) 映射为缩放因子 (Scale) 和 偏置 (Shift)
        self.fusion_map = nn.Linear(512, 512 * 2)
        # 初始化为 "无操作" 状态 (Scale=1, Shift=0)
        nn.init.constant_(self.fusion_map.weight, 0)
        nn.init.constant_(self.fusion_map.bias, 0)

        # 融合力度控制参数
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

        # 2. 对齐训练/推理分布 & 🔴 [关键修复：输入归一化]
        if prompt_history is None:
            # 训练时的“假时序”增强：增加更大的噪声来模拟运动，防止过拟合静态
            prompt_history = p_anchor.repeat(1, 16, 1)
            if self.training:
                # 增大噪声幅度 (0.05)，模拟帧间变化
                prompt_history = prompt_history + torch.randn_like(prompt_history) * 0.05

        # 🟢 [关键修复] Mamba 输入必须归一化
        prompt_history_norm = F.normalize(prompt_history, p=2, dim=-1)
        p_prior = self.predictor(prompt_history_norm).unsqueeze(1)

        # 3. Backbone Inference
        if template.shape[3] != search.shape[3]:
            padding_width = search.shape[3] - template.shape[3]
            template_padded = F.pad(template, (0, padding_width, 0, 0))
        else:
            template_padded = template
        x_in = torch.cat([template_padded, search], dim=2)

        results = self.backbone(x_in)
        f3 = results[-1]
        f3_flat = f3.flatten(2).transpose(1, 2)
        visual_feats = f3_flat[:, -self.feat_len_s:]

        # 4. UOT + Synergy
        p_obs, confidence = self.observer(p_prior, visual_feats)
        p_next = self.synergy(p_anchor, p_prior, p_obs, confidence)

        # ============================================================
        # 🟢 [关键修复] 通道调制融合
        # ============================================================

        # 1. 先进行 LayerNorm，消除模长波动
        p_next_norm = self.norm_fusion(p_next)

        # 2. 生成调制参数
        style = self.fusion_map(p_next_norm)
        scale, shift = style.chunk(2, dim=-1)

        # 3. 门控系数
        alpha = torch.tanh(self.fusion_alpha)

        # 4. 调制公式: Visual * (1 + Scale) + Shift
        modulated_feats = visual_feats * (1.0 + alpha * torch.sigmoid(scale)) + alpha * shift

        # 残差连接
        refined_feats = visual_feats + modulated_feats

        # ============================================================

        out = self.forward_head(refined_feats)
        out.update({'p_next': p_next, 'p_anchor': p_anchor, 'p_obs': p_obs})
        return out

    def forward_head(self, cat_feature):
        enc_opt = cat_feature[:, -self.feat_len_s:]
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
        # weights_only=False 适配不同 torch 版本
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

        # 严格检查 Box Head
        head_loaded = any("box_head" in k for k in new_dict.keys())
        if not head_loaded:
            # 有时候 Phase 1 的 checkpoint 键名可能有差异，这里做一个容错或者报错
            print("!!! Warning: Box Head weights might be missing. Check your Phase 1 checkpoint.")

        model.load_state_dict(new_dict, strict=False)
        print(f">>> [Phase 3] Loaded {load_count} keys from Phase 1.")

    if training:
        mamba_path = os.path.join(pretrained_path, "mamba_phase2.pth")
        if os.path.exists(mamba_path):
            model.predictor.load_state_dict(torch.load(mamba_path, map_location='cpu', weights_only=False))
            print("[Phase 3] Loaded Mamba Pre-trained Weights.")

            # 🟢 [SOTA 策略] 必须解冻 Mamba
            for p in model.predictor.parameters():
                p.requires_grad = True
            print(">>> [Phase 3 Strategy] Mamba Predictor UNLOCKED.")
        else:
            print("[Warning] Mamba weights not found! Using Random Init.")

        # 🟢 [SOTA 策略] 强制解冻 Backbone
        for n, p in model.backbone.named_parameters():
            p.requires_grad = True
        print(">>> [Phase 3 SOTA Strategy] FORCED Backbone requires_grad = True.")

    return model