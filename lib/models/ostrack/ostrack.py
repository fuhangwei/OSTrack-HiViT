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

        # Phase 3 组件
        self.predictor = MambaPredictor(dim=512)
        self.observer = UOTObserver(dim=512)
        self.synergy = BayesianSynergy(dim=512)

        # 零初始化阀门 (完美继承 Phase 1 性能的关键)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.0))

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                prompt_history=None,
                **kwargs):

        B = template.shape[0]

        # 1. Anchor (Detached! 极其重要，不要让梯度回传给 backbone)
        with torch.no_grad():
            z_patch, _ = self.backbone.patch_embed(template)
            p_anchor = torch.mean(z_patch.reshape(B, -1, 512), dim=1, keepdim=True)
            p_anchor = p_anchor.detach()  # 🔒 锁死 Anchor

        # 2. Mamba Prediction
        # 在 ProTeusH.forward 中修改训练分支逻辑
        if prompt_history is None:
            # 模拟训练阶段
            prompt_history = p_anchor.repeat(1, 16, 1)

            if self.training:
                # 🚀 根本性改进：训练时引入 10% 的时序扰动噪声
                # 强迫 Backbone 学会纠正那些“稍微有点偏”的时序特征，而不是只依赖完美的 anchor
                noise = torch.randn_like(prompt_history) * 0.02
                prompt_history = prompt_history + noise

        p_prior = self.predictor(prompt_history).unsqueeze(1)

        # 3. Backbone Inference
        if template.shape[3] != search.shape[3]:
            padding_width = search.shape[3] - template.shape[3]
            template_padded = F.pad(template, (0, padding_width, 0, 0))
        else:
            template_padded = template
        x_in = torch.cat([template_padded, search], dim=2)

        # Backbone 正常前向传播 (允许梯度回传)
        results = self.backbone(x_in)
        f3 = results[-1]
        visual_feats = f3.flatten(2).transpose(1, 2)

        # 4. UOT + Synergy
        p_obs, confidence = self.observer(p_prior, visual_feats)
        p_next = self.synergy(p_anchor, p_prior, p_obs, confidence)

        # 5. 根本性融合重构：Uncertainty-Weighted Fusion
        alpha = torch.tanh(self.fusion_alpha)

        # 🚀 关键：利用 Synergy 计算出的 confidence (最优传输代价导出的置信度)
        # 当观测与预测冲突很大时，confidence 趋于 0，自动关闭时序分支对视觉特征的影响
        dynamic_alpha = alpha * torch.sigmoid(confidence)

        feat_scale = visual_feats.abs().mean().detach()
        p_next_scaled = F.normalize(p_next, dim=-1) * feat_scale

        # 使用 dynamic_alpha 进行残差融合
        refined_feats = visual_feats + dynamic_alpha * p_next_scaled

        out = self.forward_head(refined_feats)

        # Return history for next frame
        out['p_next'] = p_next
        out['p_anchor'] = p_anchor
        out['p_obs'] = p_obs  # <--- 关键修复：必须传出这个观测值
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
            # 只有加上这就话，Backbone 才会乖乖去适应 Mamba，而不是两个一起乱跑
            for p in model.predictor.parameters():
                p.requires_grad = False
            print(">>> [Phase 3 Strategy] Mamba Predictor is FROZEN (Acting as Anchor).")
            # ❄️❄️❄️ 结束 ❄️❄️❄️
        else:
            print("[Warning] Mamba weights not found! Using Random Init.")

        # -------------------------------------------------------------
        # 🚀【SOTA 必加】强制解冻 Backbone
        # -------------------------------------------------------------
        # 原因：如果不加这个，Backbone 可能因为加载了预训练权重而保持 requires_grad=False。
        # 当 ltr_trainer 创建 optimizer 时，它会检查 parameters()。
        # 如果此时是 False，optimizer 就永远不会包含这些参数，导致你以为在微调，其实没微调。
        for n, p in model.backbone.named_parameters():
            p.requires_grad = True
        print(">>> [Phase 3 SOTA Strategy] FORCED Backbone requires_grad = True. Ready for Full Finetuning.")
        # -------------------------------------------------------------

    return model