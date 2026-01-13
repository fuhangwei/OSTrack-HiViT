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
        # 训练时只需给历史加极微小的噪声，或者干脆不加
        if prompt_history is None:
            # Cold start: use anchor
            prompt_history = p_anchor.repeat(1, 16, 1)

        p_prior = self.predictor(prompt_history).unsqueeze(1)

        # 3. Backbone Inference (Frozen)
        if template.shape[3] != search.shape[3]:
            padding_width = search.shape[3] - template.shape[3]
            template_padded = F.pad(template, (0, padding_width, 0, 0))
        else:
            template_padded = template
        x_in = torch.cat([template_padded, search], dim=2)

        # 🔒 确保 Backbone 不更新
        # with torch.no_grad():
        #     results = self.backbone(x_in)
        results = self.backbone(x_in)
        f3 = results[-1]
        visual_feats = f3.flatten(2).transpose(1, 2)

        # 🛑 【已删除】 Visual Dropout (这是罪魁祸首)
        # if self.training and torch.rand(1).item() < 0.2: ...

        # 4. UOT + Synergy
        # 注意：visual_feats 需要 detach 吗？
        # 如果你想通过 UOT 训练 Backbone，则不 detach。
        # 但 Phase 3 通常冻结 Backbone，所以这里 visual_feats 视为常量。
        p_obs, confidence = self.observer(p_prior, visual_feats)  # Visual feats act as memory
        p_next = self.synergy(p_anchor, p_prior, p_obs, confidence)

        # 5. Fusion
        # 修正融合逻辑：确保 p_next 不会剧烈改变 visual_feats 的量级
        # 使用 tanh 门控，并乘以 visual_feats 的平均模长以保持尺度一致
        alpha = torch.tanh(self.fusion_alpha)

        # 广播 p_next 到每个像素
        feat_scale = visual_feats.abs().mean().detach()
        p_next_scaled = F.normalize(p_next, dim=-1) * feat_scale

        refined_feats = visual_feats + alpha * p_next_scaled

        out = self.forward_head(refined_feats)

        # Return history for next frame
        out['p_next'] = p_next
        out['p_anchor'] = p_anchor
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

        # 【修复点】添加 weights_only=False
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
            # 这里也要加 weights_only=False，以防万一
            model.predictor.load_state_dict(torch.load(mamba_path, map_location='cpu', weights_only=False))
            print("[Phase 3] Loaded Mamba Pre-trained Weights.")
        else:
            print("[Warning] Mamba weights not found! Using Random Init.")

    return model