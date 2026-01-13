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
# [在 ostrack.py 开头引入]
import torch.nn.functional as F  # 确保引入 F

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

        # 🔴 [删除旧代码]
        # self.spatial_align = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        # 🟢 [新增代码] 使用门控通道调制 (FiLM 机制的变体)
        # 将 p_next (B, 1, 512) 映射为缩放因子 (Scale) 和 偏置 (Shift)
        self.fusion_map = nn.Linear(512, 512 * 2)
        # 初始化为 "无操作" 状态 (Scale=1, Shift=0)
        nn.init.constant_(self.fusion_map.weight, 0)
        nn.init.constant_(self.fusion_map.bias, 0)

        # 这一行可以保留，用于控制整体力度
        self.fusion_alpha = nn.Parameter(torch.tensor(0.0))

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    # [在 ProTeusH 类 forward 函数中修改]
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
                # 增大噪声幅度 (0.02 -> 0.05)，模拟这一秒内的变化
                prompt_history = prompt_history + torch.randn_like(prompt_history) * 0.05

        # 🟢 [关键修复] Mamba 预训练时用了 F.normalize，这里必须加上！
        # 否则输入的模长差异会导致 Mamba 输出乱码
        prompt_history_norm = F.normalize(prompt_history, p=2, dim=-1)
        p_prior = self.predictor(prompt_history_norm).unsqueeze(1)

        # 3. Backbone Inference (保持不变)
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

        # 4. UOT + Synergy (保持不变)
        p_obs, confidence = self.observer(p_prior, visual_feats)
        p_next = self.synergy(p_anchor, p_prior, p_obs, confidence)

        # ============================================================
        # 🟢 [关键修复] 替换错误的 Attention，改用通道调制
        # ============================================================

        # p_next: [B, 1, 512]
        # 生成调制参数: [B, 1, 1024] -> split -> scale, shift: [B, 1, 512]
        style = self.fusion_map(p_next)
        scale, shift = style.chunk(2, dim=-1)

        # 门控系数
        alpha = torch.tanh(self.fusion_alpha)

        # 调制公式: Visual * (1 + Scale) + Shift
        # 这样 p_next 可以按通道增强或抑制 Visual Feature
        modulated_feats = visual_feats * (1.0 + alpha * torch.sigmoid(scale)) + alpha * shift

        # 残差连接 (可选，但推荐保留原始特征底座)
        refined_feats = visual_feats + modulated_feats

        # ============================================================

        out = self.forward_head(refined_feats)
        # 🔴 [关键] 必须把 p_next 等传出去，如果你还要用 Loss (虽然建议去掉 REG Loss)
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

    # [在 build_ostrack 函数中修改]
    if training:
        mamba_path = os.path.join(pretrained_path, "mamba_phase2.pth")
        if os.path.exists(mamba_path):
            model.predictor.load_state_dict(torch.load(mamba_path, map_location='cpu', weights_only=False))
            print("[Phase 3] Loaded Mamba Pre-trained Weights.")

            # 🔴 [删除] 不要冻结！
            # for p in model.predictor.parameters():
            #     p.requires_grad = False
            # print(">>> [Phase 3 Strategy] Mamba Predictor is FROZEN...")

            # 🟢 [新增] 确保解冻，允许 Mamba 适应新的 Backbone 特征分布
            for p in model.predictor.parameters():
                p.requires_grad = True
            print(">>> [Phase 3 Strategy] Mamba Predictor UNLOCKED for Co-adaptation.")

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