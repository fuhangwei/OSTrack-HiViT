import torch
import torch.nn as nn
import torch.nn.functional as F  # 【新增】需要用到 F


class BayesianSynergy(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        # 轻量级门控网络
        self.gating = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

        # 【核心修复】恒等初始化 (Identity Initialization)
        last_layer = self.gating[-1]
        nn.init.zeros_(last_layer.weight)

        # 设置 bias: [High, Low, Low] -> Softmax -> [1.0, 0.0, 0.0]
        custom_bias = torch.tensor([5.0, -5.0, -5.0])
        last_layer.bias.data.copy_(custom_bias)

        print(">>> [Synergy] Initialized with Identity Mapping (Trust Anchor 100%)")

    def forward(self, p_anchor, p_mamba, p_uot, confidence):
        """
        confidence: 来自 UOT 的 total_mass [B, 1, 1]
        """

        # 🚀【新增必杀技】强制归一化 (LayerNorm)
        # 解决 "Scale Mismatch" 问题，确保 Mamba 的微弱信号能被同等对待
        # 注意：我们对最后一个维度 (dim=512) 做归一化
        p_anchor = F.layer_norm(p_anchor, p_anchor.shape[-1:])
        p_mamba = F.layer_norm(p_mamba, p_mamba.shape[-1:])
        p_uot = F.layer_norm(p_uot, p_uot.shape[-1:])

        # 计算 Logits
        logits = self.gating(confidence.squeeze(-1))  # [B, 3]

        # 手动做 Softmax
        weights = torch.softmax(logits, dim=-1)

        w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        # 动态加权融合
        p_next = w1.unsqueeze(-1) * p_anchor + \
                 w2.unsqueeze(-1) * p_mamba + \
                 w3.unsqueeze(-1) * p_uot

        return p_next