import torch
import torch.nn as nn


class BayesianSynergy(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        # 轻量级门控网络
        self.gating = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 注意：这里去掉了 Softmax，放到 forward 里做
        )

        # 【核心修复】恒等初始化 (Identity Initialization)
        # 强制让初始权重偏向 Anchor (w1)，屏蔽 Mamba (w2) 和 UOT (w3)
        # 我们修改最后一层 Linear 的 bias
        last_layer = self.gating[-1]

        # 将 weight 设为极小值，消除输入(confidence)的影响
        nn.init.zeros_(last_layer.weight)

        # 设置 bias: [High, Low, Low]
        # 这样 Softmax 出来接近 [1.0, 0.0, 0.0]
        # 比如 [5.0, -5.0, -5.0] -> Softmax -> [0.9999, 0.00005, 0.00005]
        custom_bias = torch.tensor([5.0, -5.0, -5.0])
        last_layer.bias.data.copy_(custom_bias)

        print(">>> [Synergy] Initialized with Identity Mapping (Trust Anchor 100%)")

    def forward(self, p_anchor, p_mamba, p_uot, confidence):
        """
        confidence: 来自 UOT 的 total_mass [B, 1, 1]
        """
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