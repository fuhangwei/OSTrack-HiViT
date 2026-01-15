import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaPredictor(nn.Module):
    def __init__(self, dim=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # Mamba 核心块
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(dim)
        # 映射头：预测下一帧的 Prompt 向量
        self.head = nn.Linear(dim, dim)

    def forward(self, p_history):
        """
        p_history: [B, L, 512] 历史特征序列
        return: [B, 512] 预测的 Prior 向量
        """
        # x shape: [B, L, dim]
        x = self.mamba(p_history)
        x = self.norm(x)
        # 取最后一个时间步进行预测
        p_prior = self.head(x[:, -1, :])
        return p_prior