import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaPredictor(nn.Module):
    def __init__(self, dim=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # Mamba 核心层：处理时序演化规律
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(dim)
        # 回归头：将 Mamba 输出转化为预测的下一帧 Prompt
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        """
        输入 x: [B, L, 512] (长度为 L 的历史 Prompt 序列)
        输出: [B, 512] (预测的下一帧 Prompt)
        """
        # x 形状: [batch, length, dim]
        x = self.mamba(x)
        x = self.norm(x)
        # 取序列的最后一个时间步作为预测依据
        return self.head(x[:, -1, :])