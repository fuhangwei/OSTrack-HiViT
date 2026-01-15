class BayesianSynergy(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        # 极轻量级的门控网络
        self.gating = nn.Sequential(
            nn.Linear(2, 16),  # 输入：UOT质量, Mamba方差 (或简单用1)
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, p_init, p_prior, p_obs, uot_mass):
        """
        uot_mass: 从 H-UOT 传来的质量 [B, 1]
        """
        # 构造置信度特征向量
        # 这里你可以加入 Mamba 的内部不确定性，或者简单只用 UOT 质量
        conf_feats = torch.cat([uot_mass, torch.ones_like(uot_mass)], dim=-1)

        weights = self.gating(conf_feats)  # [B, 3]
        w_init, w_prior, w_obs = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        # 贝叶斯动态融合
        p_next = w_init * p_init + w_prior * p_prior + w_obs * p_obs

        return p_next, weights