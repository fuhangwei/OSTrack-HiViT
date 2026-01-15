import torch
import torch.nn as nn
import torch.nn.functional as F


class H_UOT_Observer(nn.Module):
    def __init__(self, dim=512, reg=0.05, alpha=1.0):
        super().__init__()
        self.reg = reg  # 熵正则化系数
        self.alpha = alpha  # 非平衡约束系数
        # 【关键】可学习的垃圾桶 Token，用于吸收遮挡区域的质量
        self.dustbin_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, prompt, visual_feats):
        """
        prompt: [B, 1, 512] (当前的提示词)
        visual_feats: [B, N, 512] (视觉特征图)
        """
        B, N, C = visual_feats.shape

        # 1. 扩展视觉特征，加入垃圾桶
        # [B, N+1, 512]
        vis_extended = torch.cat([visual_feats, self.dustbin_token.expand(B, 1, -1)], dim=1)

        # 2. 计算代价矩阵 (Cost Matrix)
        # 使用 1 - 余弦相似度
        norm_p = F.normalize(prompt, dim=-1)
        norm_v = F.normalize(vis_extended, dim=-1)
        # [B, 1, N+1]
        cost_matrix = 1.0 - torch.bmm(norm_p, norm_v.transpose(1, 2))

        # 3. 求解非平衡 Sinkhorn (简化版)
        # 我们希望找到一个传输矩阵 gamma
        K = torch.exp(-cost_matrix / self.reg)  # Gibbs kernel

        # 迭代更新 (通常 3-5 次即可)
        u = torch.ones((B, 1)).to(prompt.device)
        v = torch.ones((B, N + 1)).to(prompt.device)

        for _ in range(3):
            # 非平衡项限定：(alpha / (alpha + reg))
            u = (1.0 / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-6)) ** (self.alpha / (self.alpha + self.reg))
            v = (1.0 / (torch.bmm(u.unsqueeze(1), K).squeeze(1) + 1e-6)) ** (self.alpha / (self.alpha + self.reg))

        gamma = u.unsqueeze(-1) * K * v.unsqueeze(1)  # [B, 1, N+1]

        # 4. 生成观测 P_obs
        # 排除最后一列（即流向垃圾桶的部分）
        gamma_target = gamma[:, 0, :N]  # [B, N]
        total_mass = gamma_target.sum(dim=-1, keepdim=True)  # 传输的总质量 = 视觉置信度

        # 加权合成新 Prompt
        p_obs = torch.bmm(gamma_target.unsqueeze(1), visual_feats).squeeze(1)

        return p_obs, total_mass  # total_mass 将作为 Phase 4 门控的输入