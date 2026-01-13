import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UOTObserver(nn.Module):
    def __init__(self, dim=512):
        super().__init__()

        # 【改进1】自适应温度系数 (Adaptive Temperature)
        # 初始 eps ≈ 0.1 (log(0.1) ≈ -2.3)
        # 让网络自己学习特征分布的疏密程度
        self.log_eps = nn.Parameter(torch.tensor(math.log(0.1)))

        # 可学习的 Dustbin Token (垃圾桶语义中心)
        self.dustbin_token = nn.Parameter(torch.zeros(1, 1, dim))
        torch.nn.init.trunc_normal_(self.dustbin_token, std=0.02)

        # 可学习的垃圾桶代价阈值
        # 初始设为 0.5 (Cosine Distance 范围 0~2)
        # 意味着：当相似度 < 0.5 时，倾向于认为是遮挡
        self.dustbin_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, prompt, visual_feats):
        """
        极速版 Sinkhorn (Softmax Approximation)
        prompt: [B, 1, C]
        visual_feats: [B, N, C]
        """
        B, N, C = visual_feats.shape

        # 1. 归一化 (L2 Norm)
        prompt = F.normalize(prompt, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        dustbin = F.normalize(self.dustbin_token, dim=-1)

        # 2. 构建扩展特征矩阵 [B, N+1, C]
        # 拼接 Visual Tokens 和 Dustbin Token
        dustbin_batch = dustbin.expand(B, 1, -1)
        extended_feats = torch.cat([visual_feats, dustbin_batch], dim=1)

        # 3. 计算 Cosine Distance Matrix [B, 1, N+1]
        # sim: [B, 1, N+1] -> range [-1, 1]
        sim = torch.bmm(prompt, extended_feats.transpose(1, 2))

        # dist: [B, 1, N+1] -> range [0, 2]
        dist = 1.0 - sim

        # 4. 动态调整最后一列 (Dustbin Cost)
        # 用可学习的 dustbin_param 替换最后一列的距离
        dist_main = dist[:, :, :-1]
        dist_bin = self.dustbin_param.view(1, 1, 1).expand(B, 1, 1)

        # [B, 1, N+1]
        C_matrix = torch.cat([dist_main, dist_bin], dim=2)

        # 5. 计算传输计划 (Transport Plan)
        # 使用动态温度系数 eps
        # clamp 防止除以 0 或数值溢出
        eps = torch.exp(self.log_eps).clamp(min=1e-3, max=1.0)

        # Softmax 归一化 (等价于 One-Step Sinkhorn)
        # T: [B, 1, N+1]
        T = F.softmax(-C_matrix / eps, dim=-1)

        # 6. 提取结果
        # T_main: [B, 1, N] (分配给视觉区域的权重)
        T_main = T[:, :, :-1]

        # total_mass: [B, 1, 1] (置信度)
        # 如果遮挡严重，大部分权重会流向最后一列，total_mass 会变小
        total_mass = T_main.sum(dim=-1, keepdim=True)

        # 7. 加权求和得到观测特征
        # [B, 1, N] @ [B, N, C] -> [B, 1, C]
        # 增加 1e-6 防止除以 0
        obs_prompt = torch.bmm(T_main, visual_feats) / (total_mass + 1e-6)

        return obs_prompt, total_mass