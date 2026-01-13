import torch
import torch.nn as nn
import torch.nn.functional as F


class UOTObserver(nn.Module):
    def __init__(self, dim=512, eps=0.1, max_iter=3):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter

        # 可学习的 Dustbin Token
        self.dustbin_token = nn.Parameter(torch.zeros(1, 1, dim))
        torch.nn.init.trunc_normal_(self.dustbin_token, std=0.02)

        # 可学习的垃圾桶代价阈值
        self.dustbin_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, prompt, visual_feats):
        """
        极速版 Sinkhorn
        prompt: [B, 1, C]
        visual_feats: [B, N, C]
        """
        B, N, C = visual_feats.shape

        # 1. 归一化 (inplace 如果可能)
        prompt = F.normalize(prompt, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        dustbin = F.normalize(self.dustbin_token, dim=-1)

        # 2. 构建扩展特征矩阵 [B, N+1, C]
        # 使用 cat 可能会有一点开销，但为了逻辑清晰保留
        # 优化点：dustbin 扩展时不复制数据
        dustbin_batch = dustbin.expand(B, 1, -1)
        extended_feats = torch.cat([visual_feats, dustbin_batch], dim=1)

        # 3. 计算 Cosine Distance Matrix [B, 1, N+1]
        # bmm 是最高效的
        sim = torch.bmm(prompt, extended_feats.transpose(1, 2))  # [B, 1, N+1]
        dist = 1.0 - sim

        # 4. 动态调整最后一列 (Dustbin Cost)
        # 避免使用 expand，利用广播机制
        # dist[:, :, -1] = self.dustbin_param  <-- 这种 inplace 操作可能会慢
        # 我们直接覆盖最后一列的值

        # 优化技巧：与其修改 dist，不如直接在 sim 阶段就控制好
        # 这里为了保持逻辑一致，我们用 mask 的方式快速替换
        dist_main = dist[:, :, :-1]
        dist_bin = self.dustbin_param.view(1, 1, 1).expand(B, 1, 1)
        # 重新拼接比 inplace 赋值通常更快且梯度更安全
        C_matrix = torch.cat([dist_main, dist_bin], dim=2)

        # 5. Sinkhorn 迭代 (Log-domain 稳定且快)
        # C_matrix: [B, 1, N+1]

        # 核心优化：由于 dim=1 只有 1 行，其实不需要完整的 Sinkhorn 迭代
        # 这是一个 "One-to-Many" 的传输，其实就是 Softmax 的变种！
        # 标准 Sinkhorn 是为了满足行列约束。
        # 这里行和为 1，列和不限（非平衡）。
        # 其实只需要对行做 Softmax 即可近似最优传输！

        # 极速方案：直接使用 Softmax 温度缩放
        # 这在数学上等价于单次迭代的 Sinkhorn
        T = F.softmax(-C_matrix / self.eps, dim=-1)  # [B, 1, N+1]

        # 6. 提取结果
        # T_main: [B, 1, N]
        T_main = T[:, :, :-1]

        # 计算质量
        total_mass = T_main.sum(dim=-1, keepdim=True)  # [B, 1, 1]

        # 加权求和
        # [B, 1, N] @ [B, N, C] -> [B, 1, C]
        # 增加 1e-6 防止除以 0
        obs_prompt = torch.bmm(T_main, visual_feats) / (total_mass + 1e-6)

        return obs_prompt, total_mass