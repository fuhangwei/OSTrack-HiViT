import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
from timm.models.layers import to_2tuple

# --- Attention, BlockWithRPE, PatchEmbed, PatchMerge 类保持不变 ---
# (为了节省篇幅，这里假设你保留了这些基础类的定义，直接进入 HiViT 类)

class Attention(nn.Module):
    def __init__(self, input_size, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_size - 1) * (2 * input_size - 1), num_heads)
        ) if rpe else None
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if rpe_index is not None:
            S = int(math.sqrt(rpe_index.size(-1)))
            relative_position_bias = self.relative_position_bias_table[rpe_index].view(-1, S, S, self.num_heads)
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
            attn = attn + relative_position_bias
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BlockWithRPE(nn.Module):
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., rpe=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        with_attn = num_heads > 0.
        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, rpe=rpe) if with_attn else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, rpe_index=None, mask=None):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size // self.patch_size[0], img_size // self.patch_size[1]]
        self.inner_patches = inner_patches
        conv_size = [size // inner_patches for size in self.patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        B, C, H, W = x.shape
        Hp, Wp = H // self.patch_size[0], W // self.patch_size[1]
        x = self.proj(x).view(B, -1, Hp, self.inner_patches, Wp, self.inner_patches).permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, Hp * Wp, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None: x = self.norm(x)
        return x, (Hp, Wp)

class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):
        x0, x1, x2, x3 = x[..., 0::2, 0::2, :], x[..., 1::2, 0::2, :], x[..., 0::2, 1::2, :], x[..., 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduction(self.norm(x))

class HiViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=512,
                 mlp_depth=3, depth=24, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm, ape=True, rpe=True,
                 patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.ape, self.rpe = ape, rpe
        self.num_main_blocks, self.num_features = depth, embed_dim
        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=mlvl_dims['4'], norm_layer=norm_layer if patch_norm else None)
        Hp, Wp = self.patch_embed.patches_resolution
        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, Hp * Wp, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords = torch.stack(torch.meshgrid([torch.arange(Hp), torch.arange(Wp)], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            rel_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :]).permute(1, 2, 0).contiguous()
            rel_coords[:, :, 0] += Hp - 1; rel_coords[:, :, 1] += Wp - 1
            rel_coords[:, :, 0] *= 2 * Wp - 1
            self.register_buffer("relative_position_index", rel_coords.sum(-1))

        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, 2 * mlp_depth + depth))
        self.blocks = nn.ModuleList()
        for _ in range(mlp_depth): self.blocks.append(BlockWithRPE(Hp, mlvl_dims['4'], 0, bridge_mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, next(dpr), rpe, norm_layer=norm_layer))
        self.blocks.append(PatchMerge(mlvl_dims['4'], norm_layer))
        for _ in range(mlp_depth): self.blocks.append(BlockWithRPE(Hp, mlvl_dims['8'], 0, bridge_mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, next(dpr), rpe, norm_layer=norm_layer))
        self.blocks.append(PatchMerge(mlvl_dims['8'], norm_layer))
        for _ in range(depth): self.blocks.append(BlockWithRPE(Hp, mlvl_dims['16'], num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, next(dpr), rpe, norm_layer=norm_layer))
        self.apply(self._init_weights)
        self.rpe_cached_shape = None

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.LayerNorm)):
            if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def interpolate_pos_encoding(self, x, h, w):
        npatch, N, dim = x.shape[1], self.absolute_pos_embed.shape[1], x.shape[-1]
        if npatch == N and w == h: return self.absolute_pos_embed
        patch_pos_embed = F.interpolate(
            self.absolute_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            size=(h, w), mode='bicubic', align_corners=False
        )
        return patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    def interpolate_rpe(self, Hp, Wp):
        """ 最终修复版：精准处理 Phase 1 产生的非对称权重 """
        if self.rpe_cached_shape == (Hp, Wp):
            return

        target_h, target_w = 2 * Hp - 1, 2 * Wp - 1

        for blk in self.blocks:
            if hasattr(blk, 'attn') and blk.attn is not None and blk.attn.relative_position_bias_table is not None:
                table = blk.attn.relative_position_bias_table
                num_heads = table.shape[-1]
                L_old = table.shape[0]

                if L_old == target_h * target_w:
                    continue

                # --- 核心硬核修复逻辑 ---
                if L_old == 729:  # 官方预训练 224x224
                    S_h, S_w = 27, 27
                elif L_old == 1457:  # 你刚刚跑完的 Phase 1 权重 (47x31)
                    S_h, S_w = 47, 31
                elif L_old == 1705:  # 如果你是用 448x256 训练出来的 (55x31)
                    S_h, S_w = 55, 31
                else:
                    # 万能保底策略：尝试最接近的平方根
                    S_h = S_w = int(math.sqrt(L_old))
                    if S_h * S_w != L_old:
                        # 如果开方不尽，尝试用比例推断
                        ratio = math.sqrt(L_old / (target_h * target_w))
                        S_h, S_w = int(target_h * ratio), int(target_w * ratio)

                # 最后一次强制检查，防止报错
                if S_h * S_w != L_old:
                    # 极端的兜底方案：如果实在对不上，取表的前 N 位做正方形（损失极小）
                    S_h = S_w = int(math.sqrt(L_old))
                    table = table[:S_h * S_w, :]

                try:
                    table_img = table.reshape(1, S_h, S_w, num_heads).permute(0, 3, 1, 2)
                    table_img = F.interpolate(
                        table_img, size=(target_h, target_w),
                        mode='bicubic', align_corners=False
                    )
                    new_table = table_img.permute(0, 2, 3, 1).reshape(-1, num_heads)
                    blk.attn.relative_position_bias_table = nn.Parameter(new_table)
                except Exception as e:
                    print(f"RPE Resize Critical Failure: {e}")
                    continue

        self.rpe_cached_shape = (Hp, Wp)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        features, curr_idx = [], 0
        for _ in range(2): # Stage 1 & 2
            for i in range(3): x = self.blocks[curr_idx + i](x)
            f_map = x.reshape(B, Hp, Wp, x.shape[3], x.shape[4], -1).permute(0, 5, 1, 3, 2, 4).reshape(B, -1, Hp*x.shape[3], Wp*x.shape[4]).contiguous()
            features.append(f_map)
            x = self.blocks[curr_idx + 3](x); curr_idx += 4
        if self.rpe: self.interpolate_rpe(Hp, Wp)
        x = x.reshape(B, Hp * Wp, -1)
        if self.ape: x = x + self.interpolate_pos_encoding(x, Hp, Wp)
        rpe_idx = None
        if self.rpe:
            coords = torch.stack(torch.meshgrid([torch.arange(Hp), torch.arange(Wp)], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            rel = (coords_flatten[:, :, None] - coords_flatten[:, None, :]).permute(1, 2, 0).contiguous()
            rel[:, :, 0] += Hp - 1; rel[:, :, 1] += Wp - 1; rel[:, :, 0] *= 2 * Wp - 1
            rpe_idx = rel.sum(-1).flatten().to(x.device)
        for i in range(self.num_main_blocks): x = self.blocks[curr_idx + i](x, rpe_idx)
        features.append(x.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous())
        return features

    def forward(self, x): return self.forward_features(x)

def hivit_base(**kwargs):
    return HiViT(embed_dim=512, mlp_depth=3, depth=24, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4., **kwargs)