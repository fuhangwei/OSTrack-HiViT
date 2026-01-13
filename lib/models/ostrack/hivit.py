import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
from timm.models.layers import to_2tuple


# --- 基础组件保持不变 ---

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


# --- 修改后的 HiViT 类 ---

class HiViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=512,
                 mlp_depth=3, depth=24, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm, ape=True, rpe=True,
                 patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        # ... (初始化代码保持不变) ...
        self.ape = ape
        self.rpe = rpe
        self.num_main_blocks = depth
        self.num_features = embed_dim
        self.use_checkpoint = use_checkpoint

        # ... (PatchEmbed, PosEmbed 初始化保持不变) ...
        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=mlvl_dims['4'], norm_layer=norm_layer if patch_norm else None)
        Hp, Wp = self.patch_embed.patches_resolution
        num_patches = Hp * Wp

        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        if rpe:
            coords = torch.stack(torch.meshgrid([torch.arange(Hp), torch.arange(Wp)]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :]).permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Hp - 1
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, 2 * mlp_depth + depth))
        self.blocks = nn.ModuleList()

        # ... (Block 构建循环保持不变) ...
        for i in range(mlp_depth):
            self.blocks.append(
                BlockWithRPE(Hp, mlvl_dims['4'], 0, bridge_mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                             next(dpr), rpe, norm_layer=norm_layer))
        self.blocks.append(PatchMerge(mlvl_dims['4'], norm_layer))
        for i in range(mlp_depth):
            self.blocks.append(
                BlockWithRPE(Hp, mlvl_dims['8'], 0, bridge_mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                             next(dpr), rpe, norm_layer=norm_layer))
        self.blocks.append(PatchMerge(mlvl_dims['8'], norm_layer))
        for i in range(depth):
            self.blocks.append(
                BlockWithRPE(Hp, mlvl_dims['16'], num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                             next(dpr), rpe, norm_layer=norm_layer))

        self.apply(self._init_weights)

        # 【新增】缓存变量，防止重复插值
        self.rpe_cached_shape = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.absolute_pos_embed.shape[1]
        if npatch == N and w == h: return self.absolute_pos_embed
        dim = x.shape[-1]
        w0, h0 = w + 0.1, h + 0.1
        patch_pos_embed = nn.functional.interpolate(
            self.absolute_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic', align_corners=False
        )
        return patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    def interpolate_rpe(self, Hp, Wp):
        """
        【优化版】动态插值，增加缓存机制
        """
        # 1. 检查缓存：如果尺寸没变，直接返回，不做任何计算！
        if self.rpe_cached_shape == (Hp, Wp):
            return

        target_h, target_w = 2 * Hp - 1, 2 * Wp - 1

        for blk in self.blocks:
            if hasattr(blk, 'attn') and blk.attn is not None and blk.attn.relative_position_bias_table is not None:
                table = blk.attn.relative_position_bias_table
                num_heads = table.shape[-1]
                L_old = table.shape[0]

                # 再次检查：如果 Parameter 形状已经对了，就不用插值
                if L_old == target_h * target_w:
                    continue

                S_old = int(math.sqrt(L_old))
                table_img = table.reshape(1, S_old, S_old, num_heads).permute(0, 3, 1, 2)
                table_img = F.interpolate(
                    table_img, size=(target_h, target_w),
                    mode='bicubic', align_corners=False
                )
                new_table = table_img.permute(0, 2, 3, 1).reshape(-1, num_heads)

                # 关键：更新 Parameter
                blk.attn.relative_position_bias_table = nn.Parameter(new_table)

        # 2. 更新缓存记录
        self.rpe_cached_shape = (Hp, Wp)
        # print(f"[HiViT Info] RPE interpolated to {target_h}x{target_w}")

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        features = []
        curr_block_idx = 0

        # Stage 1 & 2 (保持不变)
        for i in range(3):
            x = self.blocks[curr_block_idx](x)
            curr_block_idx += 1
        f1 = x.reshape(B, Hp, Wp, 4, 4, -1).permute(0, 5, 1, 3, 2, 4).reshape(B, -1, Hp * 4, Wp * 4).contiguous()
        features.append(f1)

        x = self.blocks[curr_block_idx](x)
        curr_block_idx += 1

        for i in range(3):
            x = self.blocks[curr_block_idx](x)
            curr_block_idx += 1
        f2 = x.reshape(B, Hp, Wp, 2, 2, -1).permute(0, 5, 1, 3, 2, 4).reshape(B, -1, Hp * 2, Wp * 2).contiguous()
        features.append(f2)

        x = self.blocks[curr_block_idx](x)
        curr_block_idx += 1

        # --- Stage 3: 优化后的调用 ---
        if self.rpe:
            self.interpolate_rpe(Hp, Wp)

        x = x.reshape(B, Hp * Wp, -1)

        if self.ape:
            x = x + self.interpolate_pos_encoding(x, Hp, Wp)
        x = self.pos_drop(x)

        rpe_index = None
        if self.rpe:
            # RPE Index 计算也可以缓存，这里先保留实时计算，因为它开销很小
            coords = torch.stack(torch.meshgrid([torch.arange(Hp), torch.arange(Wp)]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :]).permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Hp - 1
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            rpe_index = relative_coords.sum(-1).flatten().to(x.device)

        for i in range(self.num_main_blocks):
            blk = self.blocks[curr_block_idx]
            x = blk(x, rpe_index)
            curr_block_idx += 1

        f3 = x.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()
        features.append(f3)

        return features

    def forward(self, x):
        return self.forward_features(x)


def hivit_base(**kwargs):
    return HiViT(embed_dim=512, mlp_depth=3, depth=24, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4., **kwargs)