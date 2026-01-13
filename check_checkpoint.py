import torch
import os
from lib.models.ostrack import build_ostrack
from lib.config.ostrack.config import cfg, update_config_from_file

# ================= 配置 =================
# 指向你的 Phase 1 权重文件
CKPT_PATH = "output/phase1_full/checkpoints/train/ostrack/hivit_base_256/ProTeusH_ep0300.pth.tar"
# 指向你的 Phase 3 配置文件
CONFIG_FILE = "experiments/ostrack/proteus_h_phase3.yaml"


# =======================================

def check():
    print(f">>> 正在检查权重文件: {CKPT_PATH}")

    if not os.path.exists(CKPT_PATH):
        print("❌ 文件不存在！请检查路径！")
        return

    # 1. 加载权重文件
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    state_dict = ckpt['net'] if 'net' in ckpt else ckpt

    print(f"✅ 权重加载成功。包含 {len(state_dict)} 个参数。")

    # 检查 Box Head 是否存在
    head_keys = [k for k in state_dict.keys() if 'box_head' in k]
    if len(head_keys) > 0:
        print(f"✅ Box Head 参数存在！共 {len(head_keys)} 个。")
        print(f"   示例 Key: {head_keys[0]}")
    else:
        print("❌ 致命错误：权重文件中没有 'box_head' 相关参数！")
        print("   这意味着 Phase 1 根本没保存检测头，或者你用了 ImageNet 权重。")
        return

    # 2. 构建 Phase 3 模型
    print("\n>>> 正在构建 Phase 3 模型...")
    update_config_from_file(CONFIG_FILE)
    model = build_ostrack(cfg, training=False)  # training=False 避免加载 Mamba 干扰视线

    model_keys = list(model.state_dict().keys())
    print(f"✅ 模型构建成功。")

    # 3. 模拟匹配
    print("\n>>> 开始匹配测试...")
    matched_keys = []
    missing_keys = []

    for k_model in model_keys:
        # 模拟代码里的加载逻辑
        k_ckpt = "module." + k_model  # 假设 ckpt 是 DDP 保存的
        k_ckpt_noddp = k_model

        if k_ckpt in state_dict:
            matched_keys.append(k_model)
        elif k_ckpt_noddp in state_dict:
            matched_keys.append(k_model)
        else:
            missing_keys.append(k_model)

    print(f"✅ 成功匹配: {len(matched_keys)} 个参数")
    print(f"⚠️ 未匹配: {len(missing_keys)} 个参数")

    # 检查 Box Head 是否匹配
    head_missing = [k for k in missing_keys if 'box_head' in k]
    if len(head_missing) == 0:
        print("🎉 恭喜！Box Head 参数可以完美加载！")
    else:
        print(f"❌ 警告：Box Head 参数加载失败！缺失 {len(head_missing)} 个。")
        print(f"   缺失示例: {head_missing[:5]}")


if __name__ == "__main__":
    check()