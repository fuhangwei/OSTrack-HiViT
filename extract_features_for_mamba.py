import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from lib.models.ostrack.ostrack import build_ostrack
from lib.config.ostrack.config import cfg, update_config_from_file

# ================= 配置区域 =================
# 1. 指向 Phase 1 训练好的权重 (等 Phase 1 跑完 300 轮后，这里改成最终的 .pth)
#    目前先写成占位符，或者指向你为了测试跑出来的中间权重
CHECKPOINT_PATH = "output/phase1_full/checkpoints/train/ostrack/hivit_base_256/ProTeusH_ep0300.pth.tar"

# 2. 使用 LaSOT 数据集 (LaSOT 最适合 Mamba)
DATA_ROOT = "data/lasot"  # 你的软链接路径

# 3. 保存路径
SAVE_DIR = "data/mamba_pretrain_features"
os.makedirs(SAVE_DIR, exist_ok=True)

# 4. 采样间隔 (每隔几帧取一帧，减少数据量但保留时序)
SAMPLE_INTERVAL = 2
# ===========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    print(f"Loading config from experiments/ostrack/hivit_base_256.yaml...")
    update_config_from_file("experiments/ostrack/hivit_base_256.yaml")

    # 强制构建纯净的 HiViT (不带 Mamba/UOT)
    model = build_ostrack(cfg, training=False)

    if os.path.isfile(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint

        # 智能加载逻辑
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v

        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        print(f"[Warning] Checkpoint not found at {CHECKPOINT_PATH}. Using random init (ONLY FOR DEBUG).")

    model.to(device).eval()
    return model


# 图像预处理 (128x128 Template 模式)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def safe_crop(img, box):
    """
    安全裁剪：处理边界情况，如果框超出图像，自动进行 Padding
    box: [x, y, w, h]
    """
    x, y, w, h = map(int, box)
    cx, cy = x + w // 2, y + h // 2

    # 扩大 2 倍范围 (Template Factor)
    side = int(max(w, h) * 2.0)
    half_side = side // 2

    # 计算裁剪区域
    x1, y1 = cx - half_side, cy - half_side
    x2, y2 = x1 + side, y1 + side

    h_img, w_img, _ = img.shape

    # 如果在图像内部，直接裁
    if x1 >= 0 and y1 >= 0 and x2 <= w_img and y2 <= h_img:
        crop = img[y1:y2, x1:x2]
    else:
        # 如果超出边界，进行 Padding
        pad_x1 = max(0, -x1)
        pad_y1 = max(0, -y1)
        pad_x2 = max(0, x2 - w_img)
        pad_y2 = max(0, y2 - h_img)

        new_x1 = x1 + pad_x1
        new_y1 = y1 + pad_y1
        new_x2 = x2 - pad_x2
        new_y2 = y2 - pad_y2

        crop_part = img[new_y1:new_y2, new_x1:new_x2]

        # 使用图像均值进行填充 (更平滑)
        avg_color = np.mean(img, axis=(0, 1)).astype(int).tolist()
        crop = cv2.copyMakeBorder(crop_part, pad_y1, pad_y2, pad_x1, pad_x2,
                                  cv2.BORDER_CONSTANT, value=avg_color)

    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


@torch.no_grad()
def main():
    model = load_model()

    # 遍历 LaSOT 数据集结构 (LaSOT/category/sequence)
    # 注意：根据你的解压方式，可能有 category 层，也可能直接是 sequence
    # 这里假设标准结构：data/lasot/airplane/airplane-1/...

    # 1. 获取所有类别
    if not os.path.exists(DATA_ROOT):
        print(f"Error: {DATA_ROOT} does not exist.")
        return

    categories = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    print(f"Found {len(categories)} categories in LaSOT.")

    total_seqs = 0
    # 收集所有序列路径
    seq_list = []
    for cat in categories:
        cat_path = os.path.join(DATA_ROOT, cat)
        seqs = sorted([s for s in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, s))])
        for s in seqs:
            seq_list.append(os.path.join(cat_path, s))

    print(f"Total sequences found: {len(seq_list)}. Starting extraction...")

    # 遍历所有序列
    for seq_path in tqdm(seq_list):
        seq_name = os.path.basename(seq_path)
        save_file = os.path.join(SAVE_DIR, f"{seq_name}_feat.pt")

        # 如果已经提取过，跳过 (支持断点续传)
        if os.path.exists(save_file):
            continue

        # 读取 GT
        gt_file = os.path.join(seq_path, "groundtruth.txt")
        if not os.path.exists(gt_file): continue

        try:
            # LaSOT GT 分隔符可能是逗号也可能是空格
            gt_boxes = np.loadtxt(gt_file, delimiter=',')
        except:
            try:
                gt_boxes = np.loadtxt(gt_file, delimiter=' ')  # 尝试空格
            except:
                continue  # 读不出来就跳过

        img_dir = os.path.join(seq_path, "img")
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

        feature_seq = []

        # 遍历帧
        for i, img_name in enumerate(img_files):
            # 降采样
            if i % SAMPLE_INTERVAL != 0: continue
            if i >= len(gt_boxes): break

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            # 安全裁剪
            try:
                crop = safe_crop(img, gt_boxes[i])
                input_tensor = transform(crop).unsqueeze(0).to(device)  # [1, 3, 128, 128]

                # 提取特征
                # HiViT Backbone 返回 [f1, f2, f3]
                feats = model.backbone(input_tensor)
                f3 = feats[-1]  # [1, 512, 8, 8]

                # GAP (Global Average Pooling) -> [512]
                vec = F.adaptive_avg_pool2d(f3, (1, 1)).flatten()
                feature_seq.append(vec.cpu())  # 转到 CPU 节省显存
            except Exception as e:
                continue

        # 保存序列特征
        if len(feature_seq) > 20:  # 太短的序列不要
            feature_tensor = torch.stack(feature_seq)  # [L, 512]
            torch.save(feature_tensor, save_file)


if __name__ == "__main__":
    main()