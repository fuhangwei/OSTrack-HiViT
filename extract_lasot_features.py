import torch
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from lib.models.ostrack.ostrack import build_ostrack
from lib.config.ostrack.config import cfg, update_config_from_file
import gc

# --- 配置 ---
LASOT_ROOT = "/root/autodl-tmp/ostrack-hivit/data/lasot"
SAVE_ROOT = "data/mamba_features_lasot"
CHECKPOINT = "output/phase1_final/checkpoints/train/ostrack/hivit_base_256/ProTeusH_ep0060.pth.tar"

# --- 初始化 ---
update_config_from_file("experiments/ostrack/hivit_base_256.yaml")
model = build_ostrack(cfg, training=False)

print(f">>> Loading checkpoint from {CHECKPOINT}")
checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
state_dict = checkpoint.get('net', checkpoint)

# 处理 RPE 尺寸不匹配（防止 load_state_dict 报错）
model_dict = model.state_dict()
for k, v in state_dict.items():
    if k in model_dict and v.shape != model_dict[k].shape and 'relative_position_bias_table' in k:
        print(f"  [Resize] {k}: {model_dict[k].shape} -> {v.shape}")
        parts = k.split('.')
        sub = model
        for p in parts[:-1]: sub = getattr(sub, p)
        setattr(sub, parts[-1], torch.nn.Parameter(torch.zeros_like(v)))

model.load_state_dict(state_dict, strict=False)
model.cuda().eval()
print(">>> Checkpoint loaded successfully!")

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def extract_seq(img_dir, gt_path, save_path):
    if os.path.exists(save_path): return
    try:
        gt = np.loadtxt(gt_path, delimiter=',')
    except:
        gt = np.loadtxt(gt_path)

    img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    seq_feats = []

    for i, box in enumerate(gt):
        if i >= len(img_names): break
        try:
            img = Image.open(os.path.join(img_dir, img_names[i])).convert('RGB')
            x, y, w, h = box
            cx, cy, s = x + w / 2, y + h / 2, max(w, h) * 2.0
            crop = img.crop((cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2))
            input_tensor = preprocess(crop).unsqueeze(0).cuda()

            # 提取特征
            results = model.backbone(input_tensor)
            feat_vec = torch.mean(results[-1], dim=(2, 3)).squeeze().cpu().numpy()
            seq_feats.append(feat_vec)
        except Exception as e:
            if i < 3: print(f"\n[Frame Error] {e}")  # 打印出具体报错
            continue

    if len(seq_feats) > 0:
        np.save(save_path, np.array(seq_feats))
        return True
    return False


# --- 全量遍历 ---
categories = sorted([d for d in os.listdir(LASOT_ROOT) if os.path.isdir(os.path.join(LASOT_ROOT, d))])
for cat in tqdm(categories, desc="Extraction"):
    cat_p = os.path.join(LASOT_ROOT, cat)
    save_cat_p = os.path.join(SAVE_ROOT, cat)
    os.makedirs(save_cat_p, exist_ok=True)

    seqs = sorted([d for d in os.listdir(cat_p) if os.path.isdir(os.path.join(cat_p, d))])
    for s in seqs:
        if extract_seq(os.path.join(cat_p, s, "img"), os.path.join(cat_p, s, "groundtruth.txt"),
                       os.path.join(save_cat_p, f"{s}.npy")):
            # 成功保存一个序列后打印一下
            pass
    torch.cuda.empty_cache();
    gc.collect()