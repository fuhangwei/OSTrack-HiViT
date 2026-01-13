import os
import numpy as np

# ================= 配置区域 =================
# 你的结果文件存放路径 (请确认此路径正确)
RESULT_DIR = "output/test/tracking_results/ostrack/proteus_h_phase3"
# LaSOT 数据集根目录
DATASET_ROOT = "data/lasot"


# ===========================================

def load_robust_txt(path):
    """
    万能文本读取函数：
    可以同时处理逗号(,)、制表符(\t)、空格( )分隔的数值文件
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            # 1. 去除首尾空白
            line = line.strip()
            if not line: continue

            # 2. 将逗号和制表符都替换为空格
            line = line.replace(',', ' ').replace('\t', ' ')

            # 3. 按空格分割 (自动处理多个连续空格)
            parts = line.split()

            try:
                # 4. 转换为 float
                nums = [float(x) for x in parts]
                if len(nums) == 4:  # 确保是 x,y,w,h
                    data.append(nums)
            except ValueError:
                continue
    return np.array(data)


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-10)


def calculate_center_error(box1, box2):
    c1 = (box1[0] + box1[2] / 2, box1[1] + box1[3] / 2)
    c2 = (box2[0] + box2[2] / 2, box2[1] + box2[3] / 2)
    dist = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    return dist


def run_evaluation():
    if not os.path.exists(RESULT_DIR):
        print(f"❌ Error: Result path not found: {RESULT_DIR}")
        return

    txt_files = sorted([f for f in os.listdir(RESULT_DIR) if f.endswith('.txt')])
    print(f"📂 Found {len(txt_files)} result files. Starting evaluation...")

    video_aucs = []
    video_precs = []
    video_norm_precs = []

    valid_video_count = 0

    for txt_file in txt_files:
        video_name = txt_file.replace('.txt', '')

        # --- 寻找 GT 文件 ---
        category = video_name.split('-')[0]
        gt_path = os.path.join(DATASET_ROOT, category, video_name, "groundtruth.txt")

        if not os.path.exists(gt_path):
            # 尝试备选路径
            gt_path = os.path.join(DATASET_ROOT, video_name, "groundtruth.txt")
            if not os.path.exists(gt_path):
                # print(f"⚠️ GT not found for {video_name}")
                continue

        # --- 读取数据 (使用万能函数) ---
        try:
            pred_boxes = load_robust_txt(os.path.join(RESULT_DIR, txt_file))
            gt_boxes = load_robust_txt(gt_path)
        except Exception as e:
            print(f"⚠️ Error reading {video_name}: {e}")
            continue

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue

        # --- 对齐长度 & 计算指标 ---
        length = min(len(pred_boxes), len(gt_boxes))
        ious = []
        center_errors = []
        norm_errors = []

        valid_frames = 0
        for i in range(length):
            gt = gt_boxes[i]
            pred = pred_boxes[i]

            # 过滤无效 GT (NaN 或 0 尺寸)
            if np.any(np.isnan(gt)) or gt[2] <= 0 or gt[3] <= 0:
                continue

            valid_frames += 1

            # IoU
            ious.append(calculate_iou(pred, gt))

            # Center Error
            dist = calculate_center_error(pred, gt)
            center_errors.append(dist)

            # Norm Error
            gt_sz = np.sqrt((gt[2] + 1e-10) * (gt[3] + 1e-10))
            norm_errors.append(dist / gt_sz)

        if valid_frames > 0:
            ious = np.array(ious)
            center_errors = np.array(center_errors)
            norm_errors = np.array(norm_errors)

            # 1. Success (AUC)
            thresholds = np.linspace(0, 1, 101)
            succ_curve = [np.mean(ious > t) for t in thresholds]
            video_aucs.append(np.mean(succ_curve))

            # 2. Precision (20 px)
            video_precs.append(np.mean(center_errors <= 20))

            # 3. Norm Precision (0.5 AUC)
            norm_thresholds = np.linspace(0, 0.5, 51)
            norm_curve = [np.mean(norm_errors < t) for t in norm_thresholds]
            video_norm_precs.append(np.mean(norm_curve))

            valid_video_count += 1

    # --- 输出结果 ---
    if valid_video_count == 0:
        print("❌ Still no valid videos evaluated. Please check paths again.")
        return

    mean_auc = np.mean(video_aucs)
    mean_prec = np.mean(video_precs)
    mean_norm_prec = np.mean(video_norm_precs)

    print("\n" + "=" * 45)
    print(f"📊 LaSOT Evaluation Results ({valid_video_count} videos)")
    print("=" * 45)
    print(f"✅ Success Score (AUC)    : {mean_auc * 100:.2f}%")
    print(f"✅ Precision (20 px)      : {mean_prec * 100:.2f}%")
    print(f"✅ Norm Precision (P_norm): {mean_norm_prec * 100:.2f}%")
    print("=" * 45)


if __name__ == "__main__":
    run_evaluation()