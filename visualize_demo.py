import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from lib.models.ostrack.ostrack import build_ostrack
from lib.config.ostrack.config import cfg, update_config_from_file

# ================= 配置区域 =================
# 1. 你的配置文件路径
CONFIG_FILE = "experiments/ostrack/hivit_base_256.yaml"
# 2. 刚刚训练好的 Checkpoint (Ep 60)
CHECKPOINT = "output/phase3/checkpoints/train/ostrack/hivit_base_256/ProTeusH_ep0060.pth.tar"
# 3. 想测试的视频文件 (找一个有遮挡的!)
VIDEO_FILE = "demo_videos/basketball_occlusion.mp4"


# ===========================================

def get_subwindow(im, pos, model_sz, original_sz, avg_chans):
    """标准的裁剪与预处理函数"""
    if isinstance(pos, float): pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1

    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, left_pad + c:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1),
        int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1),
        int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    return im_patch


@torch.no_grad()
def run_demo():
    # 1. Setup Model
    update_config_from_file(CONFIG_FILE)
    model = build_ostrack(cfg, training=False)

    print(f"Loading checkpoint: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(checkpoint['net'], strict=True)
    model.cuda().eval()

    # 2. Setup Video
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取第一帧进行初始化 (手动画框或使用硬编码)
    ret, frame = cap.read()
    # 这里为了演示，我们假设你手动选框，或者硬编码一个初始框
    # cv2.selectROI 会弹出一个窗口让你画框，按回车确认
    print(">>> Please draw the initial bounding box and press ENTER")
    init_rect = cv2.selectROI("Tracker", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Tracker")

    # Init State
    state = init_rect  # x, y, w, h
    target_pos = np.array([state[0] + state[2] / 2, state[1] + state[3] / 2])
    target_sz = np.array([state[2], state[3]])

    # Template Preprocessing
    img_mean = np.array([128, 128, 128]).reshape(1, 1, 3)
    template_img = get_subwindow(frame, target_pos, 128, np.sqrt(np.prod(target_sz)) * 2.0, img_mean)
    template_tensor = torch.tensor(template_img).cuda().float().permute(2, 0, 1).unsqueeze(0) / 255.0
    template_tensor -= torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    template_tensor /= torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    # History Memory (for Mamba)
    prompt_history = None

    video_writer = cv2.VideoWriter('mamba_demo_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                   (frame.shape[1], frame.shape[0]))

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Search Preprocessing
        search_factor = 4.0
        search_sz = 256
        search_area = np.sqrt(np.prod(target_sz)) * search_factor

        # Crop Search Region
        search_img = get_subwindow(frame, target_pos, search_sz, search_area, img_mean)
        search_tensor = torch.tensor(search_img).cuda().float().permute(2, 0, 1).unsqueeze(0) / 255.0
        search_tensor -= torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        search_tensor /= torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        # ================= INFERENCE =================
        out = model(template_tensor, search_tensor, prompt_history=prompt_history)

        # Update Prompt History
        p_next = out['p_next']  # [B, 1, C]
        if prompt_history is None:
            # 初始化：填满 16 帧历史
            prompt_history = out['p_anchor'].repeat(1, 16, 1)
        else:
            prompt_history = torch.cat([prompt_history[:, 1:], p_next.detach()], dim=1)

        # Get Prediction
        pred_boxes = out['pred_boxes'].view(-1, 4)
        # Convert back to image coordinates
        pred_box = (pred_boxes[0] * search_area).cpu().numpy()
        pred_x, pred_y, pred_w, pred_h = pred_box

        # Update Target State
        target_pos[0] += pred_x - search_area / 2
        target_pos[1] += pred_y - search_area / 2
        target_sz[0] = target_sz[0] * (pred_w / search_area) * (search_sz / 128)  # Approximation
        target_sz[1] = target_sz[1] * (pred_h / search_area) * (search_sz / 128)

        # ================= VISUALIZATION =================
        # 核心：获取 UOT 置信度
        confidence = out['uot_confidence'].item()  # Scalar 0.0 ~ 1.0

        # 画框
        x, y = int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2)
        w, h = int(target_sz[0]), int(target_sz[1])

        # 颜色逻辑：高置信度=绿，低置信度(Mamba接管)=红
        color = (0, 255, 0)  # Green
        status_text = "VISUAL TRACKING"
        if confidence < 0.6:  # 阈值可调
            color = (0, 0, 255)  # Red
            status_text = "OCCLUSION DETECTED (Mamba Inertia)"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        # 画置信度条 (Dashboard)
        bar_x, bar_y = 50, 50
        bar_w, bar_h = 200, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

        cv2.putText(frame, f"Conf: {confidence:.2f}", (bar_x + bar_w + 10, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (bar_x, bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('ProTeus-H Demo', frame)
        video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()