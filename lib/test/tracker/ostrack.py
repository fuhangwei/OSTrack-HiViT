import math
import torch
import torch.nn.functional as F
import numpy as np
from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)

        # 1. 强制加载 Phase 1 权重并保持 1457 RPE
        checkpoint = torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('net', checkpoint)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_dict = network.state_dict()
        for k, v in new_state_dict.items():
            if k in model_dict and 'relative_position_bias_table' in k:
                if v.shape != model_dict[k].shape:
                    parts = k.split('.')
                    sub = network
                    for p in parts[:-1]: sub = getattr(sub, p)
                    setattr(sub, parts[-1], torch.nn.Parameter(v.clone()))
        network.load_state_dict(new_state_dict, strict=False)

        self.network = network.cuda().eval()
        self.preprocessor = Preprocessor()
        self.state = None

        # 2. 严格对齐 4.0 Factor
        self.search_factor = 4.0
        self.search_size = 256
        self.template_size = 128

        # 3. 响应窗口
        self.output_window = hann2d(torch.tensor([16, 16]).long(), centered=True).cuda()

    def initialize(self, image, info: dict):
        self.state = info['init_bbox']
        # 初始长宽记录，用于后续防止漂移
        self.base_w = self.state[2]
        self.base_h = self.state[3]

        # 提取 128x128 模板，不进行任何填充，保持原始尺寸
        z_patch_arr, _, z_amask_arr = sample_target(image, self.state, 2.0, output_sz=self.template_size)
        self.z_tensor = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        # 1. 裁剪搜索区域 (4.0 倍)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.search_factor,
                                                                output_sz=self.search_size)
        search_tensor = self.preprocessor.process(x_patch_arr, x_amask_arr).tensors

        # 2. 推理：【核心对齐点】
        # 在 forward 内部，template(128) 和 search(256) 会拼成 384x256，与训练完全一致
        with torch.no_grad():
            out_dict = self.network(template=self.z_tensor, search=search_tensor)

        # 3. 响应图处理
        response = out_dict['score_map'] * self.output_window
        score_max, _ = torch.max(response.view(-1), dim=0)

        # 4. 解码坐标 (返回 0~1 的比例)
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        best_box = pred_boxes.view(-1, 4)[0].cpu().numpy()  # [cx, cy, w, h]

        # 5. 【高精度还原公式】
        # 计算当前搜索区域在原图中的实际边长
        search_area_size_in_img = self.search_size / resize_factor

        # 获取搜索区域在原图中的中心 (即上一帧的目标中心)
        prev_cx = self.state[0] + 0.5 * self.state[2]
        prev_cy = self.state[1] + 0.5 * self.state[3]

        # 计算预测点相对于搜索区域中心 (128, 128) 的偏移量，并还原到原图尺度
        dx_real = (best_box[0] * 256.0 - 128.0) / resize_factor
        dy_real = (best_box[1] * 256.0 - 128.0) / resize_factor

        # 计算预测的绝对宽高
        w_real = (best_box[2] * 256.0) / resize_factor
        h_real = (best_box[3] * 256.0) / resize_factor

        # 6. 【极简平滑】尺度更新率 0.1，这是 OSTrack 默认值
        if self.state is not None:
            w_real = self.state[2] * 0.9 + w_real * 0.1
            h_real = self.state[3] * 0.9 + h_real * 0.1

        # 7. 更新状态
        cx_real = prev_cx + dx_real
        cy_real = prev_cy + dy_real
        self.state = clip_box([cx_real - w_real / 2, cy_real - h_real / 2, w_real, h_real], H, W, margin=10)

        return {"target_bbox": self.state, "best_score": score_max.item()}


def get_tracker_class():
    return OSTrack