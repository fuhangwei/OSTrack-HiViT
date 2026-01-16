import math
import time
from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
import cv2
import os
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)

        # 1. 构建网络
        network = build_ostrack(params.cfg, training=False)

        # 2. 智能加载权重 (带自动缩放功能)
        self.load_pretrain(network, self.params.checkpoint)

        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.params.cfg.TEST.SEARCH_SIZE // self.params.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.debug = params.debug
        self.frame_id = 0

        # ProTeus-H: 历史记忆
        self.history_len = 16
        self.prompt_history = []

    def load_pretrain(self, network, checkpoint_path):
        print(f">>> [Tracker] Loading weights from: {checkpoint_path}")
        try:
            # 兼容不同版本的 PyTorch 加载
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

        state_dict = checkpoint.get('net', checkpoint)

        # 移除分布式训练产生的 'module.' 前缀
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_dict = network.state_dict()
        final_dict = {}

        for k, v in new_state_dict.items():
            if k in model_dict:
                if v.shape != model_dict[k].shape:
                    # 关键逻辑：处理 Phase 1 产生的 47x31 非对称 RPE 表
                    if 'relative_position_bias_table' in k:
                        src_len = v.shape[0]
                        dst_len = model_dict[k].shape[0]
                        num_heads = v.shape[1]

                        # 识别 1457 (47*31) 形状
                        if src_len == 1457:
                            src_h, src_w = 47, 31
                        elif src_len == 729:
                            src_h, src_w = 27, 27
                        else:
                            src_h = src_w = int(math.sqrt(src_len))

                        dst_size = int(math.sqrt(dst_len))

                        # 执行双线性插值缩放
                        v_img = v.permute(1, 0).view(1, num_heads, src_h, src_w)
                        v_resized = F.interpolate(v_img, size=(dst_size, dst_size),
                                                  mode='bicubic', align_corners=False)
                        final_dict[k] = v_resized.view(num_heads, dst_len).permute(1, 0)
                    else:
                        print(f"[Warning] Skip {k} due to shape mismatch: {v.shape} vs {model_dict[k].shape}")
                else:
                    final_dict[k] = v

        msg = network.load_state_dict(final_dict, strict=False)
        print(f">>> [Tracker] Smart Load Status: {msg}")

    def initialize(self, image, info: dict):
        self.state = info['init_bbox']
        self.frame_id = 0

        # 1. 采样并处理模板
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        self.z_dict1 = template

        # 2. 【核心修复】获取 Stage 3 的语义锚点
        with torch.no_grad():
            template_t = self.z_dict1.tensors.cuda()
            # 必须跑完 backbone 拿到 list [F1, F2, F3]
            features = self.network.backbone(template_t)
            f3 = features[-1]  # 取最后层语义特征 [1, 512, 8, 8]
            # 全局池化得到 1x1x512 的语义向量
            p_anchor = torch.mean(f3.flatten(2), dim=2, keepdim=True).transpose(1, 2)
            self.prompt_history = [p_anchor.detach()] * self.history_len

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # ProTeus-H: 拼接历史
        history_tensor = torch.cat(self.prompt_history, dim=1).cuda()

        # 修改 lib/test/tracker/ostrack.py 中的 track 函数
        with torch.no_grad():
            # 🚀 [修复语法] 显式传递参数，删除 ...
            out_dict = self.network(
                template=self.z_dict1.tensors.cuda(),
                search=search.tensors.cuda(),
                ce_template_mask=None,
                prompt_history=history_tensor
            )

            if 'p_obs' in out_dict:
                conf = out_dict['score_map'].max().item()
                # 🚀 [核心逻辑] 只有在看清楚时才更新观测，否则依靠 Mamba 惯性
                if conf > 0.45:
                    current_feat = out_dict['p_obs'].detach()
                else:
                    current_feat = out_dict['p_next'].detach()

                self.prompt_history.append(current_feat)
                self.prompt_history.pop(0)

        # 获取 Confidence
        uot_conf = out_dict.get('uot_confidence', torch.tensor(1.0)).item() if 'uot_confidence' in out_dict else 1.0

        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)

        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        return {
            "target_bbox": self.state,
            "best_score": response.max().item(),
            "uot_confidence": uot_conf
        }

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        for i in range(12):
            self.score_map = self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1]))
        return conv_features, enc_attn_weights, dec_attn_weights


def get_tracker_class():
    return OSTrack