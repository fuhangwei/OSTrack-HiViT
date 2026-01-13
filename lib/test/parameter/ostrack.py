from lib.test.utils import TrackerParams
import os
# --- 核心修改：导入配置加载工具 ---
from lib.config.ostrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    # ... 原有的 cfg 加载和 checkpoint 路径代码 ...
    params.cfg_file = os.path.join(prj_dir, 'experiments/ostrack/{}.yaml'.format(yaml_name))
    update_config_from_file(params.cfg_file)
    params.cfg = cfg
    # params.checkpoint = os.path.join(prj_dir, 'pretrained_models/ProTeusH_ep0060.pth.tar')
    # 【核心修改】直接在这里把路径写死，指向你 Phase 3 跑出来的权重
    # 注意：请确保文件名是你的实际文件名 (可能是 OSTrack_ep0060.pth.tar 或 ProTeusH_ep0060.pth.tar，去文件夹看一眼)
    # params.checkpoint = "/root/autodl-tmp/ostrack-hivit/output/phase3_final/checkpoints/train/ostrack/proteus_h_phase3/ProTeusH_ep0001.pth.tar"
    params.checkpoint = "/root/autodl-tmp/ostrack-hivit/output/phase1_full/checkpoints/train/ostrack/hivit_base_256/ProTeusH_ep0300.pth.tar"
    # --- 核心修改：添加缺失的测试开关参数 ---
    params.save_all_boxes = False  # 正式测试通常不需要保存所有候选框
    params.debug = False           # 关闭调试模式以提高速度
    # --------------------------------------

    params.template_size = 128
    params.template_factor = 2.0
    params.search_size = 256
    params.search_factor = 4.0
    params.out_size = 256
    params.v_color = (0, 255, 0)

    return params