import torch
from lib.models.ostrack.ostrack import build_ostrack
from lib.config.ostrack.config import cfg, update_config_from_file

# 1. 加载配置
config_path = "experiments/ostrack/hivit_base_256.yaml"
update_config_from_file(config_path)

# 2. 构建模型
model = build_ostrack(cfg, training=True).cuda()

# 3. 构造假数据 (Template 128x128, Search 320x320)
template = torch.randn(1, 3, 128, 128).cuda()
search = torch.randn(1, 3, 320, 320).cuda()

# 4. 前向传播
try:
    out = model(template, search)
    print("Forward Success!")
    print("Output Keys:", out.keys())

    # 5. 反向传播测试 (验证梯度)
    loss = out['score_map'].sum()
    loss.backward()
    print("Backward Success!")
except Exception as e:
    print("Error during model test:", e)