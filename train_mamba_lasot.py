import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lib.models.ostrack.mamba_predictor import MambaPredictor


class LaSOTMambaDataset(Dataset):
    def __init__(self, root_dir, seq_len=64, stride=10):
        """
        seq_len: 历史窗口长度（64帧约合2秒多，足以学习惯性）
        stride: 采样步长（防止连续帧太像导致过拟合）
        """
        self.seq_len = seq_len
        self.samples = []

        # 搜索所有 .npy 文件
        files = glob.glob(os.path.join(root_dir, "*/*.npy"))
        print(f">>> Found {len(files)} sequence files. Preparing windows...")

        for f in files:
            feat = np.load(f)  # [Num_Frames, 512]
            if len(feat) <= seq_len:
                continue

            # 建立滑动窗口：用前 seq_len 帧预测下一帧
            for i in range(0, len(feat) - seq_len, stride):
                x = feat[i: i + seq_len]
                y = feat[i + seq_len]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def train_mamba():
    # --- 配置 ---
    DATA_DIR = "data/mamba_features_lasot"
    SAVE_PATH = "pretrained_models/mamba_predictor_lasot.pth"
    os.makedirs("pretrained_models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    dataset = LaSOTMambaDataset(DATA_DIR, seq_len=64)
    if len(dataset) == 0:
        print("Error: No samples found! Please wait for extraction to finish more sequences.")
        return

    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)
    print(f">>> Dataset prepared. Total windows: {len(dataset)}")

    # 2. 初始化 Mamba 预测器
    model = MambaPredictor(dim=512).to(device)

    # 3. 优化配置
    criterion = nn.MSELoss()  # 回归任务使用 MSE
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 4. 训练循环
    print(">>> Starting Mamba Training...")
    model.train()
    for epoch in range(50):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f"Epoch [{epoch + 1}/50], Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # 每 10 轮存一次
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved: {SAVE_PATH}")


if __name__ == "__main__":
    train_mamba()