import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from mamba_ssm import Mamba
from tqdm import tqdm

# ================= 配置区域 =================
FEATURE_DIR = "data/mamba_pretrain_features"
SAVE_PATH = "pretrained_models/mamba_phase2.pth"
os.makedirs("pretrained_models", exist_ok=True)

SEQ_LEN = 16
BATCH_SIZE = 256  # 4090D 可以开大点
EPOCHS = 50
LR = 5e-4  # 稍微调大一点学习率，Cosine Loss 收敛需要大步长


# ===========================================

class MambaPredictor(nn.Module):
    def __init__(self, dim=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.mamba(x)
        x = self.norm(x)
        return self.head(x[:, -1, :])


class FeatureDataset(Dataset):
    def __init__(self, feature_dir, seq_len=16):
        self.seq_len = seq_len
        self.samples = []
        if not os.path.exists(feature_dir): return
        files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith('.pt')]
        print(f"Loading features from {len(files)} sequences...")
        for f in tqdm(files):
            try:
                feats = torch.load(f, map_location='cpu', weights_only=False)
                if feats.shape[0] <= seq_len: continue
                # 步长设为 2，增加数据量
                for i in range(0, feats.shape[0] - seq_len, 2):
                    self.samples.append((feats[i: i + seq_len], feats[i + seq_len]))
            except:
                continue
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train():
    device = torch.device("cuda")
    dataset = FeatureDataset(FEATURE_DIR, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model = MambaPredictor().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(">>> Start Mamba Pre-training (Cosine Loss)...")
    min_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            # 【双保险】: 先归一化输入，消除模长噪声，让 Mamba 专心学方向
            inputs = F.normalize(inputs, p=2, dim=-1)
            # targets 不需要手动归一化，因为 F.cosine_similarity 会自动做

            optimizer.zero_grad()
            preds = model(inputs)

            # Loss: 1 - Cosine Similarity
            loss = 1.0 - F.cosine_similarity(preds, targets, dim=-1).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"Phase 2 Done. Best Cosine Loss: {min_loss:.6f}")


if __name__ == "__main__":
    train()