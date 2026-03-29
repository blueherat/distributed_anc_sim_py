#!/usr/bin/env python3
"""Smoke-run script to validate the training notebook core in a fast, safe way.
- Uses a small subset (max 64 samples), batch_size=8, epochs=1.
- Runs in the active Python environment (we'll activate `myenv` in the terminal before running).
"""
import sys
import time
import random
from pathlib import Path
import argparse

import numpy as np

try:
    import h5py
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, Subset
except Exception as e:
    print('IMPORT_ERROR:', e)
    raise

# CLI arguments to control a fast smoke-run without changing env packages
parser = argparse.ArgumentParser(description='Smoke-run for ANC notebook core')
parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
parser.add_argument('--subset-size', type=int, default=64)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--max-batches', type=int, default=0, help='0 means no limit')
args = parser.parse_args()

seed = 20260329
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if args.device == 'cpu':
    device = torch.device('cpu')
elif args.device == 'cuda':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Smoke-run device:', device)

H5_PATH = Path('python_impl') / 'python_scripts' / 'cfxlms_qc_dataset_cross_500_seeded.h5'
if not H5_PATH.exists():
    print('ERROR: HDF5 dataset not found at', H5_PATH)
    sys.exit(2)

class HDF5ANCConditionedDataset(Dataset):
    def __init__(self, h5_path, device=torch.device('cpu')):
        super().__init__()
        self.h5 = h5py.File(str(h5_path), 'r')
        self.n_samples = int(self.h5['processed/gcc_phat'].shape[0])
        self.device = device
        W_comp = np.asarray(self.h5['processed/global_svd/W_components'], dtype=np.float32)
        W_mean = np.asarray(self.h5['processed/global_svd/W_mean'], dtype=np.float32)
        # move to device for fast matmul in loss (ok since num_workers=0 below)
        self.W_components = torch.from_numpy(W_comp).to(self.device)
        self.W_mean = torch.from_numpy(W_mean).to(self.device)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        gcc = np.asarray(self.h5['processed/gcc_phat'][idx], dtype=np.float32)   # (3,129)
        psd = np.asarray(self.h5['processed/psd_features'][idx], dtype=np.float32) # (129,)
        psd = psd.reshape(1, -1)
        x_p = np.vstack([gcc, psd]).astype(np.float32)  # (4,129)
        x_s = np.asarray(self.h5['processed/S_pca_coeffs'][idx], dtype=np.float32)  # (32,)
        y_c = np.asarray(self.h5['processed/W_pca_coeffs'][idx], dtype=np.float32)  # (32,)
        return torch.from_numpy(x_p), torch.from_numpy(x_s), torch.from_numpy(y_c)

class MIMO_Conditioned_ANCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.ln1 = nn.LayerNorm((16, 65))
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.ln2 = nn.LayerNorm((32, 33))
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.ln3 = nn.LayerNorm((64, 17))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.s_fc1 = nn.Linear(32, 64)
        self.s_ln1 = nn.LayerNorm(64)
        self.s_fc2 = nn.Linear(64, 64)
        self.s_ln2 = nn.LayerNorm(64)
        self.f_fc1 = nn.Linear(128, 128)
        self.f_ln1 = nn.LayerNorm(128)
        self.f_drop = nn.Dropout(0.2)
        self.f_fc2 = nn.Linear(128, 64)
        self.f_fc3 = nn.Linear(64, 32)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                try:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                except Exception:
                    pass
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_p, x_s):
        # x_p: [B,4,129]
        z = self.conv1(x_p)
        z = F.gelu(self.ln1(z))
        z = self.conv2(z)
        z = F.gelu(self.ln2(z))
        z = self.conv3(z)
        z = F.gelu(self.ln3(z))
        z = self.pool(z)
        z_p = z.view(z.size(0), -1)
        z_s = F.gelu(self.s_ln1(self.s_fc1(x_s)))
        z_s = F.gelu(self.s_ln2(self.s_fc2(z_s)))
        zf = torch.cat([z_p, z_s], dim=1)
        zf = F.gelu(self.f_ln1(self.f_fc1(zf)))
        zf = self.f_drop(zf)
        zf = F.gelu(self.f_fc2(zf))
        c_pred = self.f_fc3(zf)
        return c_pred


def main():
    try:
        full_ds = HDF5ANCConditionedDataset(str(H5_PATH), device=device)
        n = len(full_ds)
        subset_size = min(64, n)
        if subset_size < 4:
            print('Too few samples to run smoke test:', n)
            return
        indices = list(range(subset_size))
        split = int(subset_size * 0.8)
        train_idx = indices[:split]
        val_idx = indices[split:]
        train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=8, shuffle=False, num_workers=0)

        print('subset_size', subset_size, 'train_batches', len(train_loader), 'val_batches', len(val_loader))
        print('W_components shape', full_ds.W_components.shape, 'W_mean shape', full_ds.W_mean.shape)

        model = MIMO_Conditioned_ANCNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # quick training: 1 epoch
        epochs = 1
        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            batches = 0
            t0 = time.time()
            for xb_p, xb_s, yb_c in train_loader:
                xb_p = xb_p.to(device=device, dtype=torch.float32)
                xb_s = xb_s.to(device=device, dtype=torch.float32)
                yb_c = yb_c.to(device=device, dtype=torch.float32)
                optimizer.zero_grad()
                c_pred = model(xb_p, xb_s)
                loss_c = F.mse_loss(c_pred, yb_c)
                W_pred_flat = torch.matmul(c_pred, full_ds.W_components) + full_ds.W_mean
                W_true_flat = torch.matmul(yb_c, full_ds.W_components) + full_ds.W_mean
                loss_w = F.mse_loss(W_pred_flat, W_true_flat)
                loss = loss_c + 0.1 * loss_w
                loss.backward()
                optimizer.step()
                running += float(loss.item())
                batches += 1
            train_loss = running / max(1, batches)
            # validation
            model.eval()
            vrunning = 0.0
            vbatches = 0
            with torch.no_grad():
                for xb_p, xb_s, yb_c in val_loader:
                    xb_p = xb_p.to(device=device, dtype=torch.float32)
                    xb_s = xb_s.to(device=device, dtype=torch.float32)
                    yb_c = yb_c.to(device=device, dtype=torch.float32)
                    c_pred = model(xb_p, xb_s)
                    loss_c = F.mse_loss(c_pred, yb_c)
                    W_pred_flat = torch.matmul(c_pred, full_ds.W_components) + full_ds.W_mean
                    W_true_flat = torch.matmul(yb_c, full_ds.W_components) + full_ds.W_mean
                    loss_w = F.mse_loss(W_pred_flat, W_true_flat)
                    loss = loss_c + 0.1 * loss_w
                    vrunning += float(loss.item())
                    vbatches += 1
            val_loss = vrunning / max(1, vbatches)
            print(f'Epoch {epoch} train_loss={train_loss:.6e} val_loss={val_loss:.6e} time={time.time()-t0:.1f}s')
        print('Smoke-run complete')
    except Exception as e:
        print('ERROR during smoke-run:', e)
        raise

if __name__ == '__main__':
    main()
