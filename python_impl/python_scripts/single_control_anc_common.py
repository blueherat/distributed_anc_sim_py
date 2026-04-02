from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def standardize_array(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)).astype(np.float32)


def destandardize_array(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)).astype(np.float32)


def stats_from_train(arr: np.ndarray, train_idx: np.ndarray, eps: float = 1.0e-6) -> dict[str, np.ndarray]:
    train_arr = np.asarray(arr[train_idx], dtype=np.float32)
    mean = train_arr.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_arr.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(eps))
    return {"mean": mean, "std": std}


class PCAProjector:
    def __init__(self, components: np.ndarray):
        self.components = np.asarray(components, dtype=np.float32)
        if self.components.ndim != 2:
            raise ValueError("components must be a 2D array.")
        self.output_dim = int(self.components.shape[0])
        self.input_dim = int(self.components.shape[1])

    @classmethod
    def fit(cls, x_train: np.ndarray, latent_dim: int) -> "PCAProjector":
        arr = np.asarray(x_train, dtype=np.float32)
        _, _, vt = np.linalg.svd(arr, full_matrices=False)
        return cls(vt[: min(int(latent_dim), int(vt.shape[0]))])

    @classmethod
    def from_state_dict(cls, state: dict) -> "PCAProjector":
        return cls(np.asarray(state["components"], dtype=np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        return (arr @ self.components.T).astype(np.float32)

    def inverse_np(self, z: np.ndarray) -> np.ndarray:
        arr = np.asarray(z, dtype=np.float32)
        return (arr @ self.components).astype(np.float32)

    def inverse_torch(self, z: torch.Tensor) -> torch.Tensor:
        comp = torch.as_tensor(self.components, dtype=z.dtype, device=z.device)
        return z @ comp

    def state_dict(self) -> dict:
        return {
            "output_dim": self.output_dim,
            "input_dim": self.input_dim,
            "components": self.components.tolist(),
        }


class ConvXPEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 24, kernel_size=5, stride=2, padding=2)
        self.gn1 = nn.GroupNorm(1, 24)
        self.conv2 = nn.Conv1d(24, 48, kernel_size=5, stride=2, padding=2)
        self.gn2 = nn.GroupNorm(1, 48)
        self.conv3 = nn.Conv1d(48, 96, kernel_size=3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(1, 96)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(96, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.gn1(self.conv1(x)))
        z = F.gelu(self.gn2(self.conv2(z)))
        z = F.gelu(self.gn3(self.conv3(z)))
        z = self.pool(z).squeeze(-1)
        return F.gelu(self.ln(self.fc(z)))


class XSBranch(nn.Module):
    def __init__(self, in_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleControlANCNet(nn.Module):
    def __init__(self, x_s_dim: int, latent_dim: int, dropout: float = 0.15):
        super().__init__()
        self.xp_branch = ConvXPEncoder(out_dim=128)
        self.xs_branch = XSBranch(int(x_s_dim), dropout=float(dropout))
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                try:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                except Exception:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_p: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        z_p = self.xp_branch(x_p)
        z_s = self.xs_branch(x_s)
        return self.fusion(torch.cat([z_p, z_s], dim=1))
