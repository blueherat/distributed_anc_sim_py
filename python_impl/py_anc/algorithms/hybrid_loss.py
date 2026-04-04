from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _next_pow2(n: int) -> int:
    n_i = int(max(n, 1))
    return 1 if n_i <= 1 else 1 << (n_i - 1).bit_length()


def _pad_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[-1] >= int(target_len):
        return x[..., : int(target_len)]
    return F.pad(x, (0, int(target_len) - int(x.shape[-1])))


def _full_convolution_1d(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Vectorized full convolution for batched 1D signals.

    signal and kernel can be shaped [B, L] or [B, C, L].
    """
    signal_was_2d = signal.ndim == 2
    kernel_was_2d = kernel.ndim == 2
    if signal_was_2d:
        signal = signal.unsqueeze(1)
    if kernel_was_2d:
        kernel = kernel.unsqueeze(1)
    if signal.ndim != 3 or kernel.ndim != 3:
        raise ValueError("signal/kernel must be 2D or 3D tensors.")
    if signal.shape[:2] != kernel.shape[:2]:
        raise ValueError(
            f"signal and kernel batch/channel dims must match. Got {signal.shape[:2]} vs {kernel.shape[:2]}"
        )

    bsz, channels, signal_len = signal.shape
    kernel_len = int(kernel.shape[-1])

    x = signal.reshape(1, int(bsz * channels), int(signal_len))
    # conv1d computes correlation; flip kernel to obtain mathematical convolution.
    k = torch.flip(kernel, dims=[-1]).reshape(int(bsz * channels), 1, int(kernel_len))
    y = F.conv1d(x, k, groups=int(bsz * channels), padding=int(kernel_len - 1))
    y = y.reshape(int(bsz), int(channels), int(signal_len + kernel_len - 1))

    if signal_was_2d and kernel_was_2d:
        return y[:, 0, :]
    return y


class HybridAcousticLoss(nn.Module):
    """Physics-based loss for Hybrid Deep-FxLMS.

    Computes:
      L_acoustic = (1/B) * sum || P_true + (S_true * W_pred_effective) ||_2^2
      L_reg = lambda * ||W_pred||_2^2
      L_total = L_acoustic + L_reg

    W_pred_effective is computed from reference-wise filters and primary-to-reference paths:
      W_pred_effective = sum_r (W_pred[r] * P_ref[r])
    """

    def __init__(
        self,
        lambda_reg: float = 1.0e-3,
        conv_domain: str = "freq",
        nr_margin_weight: float = 0.0,
        nr_target_ratio: float = 0.5,
        nr_margin_mode: str = "power",
        nr_margin_focus_ratio: float = 1.0,
    ):
        super().__init__()
        if str(conv_domain) not in ("freq", "time"):
            raise ValueError(f"Unsupported conv_domain: {conv_domain}")
        if float(nr_margin_weight) < 0.0:
            raise ValueError("nr_margin_weight must be non-negative.")
        if float(nr_target_ratio) <= 0.0:
            raise ValueError("nr_target_ratio must be positive.")
        if str(nr_margin_mode) not in ("power", "db"):
            raise ValueError("nr_margin_mode must be one of: power, db")
        if not (0.0 < float(nr_margin_focus_ratio) <= 1.0):
            raise ValueError("nr_margin_focus_ratio must be in (0, 1].")
        self.lambda_reg = float(lambda_reg)
        self.conv_domain = str(conv_domain)
        self.nr_margin_weight = float(nr_margin_weight)
        self.nr_target_ratio = float(nr_target_ratio)
        self.nr_margin_mode = str(nr_margin_mode)
        self.nr_margin_focus_ratio = float(nr_margin_focus_ratio)

    def _forward_time(
        self,
        w_pred: torch.Tensor,
        p_ref: torch.Tensor,
        p_true: torch.Tensor,
        s_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Effective filter at the control point: sum_r (W_r * P_ref_r)
        w_eff_each = _full_convolution_1d(w_pred, p_ref)
        w_eff = torch.sum(w_eff_each, dim=1)

        s_conv_w = _full_convolution_1d(s_true, w_eff)
        target_len = int(max(int(s_conv_w.shape[-1]), int(p_true.shape[-1])))
        p_true_pad = _pad_to_length(p_true, target_len)
        s_conv_w_pad = _pad_to_length(s_conv_w, target_len)
        residual = p_true_pad + s_conv_w_pad
        return residual, p_true_pad

    def _forward_freq(
        self,
        w_pred: torch.Tensor,
        p_ref: torch.Tensor,
        p_true: torch.Tensor,
        s_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w_len = int(w_pred.shape[-1])
        p_ref_len = int(p_ref.shape[-1])
        p_true_len = int(p_true.shape[-1])
        s_len = int(s_true.shape[-1])

        eff_len = int(w_len + p_ref_len - 1)
        residual_len = int(s_len + eff_len - 1)
        target_len = int(max(residual_len, p_true_len))
        n_fft = int(_next_pow2(target_len))

        w_f = torch.fft.rfft(w_pred, n=n_fft, dim=-1)
        p_ref_f = torch.fft.rfft(p_ref, n=n_fft, dim=-1)
        w_eff_f = torch.sum(w_f * p_ref_f, dim=1)

        s_f = torch.fft.rfft(s_true, n=n_fft, dim=-1)
        p_true_f = torch.fft.rfft(p_true, n=n_fft, dim=-1)

        residual_f = p_true_f + s_f * w_eff_f
        residual = torch.fft.irfft(residual_f, n=n_fft, dim=-1)[..., :target_len]
        p_true_pad = _pad_to_length(p_true, target_len)
        return residual, p_true_pad

    def forward(
        self,
        w_pred: torch.Tensor,
        p_ref: torch.Tensor,
        p_true: torch.Tensor,
        s_true: torch.Tensor,
        target_nr_db: torch.Tensor | None = None,
        margin_weight: float | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if w_pred.ndim != 3 or p_ref.ndim != 3:
            raise ValueError("w_pred and p_ref must be [B, R, L] tensors.")
        if p_true.ndim != 2 or s_true.ndim != 2:
            raise ValueError("p_true and s_true must be [B, L] tensors.")
        if w_pred.shape[:2] != p_ref.shape[:2]:
            raise ValueError(
                f"w_pred and p_ref batch/ref dims must match. Got {w_pred.shape[:2]} vs {p_ref.shape[:2]}"
            )
        if w_pred.shape[0] != p_true.shape[0] or w_pred.shape[0] != s_true.shape[0]:
            raise ValueError("Batch size mismatch between predicted filters and true paths.")

        if self.conv_domain == "time":
            residual, p_true_pad = self._forward_time(w_pred=w_pred, p_ref=p_ref, p_true=p_true, s_true=s_true)
        else:
            residual, p_true_pad = self._forward_freq(w_pred=w_pred, p_ref=p_ref, p_true=p_true, s_true=s_true)

        if margin_weight is None:
            margin_weight_t = torch.tensor(self.nr_margin_weight, device=w_pred.device, dtype=w_pred.dtype)
        else:
            margin_weight_t = torch.as_tensor(margin_weight, device=w_pred.device, dtype=w_pred.dtype)

        acoustic = torch.mean(residual.pow(2))
        reg = self.lambda_reg * torch.mean(w_pred.pow(2))

        margin = torch.zeros((), device=w_pred.device, dtype=w_pred.dtype)
        if target_nr_db is not None and float(margin_weight_t.detach().cpu()) > 0.0:
            target_nr = torch.as_tensor(target_nr_db, device=w_pred.device, dtype=w_pred.dtype).reshape(-1)
            if target_nr.shape[0] != residual.shape[0]:
                raise ValueError(
                    "target_nr_db batch size mismatch. "
                    f"Expected {residual.shape[0]}, got {target_nr.shape[0]}"
                )

            valid = torch.isfinite(target_nr)
            if torch.any(valid):
                if self.nr_margin_focus_ratio < 1.0:
                    focus_len = max(1, int(round(float(residual.shape[-1]) * self.nr_margin_focus_ratio)))
                    residual_margin = residual[..., :focus_len]
                    p_true_margin = p_true_pad[..., :focus_len]
                else:
                    residual_margin = residual
                    p_true_margin = p_true_pad

                residual_power = torch.mean(residual_margin.pow(2), dim=-1)
                noise_power = torch.mean(p_true_margin.pow(2), dim=-1)
                target_db = target_nr * self.nr_target_ratio

                if self.nr_margin_mode == "power":
                    target_residual_power = noise_power * torch.pow(
                        torch.tensor(10.0, device=w_pred.device, dtype=w_pred.dtype),
                        -target_db / 10.0,
                    )
                    margin_sample = F.relu(residual_power - target_residual_power)
                else:
                    achieved_db = 10.0 * torch.log10((noise_power + 1.0e-12) / (residual_power + 1.0e-12))
                    # Penalize dB shortfall directly; this is much more sensitive when far from target.
                    margin_sample = F.relu(target_db - achieved_db)

                margin = torch.mean(margin_sample[valid])

        total = acoustic + reg + margin_weight_t * margin

        with torch.no_grad():
            noise_power = torch.mean(p_true_pad.pow(2))
            nr_db = 10.0 * torch.log10((noise_power + 1.0e-12) / (acoustic + 1.0e-12))

        return {
            "total": total,
            "acoustic": acoustic,
            "reg": reg,
            "margin": margin,
            "margin_weight": margin_weight_t,
            "nr_db": nr_db,
        }
