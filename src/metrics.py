"""
Evaluation metrics for adversarial attacks.

All image-based functions expect batched tensors in [0, 1] range with shape
(B, C, H, W) unless otherwise noted.

Metric groups:
    - Attack effectiveness (ASR, confidence drop)
    - Perceptual quality   (LPIPS, SSIM, PSNR, Lp norms)
    - Attention drift       (observed dissimilarity, spatial entropy, centroid displacement)
    - MetricsAccumulator   (batch-level aggregation wrapper)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import lpips
from pytorch_msssim import ssim as _ssim_fn

# ---------------------------------------------------------------------------
# Module-level LPIPS singleton
# ---------------------------------------------------------------------------
_LPIPS_MODELS: dict[str, lpips.LPIPS] = {}


def _get_lpips_model(net: str = "alex", device: torch.device | str = "cpu") -> lpips.LPIPS:
    """Return a cached, eval-mode LPIPS network on *device*."""
    key = f"{net}_{device}"
    if key not in _LPIPS_MODELS:
        model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        # Freeze all parameters so no accidental gradient tracking occurs.
        for p in model.parameters():
            p.requires_grad_(False)
        _LPIPS_MODELS[key] = model
    return _LPIPS_MODELS[key]


# ===================================================================
# Attack Effectiveness
# ===================================================================

@torch.no_grad()
def compute_asr(
    model: torch.nn.Module,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device | str = "cpu",
) -> float:
    """Attack Success Rate -- percentage of adversarial images misclassified.

    Parameters
    ----------
    model : torch.nn.Module
        Classifier that returns logits of shape (B, num_classes).
    adv_images : Tensor (B, C, H, W) in [0, 1].
    labels : Tensor (B,) with ground-truth class indices.
    device : device to run inference on.

    Returns
    -------
    float  Percentage in [0, 100].
    """
    model.eval()
    adv_images = adv_images.to(device)
    labels = labels.to(device)

    logits = model(adv_images)
    preds = logits.argmax(dim=1)
    misclassified = (preds != labels).float().mean().item()
    return misclassified * 100.0


@torch.no_grad()
def compute_confidence_drop(
    model: torch.nn.Module,
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device | str = "cpu",
) -> float:
    """Mean decrease in the true-class probability after perturbation.

    Parameters
    ----------
    model : torch.nn.Module
        Classifier returning logits (B, num_classes).
    clean_images, adv_images : Tensor (B, C, H, W) in [0, 1].
    labels : Tensor (B,) ground-truth indices.
    device : device for inference.

    Returns
    -------
    float  Mean probability drop (positive means the attack reduced confidence).
    """
    model.eval()
    clean_images = clean_images.to(device)
    adv_images = adv_images.to(device)
    labels = labels.to(device)

    clean_probs = F.softmax(model(clean_images), dim=1)
    adv_probs = F.softmax(model(adv_images), dim=1)

    # Gather the true-class probability for each sample.
    idx = labels.unsqueeze(1)  # (B, 1)
    clean_conf = clean_probs.gather(1, idx).squeeze(1)  # (B,)
    adv_conf = adv_probs.gather(1, idx).squeeze(1)  # (B,)

    drop = (clean_conf - adv_conf).mean().item()
    return drop


# ===================================================================
# Perceptual Quality
# ===================================================================

@torch.no_grad()
def compute_lpips(
    clean: torch.Tensor,
    adv: torch.Tensor,
    device: torch.device | str = "cpu",
    net: str = "alex",
) -> np.ndarray:
    """Per-image LPIPS distance.

    Parameters
    ----------
    clean, adv : Tensor (B, C, H, W) in [0, 1].
    device : device to run the LPIPS network on.
    net : backbone -- ``'alex'`` (default) or ``'vgg'``.

    Returns
    -------
    numpy.ndarray  Shape (B,) with per-image LPIPS values.
    """
    lpips_model = _get_lpips_model(net=net, device=device)

    # LPIPS expects images in [-1, 1].
    clean_scaled = (clean.to(device) * 2.0 - 1.0)
    adv_scaled = (adv.to(device) * 2.0 - 1.0)

    # lpips_model returns (B, 1, 1, 1)
    dist = lpips_model(clean_scaled, adv_scaled)
    return dist.squeeze().cpu().numpy()


@torch.no_grad()
def compute_ssim(
    clean: torch.Tensor,
    adv: torch.Tensor,
) -> np.ndarray:
    """Per-image SSIM (Structural Similarity Index).

    Parameters
    ----------
    clean, adv : Tensor (B, C, H, W) in [0, 1].

    Returns
    -------
    numpy.ndarray  Shape (B,) with per-image SSIM values.
    """
    B = clean.shape[0]
    ssim_vals = torch.empty(B, device=clean.device)
    # Compute per-image to get individual values.
    for i in range(B):
        c_i = clean[i : i + 1]
        a_i = adv[i : i + 1]
        ssim_vals[i] = _ssim_fn(c_i, a_i, data_range=1.0, size_average=True)
    return ssim_vals.cpu().numpy()


@torch.no_grad()
def compute_psnr(
    clean: torch.Tensor,
    adv: torch.Tensor,
) -> np.ndarray:
    """Per-image PSNR (Peak Signal-to-Noise Ratio).

    PSNR = 10 * log10(1.0 / MSE) for each image.

    Parameters
    ----------
    clean, adv : Tensor (B, C, H, W) in [0, 1].

    Returns
    -------
    numpy.ndarray  Shape (B,).
    """
    # MSE per image: mean over (C, H, W).
    mse = (clean - adv).pow(2).flatten(1).mean(dim=1)  # (B,)
    # Clamp to avoid log(0).
    mse = mse.clamp(min=1e-10)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr.cpu().numpy()


@torch.no_grad()
def compute_l2(
    clean: torch.Tensor,
    adv: torch.Tensor,
) -> np.ndarray:
    """Per-image L2 distance.

    Parameters
    ----------
    clean, adv : Tensor (B, C, H, W) in [0, 1].

    Returns
    -------
    numpy.ndarray  Shape (B,).
    """
    diff = (clean - adv).flatten(1)  # (B, C*H*W)
    l2 = diff.norm(p=2, dim=1)  # (B,)
    return l2.cpu().numpy()


@torch.no_grad()
def compute_l0(
    clean: torch.Tensor,
    adv: torch.Tensor,
    threshold: float = 1e-6,
) -> np.ndarray:
    """Per-image L0 -- fraction of pixels changed above *threshold*.

    A "pixel" is a spatial location across all channels: a pixel counts as
    changed if *any* channel differs by more than *threshold*.

    Parameters
    ----------
    clean, adv : Tensor (B, C, H, W) in [0, 1].
    threshold : float

    Returns
    -------
    numpy.ndarray  Shape (B,) with fractions in [0, 1].
    """
    diff_abs = (clean - adv).abs()  # (B, C, H, W)
    # A pixel is "changed" if the max channel-wise difference exceeds threshold.
    changed = diff_abs.amax(dim=1) > threshold  # (B, H, W)
    num_pixels = changed.shape[1] * changed.shape[2]
    fraction = changed.float().flatten(1).sum(dim=1) / num_pixels  # (B,)
    return fraction.cpu().numpy()


@torch.no_grad()
def compute_linf(
    clean: torch.Tensor,
    adv: torch.Tensor,
) -> np.ndarray:
    """Per-image L-infinity distance.

    Parameters
    ----------
    clean, adv : Tensor (B, C, H, W) in [0, 1].

    Returns
    -------
    numpy.ndarray  Shape (B,).
    """
    linf = (clean - adv).abs().flatten(1).amax(dim=1)  # (B,)
    return linf.cpu().numpy()


# ===================================================================
# Attention Drift Metrics
# ===================================================================

@torch.no_grad()
def compute_mean_observed_dissimilarity(
    cam_sequence: List[torch.Tensor],
) -> float:
    """Mean Observed Dissimilarity (MOD) across consecutive CAM steps.

    Measures how much the model's attention drifts between successive attack
    iterations.

    Parameters
    ----------
    cam_sequence : list of Tensors, each (B, 1, H, W).
        One CAM map per optimisation step.

    Returns
    -------
    float  1 - mean(cosine_similarity) averaged over all consecutive pairs
           and all samples in the batch.
    """
    if len(cam_sequence) < 2:
        return 0.0

    similarities: list[torch.Tensor] = []
    for prev, curr in zip(cam_sequence[:-1], cam_sequence[1:]):
        # Flatten spatial dims to vectors for cosine similarity.
        p_flat = prev.flatten(1)  # (B, H*W)
        c_flat = curr.flatten(1)  # (B, H*W)
        cos_sim = F.cosine_similarity(p_flat, c_flat, dim=1)  # (B,)
        similarities.append(cos_sim)

    # Stack: (num_steps-1, B), mean over everything.
    mean_sim = torch.stack(similarities).mean().item()
    return 1.0 - mean_sim


@torch.no_grad()
def compute_spatial_entropy(
    perturbation: torch.Tensor,
) -> float:
    """Shannon entropy of the spatial energy distribution of a perturbation.

    High entropy means the perturbation energy is spread uniformly; low
    entropy means it is concentrated in a few regions.

    Parameters
    ----------
    perturbation : Tensor (B, C, H, W)
        Typically ``adv_images - clean_images``.

    Returns
    -------
    float  Mean spatial entropy across the batch (in nats).
    """
    # Per-pixel L2 norm across channels -> (B, H, W).
    energy = perturbation.pow(2).sum(dim=1).sqrt()  # (B, H, W)

    # Normalise each image's energy map to a probability distribution.
    energy_flat = energy.flatten(1)  # (B, N)  where N = H*W
    total = energy_flat.sum(dim=1, keepdim=True).clamp(min=1e-12)
    prob = energy_flat / total  # (B, N)

    # Shannon entropy: -sum(p * log(p)), ignoring zero entries.
    log_prob = torch.log(prob.clamp(min=1e-12))
    entropy = -(prob * log_prob).sum(dim=1)  # (B,)

    return entropy.mean().item()


@torch.no_grad()
def compute_attention_centroid_displacement(
    cam_sequence: List[torch.Tensor],
) -> float:
    """Mean per-step centroid displacement of the CAM across attack iterations.

    The centroid at each step is the weighted spatial mean (x, y) using the
    CAM intensities as weights.

    Parameters
    ----------
    cam_sequence : list of Tensors, each (B, 1, H, W).

    Returns
    -------
    float  Mean Euclidean displacement between consecutive centroids,
           averaged over steps and batch.  Units are in pixel coordinates.
    """
    if len(cam_sequence) < 2:
        return 0.0

    def _centroid(cam: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cx, cy) each of shape (B,)."""
        B, _, H, W = cam.shape
        cam_2d = cam.squeeze(1)  # (B, H, W)
        total = cam_2d.sum(dim=(1, 2)).clamp(min=1e-12)  # (B,)

        # Coordinate grids on the same device.
        gy = torch.arange(H, dtype=cam.dtype, device=cam.device).view(1, H, 1)
        gx = torch.arange(W, dtype=cam.dtype, device=cam.device).view(1, 1, W)

        cy = (cam_2d * gy).sum(dim=(1, 2)) / total  # (B,)
        cx = (cam_2d * gx).sum(dim=(1, 2)) / total  # (B,)
        return cx, cy

    displacements: list[torch.Tensor] = []
    prev_cx, prev_cy = _centroid(cam_sequence[0])

    for cam in cam_sequence[1:]:
        cx, cy = _centroid(cam)
        dist = ((cx - prev_cx).pow(2) + (cy - prev_cy).pow(2)).sqrt()  # (B,)
        displacements.append(dist)
        prev_cx, prev_cy = cx, cy

    return torch.stack(displacements).mean().item()


# ===================================================================
# MetricsAccumulator
# ===================================================================

class MetricsAccumulator:
    """Accumulate per-batch metrics and produce aggregated statistics.

    Usage
    -----
    >>> acc = MetricsAccumulator(device="cuda:0")
    >>> for clean, adv, labels in dataloader:
    ...     acc.update(clean, adv, labels, model)
    >>> results = acc.compute()  # dict with means and stds
    >>> acc.reset()
    """

    _METRIC_KEYS = (
        "asr",
        "confidence_drop",
        "lpips",
        "ssim",
        "psnr",
        "l2",
        "l0",
        "linf",
    )

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = device
        # Pre-warm the LPIPS model so the first batch isn't slow.
        _get_lpips_model(net="alex", device=device)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all accumulated values."""
        self._store: dict[str, list[np.ndarray | float]] = {
            k: [] for k in self._METRIC_KEYS
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        clean_batch: torch.Tensor,
        adv_batch: torch.Tensor,
        labels: torch.Tensor,
        model: torch.nn.Module,
    ) -> None:
        """Compute all metrics for one batch and accumulate results.

        Parameters
        ----------
        clean_batch, adv_batch : Tensor (B, C, H, W) in [0, 1].
        labels : Tensor (B,).
        model : classifier returning logits.
        """
        device = self.device

        # --- Attack effectiveness ---
        asr = compute_asr(model, adv_batch, labels, device)
        conf_drop = compute_confidence_drop(model, clean_batch, adv_batch, labels, device)
        self._store["asr"].append(asr)
        self._store["confidence_drop"].append(conf_drop)

        # --- Perceptual quality (batched) ---
        # Move to device once for all the torch-only metrics.
        clean_d = clean_batch.to(device)
        adv_d = adv_batch.to(device)

        self._store["lpips"].append(compute_lpips(clean_d, adv_d, device))
        self._store["ssim"].append(compute_ssim(clean_d, adv_d))
        self._store["psnr"].append(compute_psnr(clean_d, adv_d))
        self._store["l2"].append(compute_l2(clean_d, adv_d))
        self._store["l0"].append(compute_l0(clean_d, adv_d))
        self._store["linf"].append(compute_linf(clean_d, adv_d))

    # ------------------------------------------------------------------
    def compute(self) -> dict[str, float]:
        """Aggregate accumulated metrics into means and standard deviations.

        Returns
        -------
        dict  Keys are ``"{metric}_mean"`` and ``"{metric}_std"`` for every
              metric tracked by this accumulator.
        """
        results: dict[str, float] = {}
        for key in self._METRIC_KEYS:
            values = self._store[key]
            if not values:
                results[f"{key}_mean"] = 0.0
                results[f"{key}_std"] = 0.0
                continue

            # Scalar metrics (asr, confidence_drop) are stored as floats;
            # per-image metrics are numpy arrays.  Unify into a single array.
            if isinstance(values[0], (float, int)):
                arr = np.array(values, dtype=np.float64)
            else:
                arr = np.concatenate(values).astype(np.float64)

            results[f"{key}_mean"] = float(arr.mean())
            results[f"{key}_std"] = float(arr.std())

        return results
