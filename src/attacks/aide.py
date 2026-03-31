"""
AIDE: Attention-Informed Distributional Evasion attack variants.

Core idea: at each PGD step, compute a GradCAM saliency map on the current
adversarial image and perturb *away from* salient regions.  This pushes
perturbations into regions the classifier does not attend to, improving
imperceptibility while maintaining attack success.

All variants share the same base pattern:
    1. Compute GradCAM on current adversarial image (separate forward/backward)
    2. Compute inverse saliency mask: W = 1 - S
    3. Compute attack gradient (separate forward/backward)
    4. Apply mask AFTER sign():  adv = adv + alpha * sign(grad) * W
    5. Project back into epsilon-ball and [0, 1]

CRITICAL: the mask is applied *after* sign(), not before.  This preserves the
Linf geometry -- sign(grad * W) != sign(grad) * W in general.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cam import get_cam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_adv(images: torch.Tensor, labels: torch.Tensor,
              epsilon: float, device: torch.device):
    """Clone inputs to device and create random-start adversarial images."""
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    adv = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, 0.0, 1.0).detach()
    return images, labels, adv


def _project(adv: torch.Tensor, images: torch.Tensor,
             epsilon: float) -> torch.Tensor:
    """Project back into Linf epsilon-ball around *images* and clip to [0,1]."""
    perturbation = torch.clamp(adv - images, -epsilon, epsilon)
    return torch.clamp(images + perturbation, 0.0, 1.0).detach()


def _compute_cam_mask(model: nn.Module, target_layer: nn.Module,
                      adv: torch.Tensor, cam_method: str) -> torch.Tensor:
    """Create a fresh CAM, compute the inverse-saliency mask, clean up hooks.

    Returns
    -------
    W : (B, 1, H, W) inverse-saliency mask in [0, 1], detached, same device.
    """
    cam_obj = get_cam(cam_method, model, target_layer)
    saliency = cam_obj.compute(adv)          # (B, 1, H, W) in [0, 1]
    cam_obj.remove_hooks()
    W = 1.0 - saliency                       # inverse saliency
    return W.detach()


def _attack_grad(model: nn.Module, adv: torch.Tensor,
                 labels: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Forward + backward for the CE attack loss.

    Returns
    -------
    grad : (B, C, H, W) raw gradient, detached
    loss_val : scalar loss value (float)
    """
    adv_input = adv.clone().detach().requires_grad_(True)
    outputs = model(adv_input)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    return adv_input.grad.detach(), loss.item()


def _pack_result(adv: torch.Tensor,
                 drift_maps: list | None,
                 losses: list | None,
                 record_drift: bool,
                 record_loss: bool):
    """Return adv_images alone or as a tuple with requested extras."""
    extras = []
    if record_drift:
        extras.append(drift_maps)
    if record_loss:
        extras.append(losses)
    if extras:
        return (adv,) + tuple(extras)
    return adv


# ---------------------------------------------------------------------------
# AIDE-Base
# ---------------------------------------------------------------------------

def aide_base_attack(model, images, labels, epsilon, alpha, num_steps, device,
                     target_layer, cam_method='gradcam',
                     record_drift=False, record_loss=False):
    """AIDE-Base: PGD with per-step inverse-saliency masking.

    At each iteration the saliency map is freshly computed on the current
    adversarial image.  Perturbations are steered away from attended regions
    by element-wise multiplication with the inverse saliency mask *after*
    taking the sign of the gradient.

    Args:
        model: classifier in eval mode
        images: (B, C, H, W) in [0, 1]
        labels: (B,) true labels
        epsilon: Linf perturbation budget
        alpha: step size per iteration
        num_steps: number of attack iterations
        device: torch device
        target_layer: nn.Module for CAM computation
        cam_method: CAM variant name (default ``'gradcam'``)
        record_drift: if True, collect per-step CAM maps as numpy arrays
        record_loss: if True, collect per-step CE loss values

    Returns:
        adv_images: (B, C, H, W)  -- or tuple with extras if recording
    """
    images, labels, adv = _init_adv(images, labels, epsilon, device)

    drift_maps = [] if record_drift else None
    losses = [] if record_loss else None

    for _ in range(num_steps):
        # 1. Compute inverse-saliency mask from current adversarial image
        W = _compute_cam_mask(model, target_layer, adv, cam_method)

        if record_drift:
            drift_maps.append(W.cpu().numpy())

        # 2. Compute attack gradient (separate forward/backward pass)
        grad, loss_val = _attack_grad(model, adv, labels)

        if record_loss:
            losses.append(loss_val)

        # 3. Masked sign-gradient step  (sign FIRST, then mask)
        adv = adv + alpha * grad.sign() * W

        # 4. Project back into epsilon-ball and [0, 1]
        adv = _project(adv, images, epsilon)

    return _pack_result(adv, drift_maps, losses, record_drift, record_loss)


# ---------------------------------------------------------------------------
# AIDE-Momentum
# ---------------------------------------------------------------------------

def aide_momentum_attack(model, images, labels, epsilon, alpha, num_steps, device,
                         target_layer, cam_method='gradcam', decay_factor=1.0,
                         record_drift=False, record_loss=False):
    """AIDE-Momentum: AIDE with momentum accumulation on masked gradients.

    Momentum is accumulated on the L1-normalised gradient *before* masking:
        g_t = decay * g_{t-1} + grad / ||grad||_1
    Then the update is:
        adv = adv + alpha * sign(g_t) * W_t

    This stabilises the gradient direction across steps while still steering
    perturbations away from salient regions.

    Args:
        model: classifier in eval mode
        images: (B, C, H, W) in [0, 1]
        labels: (B,) true labels
        epsilon, alpha, num_steps, device: standard PGD params
        target_layer: nn.Module for CAM computation
        cam_method: CAM variant name
        decay_factor: momentum decay (default 1.0)
        record_drift: collect per-step CAM maps
        record_loss: collect per-step CE loss values

    Returns:
        adv_images or tuple with extras
    """
    images, labels, adv = _init_adv(images, labels, epsilon, device)

    momentum = torch.zeros_like(images)
    drift_maps = [] if record_drift else None
    losses = [] if record_loss else None

    for _ in range(num_steps):
        # 1. Inverse-saliency mask
        W = _compute_cam_mask(model, target_layer, adv, cam_method)

        if record_drift:
            drift_maps.append(W.cpu().numpy())

        # 2. Attack gradient
        grad, loss_val = _attack_grad(model, adv, labels)

        if record_loss:
            losses.append(loss_val)

        # 3. Momentum accumulation on L1-normalised gradient
        grad_norm = grad / (grad.abs().sum(dim=(1, 2, 3), keepdim=True) + 1e-12)
        momentum = decay_factor * momentum + grad_norm

        # 4. Masked sign-momentum step
        adv = adv + alpha * momentum.sign() * W

        # 5. Project
        adv = _project(adv, images, epsilon)

    return _pack_result(adv, drift_maps, losses, record_drift, record_loss)


# ---------------------------------------------------------------------------
# AIDE-MultiScale
# ---------------------------------------------------------------------------

def aide_multiscale_attack(model, images, labels, epsilon, alpha, num_steps, device,
                           target_layers, cam_method='gradcam',
                           record_drift=False, record_loss=False):
    """AIDE-MultiScale: Fuse CAMs from multiple layers.

    Computes CAM at each layer in *target_layers*, averages them, and then
    inverts to form the mask.  This captures both fine-grained (early layer)
    and semantic (late layer) saliency, producing a more robust avoidance
    mask.

    Args:
        model: classifier in eval mode
        images: (B, C, H, W) in [0, 1]
        labels: (B,) true labels
        epsilon, alpha, num_steps, device: standard PGD params
        target_layers: list of nn.Module layers for CAM computation
        cam_method: CAM variant name
        record_drift: collect per-step (averaged, inverted) CAM maps
        record_loss: collect per-step CE loss values

    Returns:
        adv_images or tuple with extras
    """
    images, labels, adv = _init_adv(images, labels, epsilon, device)

    drift_maps = [] if record_drift else None
    losses = [] if record_loss else None

    for _ in range(num_steps):
        # 1. Compute CAM at each layer and average
        saliency_sum = torch.zeros(images.shape[0], 1,
                                   images.shape[2], images.shape[3],
                                   device=device)
        for layer in target_layers:
            cam_obj = get_cam(cam_method, model, layer)
            saliency = cam_obj.compute(adv)  # (B, 1, H, W)
            cam_obj.remove_hooks()
            saliency_sum = saliency_sum + saliency

        avg_saliency = saliency_sum / len(target_layers)
        W = (1.0 - avg_saliency).detach()

        if record_drift:
            drift_maps.append(W.cpu().numpy())

        # 2. Attack gradient
        grad, loss_val = _attack_grad(model, adv, labels)

        if record_loss:
            losses.append(loss_val)

        # 3. Masked sign-gradient step
        adv = adv + alpha * grad.sign() * W

        # 4. Project
        adv = _project(adv, images, epsilon)

    return _pack_result(adv, drift_maps, losses, record_drift, record_loss)


# ---------------------------------------------------------------------------
# AIDE-Adaptive
# ---------------------------------------------------------------------------

def aide_adaptive_attack(model, images, labels, epsilon, alpha, num_steps, device,
                         target_layer, cam_method='gradcam', stall_patience=3,
                         record_drift=False, record_loss=False):
    """AIDE-Adaptive: Switch to uniform perturbation when attack stalls.

    Tracks the CE loss.  If the loss does not improve (increase, since we
    are attacking) for *stall_patience* consecutive steps, one step of
    uniform PGD (no masking) is applied before resuming AIDE.

    Args:
        model: classifier in eval mode
        images: (B, C, H, W) in [0, 1]
        labels: (B,) true labels
        epsilon, alpha, num_steps, device: standard PGD params
        target_layer: nn.Module for CAM computation
        cam_method: CAM variant name
        stall_patience: consecutive non-improving steps before fallback
        record_drift: collect per-step CAM maps (None for uniform steps)
        record_loss: collect per-step CE loss values

    Returns:
        adv_images or tuple with extras
    """
    images, labels, adv = _init_adv(images, labels, epsilon, device)

    drift_maps = [] if record_drift else None
    losses = [] if record_loss else None

    best_loss = -float('inf')
    stall_count = 0

    for _ in range(num_steps):
        use_mask = (stall_count < stall_patience)

        if use_mask:
            # AIDE step: compute inverse-saliency mask
            W = _compute_cam_mask(model, target_layer, adv, cam_method)

            if record_drift:
                drift_maps.append(W.cpu().numpy())
        else:
            # Uniform PGD step: no masking
            W = None
            stall_count = 0  # reset after one uniform step

            if record_drift:
                drift_maps.append(None)

        # Attack gradient
        grad, loss_val = _attack_grad(model, adv, labels)

        if record_loss:
            losses.append(loss_val)

        # Track stalling (loss should increase for a successful attack)
        if loss_val > best_loss:
            best_loss = loss_val
            stall_count = 0
        else:
            stall_count += 1

        # Gradient step
        if W is not None:
            adv = adv + alpha * grad.sign() * W
        else:
            adv = adv + alpha * grad.sign()

        # Project
        adv = _project(adv, images, epsilon)

    return _pack_result(adv, drift_maps, losses, record_drift, record_loss)


# ---------------------------------------------------------------------------
# AIDE-Soft
# ---------------------------------------------------------------------------

def aide_soft_attack(model, images, labels, epsilon, alpha, num_steps, device,
                     target_layer, cam_method='gradcam', temperature=0.1,
                     record_drift=False, record_loss=False):
    """AIDE-Soft: Temperature-controlled softmax mask instead of hard inverse.

    Instead of W = 1 - S, the inverse saliency is passed through a spatial
    softmax with a temperature parameter:
        W_t = softmax((1 - S_t) / temperature)
    The result is normalised so its spatial mean equals 1.0, preserving the
    overall perturbation magnitude.

    Low temperature  -> strict avoidance of salient regions (peaky mask)
    High temperature -> more uniform perturbation (flat mask)

    Args:
        model: classifier in eval mode
        images: (B, C, H, W) in [0, 1]
        labels: (B,) true labels
        epsilon, alpha, num_steps, device: standard PGD params
        target_layer: nn.Module for CAM computation
        cam_method: CAM variant name
        temperature: softmax temperature (default 0.1)
        record_drift: collect per-step CAM maps
        record_loss: collect per-step CE loss values

    Returns:
        adv_images or tuple with extras
    """
    images, labels, adv = _init_adv(images, labels, epsilon, device)

    drift_maps = [] if record_drift else None
    losses = [] if record_loss else None

    for _ in range(num_steps):
        # 1. Compute saliency and inverse
        cam_obj = get_cam(cam_method, model, target_layer)
        saliency = cam_obj.compute(adv)  # (B, 1, H, W)
        cam_obj.remove_hooks()

        inv_saliency = 1.0 - saliency  # (B, 1, H, W)

        # 2. Spatial softmax with temperature
        B, _, H, W_dim = inv_saliency.shape
        flat = inv_saliency.view(B, -1) / temperature      # (B, H*W)
        W = F.softmax(flat, dim=1).view(B, 1, H, W_dim)    # (B, 1, H, W)

        # 3. Normalise so spatial mean = 1.0 (preserves perturbation magnitude)
        spatial_mean = W.mean(dim=(2, 3), keepdim=True).clamp(min=1e-12)
        W = (W / spatial_mean).detach()

        if record_drift:
            drift_maps.append(W.cpu().numpy())

        # 4. Attack gradient
        grad, loss_val = _attack_grad(model, adv, labels)

        if record_loss:
            losses.append(loss_val)

        # 5. Masked sign-gradient step
        adv = adv + alpha * grad.sign() * W

        # 6. Project
        adv = _project(adv, images, epsilon)

    return _pack_result(adv, drift_maps, losses, record_drift, record_loss)
