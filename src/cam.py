"""
Class Activation Map (CAM) implementations for AIDE adversarial attack experiments.

All CAM classes share a consistent interface:
    - ``__init__(model, target_layer)``
    - ``compute(x, class_idx=None)`` -> ``(B, 1, H, W)`` tensor normalised to [0, 1]
    - ``remove_hooks()`` to clean up registered hooks

Supported methods: GradCAM, GradCAM++, LayerCAM, ScoreCAM.

IMPORTANT: ``compute()`` works on *detached copies* of the input so that the
caller's autograd graph is never affected.  All returned tensors are detached.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _BaseCAM:
    """Abstract base providing hook registration and common normalisation.

    Gradient-based CAM methods use a forward hook to capture activations and
    then ``torch.autograd.grad`` to compute gradients with respect to the
    captured activation tensor.  This avoids ``register_full_backward_hook``,
    which conflicts with downstream in-place operations (e.g. VGG's inplace
    ReLU) in recent PyTorch versions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

    # -- hook management ----------------------------------------------------

    def _register_hooks(self) -> None:
        self._hooks.append(
            self.target_layer.register_forward_hook(self._forward_hook)
        )

    def _forward_hook(self, module, input, output):
        # Handle tuple outputs (e.g. from transformer blocks).
        act = output[0] if isinstance(output, tuple) else output
        # retain_grad() so we can compute gradients via autograd.grad later.
        # We store the *live* tensor (not a clone) so it remains part of
        # the computation graph.  Skip retain_grad when running under
        # torch.no_grad() (e.g. ScoreCAM's activation capture pass).
        if act.requires_grad:
            act.retain_grad()
        self._activations = act

    def remove_hooks(self) -> None:
        """Remove all registered hooks from the model."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._activations = None

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _normalise(cam: torch.Tensor) -> torch.Tensor:
        """Per-image min-max normalisation to [0, 1].

        Parameters
        ----------
        cam : (B, 1, H, W)

        Returns
        -------
        (B, 1, H, W) in [0, 1]
        """
        b = cam.shape[0]
        flat = cam.view(b, -1)
        min_vals = flat.min(dim=1, keepdim=True).values
        max_vals = flat.max(dim=1, keepdim=True).values
        denom = (max_vals - min_vals).clamp(min=1e-8)
        flat = (flat - min_vals) / denom
        return flat.view_as(cam)

    @staticmethod
    def _to_spatial(act: torch.Tensor) -> torch.Tensor:
        """Reshape activations to (B, C, H, W) if they arrive as (B, L, C)
        from a vision-transformer style layer."""
        if act.dim() == 3:
            B, L, C = act.shape
            H = W = int(L ** 0.5)
            if H * W != L:
                raise ValueError(
                    f"Cannot reshape token sequence of length {L} to a square "
                    f"spatial grid ({H}x{W})."
                )
            act = act.permute(0, 2, 1).reshape(B, C, H, W)
        return act

    def _forward_and_backward(
        self, x: torch.Tensor, class_idx: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward + backward and return (activations, gradients) both in
        (B, C, H, W) layout.

        Uses ``torch.autograd.grad`` instead of backward hooks to avoid
        conflicts with in-place operations in the model.
        """
        self.model.zero_grad()

        output = self.model(x)  # (B, num_classes)
        B = output.shape[0]

        if class_idx is None:
            class_idx = output.argmax(dim=1)  # (B,)
        elif isinstance(class_idx, int):
            class_idx = torch.full((B,), class_idx, dtype=torch.long, device=x.device)

        # One-hot encode and compute the scalar target
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, class_idx.unsqueeze(1), 1.0)
        target = (output * one_hot).sum()

        # Compute gradient w.r.t. the captured activation tensor
        act_tensor = self._activations
        grads = torch.autograd.grad(
            target, act_tensor, retain_graph=False, create_graph=False
        )[0]

        act = self._to_spatial(act_tensor.detach())
        grad = self._to_spatial(grads.detach())
        return act, grad

    # -- abstract compute ---------------------------------------------------

    def compute(
        self,
        x: torch.Tensor,
        class_idx: Optional[int | torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the CAM heatmap.

        Parameters
        ----------
        x : (B, C, H, W) input tensor (should be on the same device as model).
        class_idx : target class index (int, (B,) tensor, or None for predicted).

        Returns
        -------
        (B, 1, H, W) normalised CAM in [0, 1], detached.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------


class GradCAM(_BaseCAM):
    """Standard Grad-CAM (Selvaraju et al., 2017).

    Weights feature maps by globally-average-pooled gradients, applies ReLU,
    upsamples to input spatial size, and normalises per image to [0, 1].
    """

    def compute(self, x, class_idx=None):
        # Work on a detached clone so caller's graph is untouched
        inp = x.detach().clone().requires_grad_(True)

        act, grad = self._forward_and_backward(inp, class_idx)

        # Global average pooling of gradients -> (B, C, 1, 1)
        weights = grad.mean(dim=(2, 3), keepdim=True)

        # Weighted combination -> (B, 1, H_feat, W_feat)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input spatial size
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = self._normalise(cam)
        return cam.detach()


# ---------------------------------------------------------------------------
# GradCAM++
# ---------------------------------------------------------------------------


class GradCAMPP(_BaseCAM):
    """Grad-CAM++ (Chattopadhyay et al., 2018).

    Uses second- and third-order gradient information to compute better
    per-pixel weights (alpha) for the feature-map channels.
    """

    def compute(self, x, class_idx=None):
        inp = x.detach().clone().requires_grad_(True)

        act, grad = self._forward_and_backward(inp, class_idx)

        # Second- and third-order gradient terms
        grad2 = grad ** 2
        grad3 = grad ** 3

        # alpha_{k}^{c} = grad^2 / (2 * grad^2 + sum_ij(A_k * grad^3) + eps)
        sum_act_grad3 = (act * grad3).sum(dim=(2, 3), keepdim=True)
        alpha = grad2 / (2.0 * grad2 + sum_act_grad3 + 1e-8)

        # weights = sum_{ij} alpha * relu(grad)
        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = self._normalise(cam)
        return cam.detach()


# ---------------------------------------------------------------------------
# LayerCAM
# ---------------------------------------------------------------------------


class LayerCAM(_BaseCAM):
    """LayerCAM (Jiang et al., 2021).

    Element-wise positive-gradient weighting: ``relu(grad) * activation``,
    then channel-wise summation.  Works better for shallow layers than
    standard Grad-CAM.
    """

    def compute(self, x, class_idx=None):
        inp = x.detach().clone().requires_grad_(True)

        act, grad = self._forward_and_backward(inp, class_idx)

        # Positive-gradient gated activations
        cam = (F.relu(grad) * act).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = self._normalise(cam)
        return cam.detach()


# ---------------------------------------------------------------------------
# ScoreCAM
# ---------------------------------------------------------------------------


class ScoreCAM(_BaseCAM):
    """Score-CAM (Wang et al., 2020).

    Gradient-free method that uses each activation channel as a mask,
    measures the resulting confidence increase, and uses that as the
    channel weight.

    WARNING: This method is *much* slower than gradient-based alternatives
    because it requires N forward passes (one per activation channel).
    The ``max_channels`` parameter caps N to keep computation tractable.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        max_channels: int = 64,
    ) -> None:
        super().__init__(model, target_layer)
        self.max_channels = max_channels

    def compute(self, x, class_idx=None):
        inp = x.detach().clone()
        B, C_in, H_in, W_in = inp.shape

        # --- Step 1: Get activation maps (no grad needed) -----------------
        self.model.eval()
        with torch.no_grad():
            output = self.model(inp)
            act = self._to_spatial(self._activations.detach())  # (B, C_act, H_f, W_f)

        C_act = act.shape[1]

        if class_idx is None:
            target_cls = output.argmax(dim=1)  # (B,)
        elif isinstance(class_idx, int):
            target_cls = torch.full((B,), class_idx, dtype=torch.long, device=inp.device)
        else:
            target_cls = class_idx

        # Upsample activations to input size -> (B, C_act, H_in, W_in)
        act_upsampled = F.interpolate(
            act, size=(H_in, W_in), mode="bilinear", align_corners=False
        )

        # Per-image, per-channel min-max normalise
        act_flat = act_upsampled.view(B, C_act, -1)
        a_min = act_flat.min(dim=2, keepdim=True).values.unsqueeze(-1)  # (B, C_act, 1, 1)
        a_max = act_flat.max(dim=2, keepdim=True).values.unsqueeze(-1)
        act_norm = (act_upsampled - a_min) / (a_max - a_min + 1e-8)  # (B, C_act, H, W)

        # --- Step 2: Limit channels if needed -----------------------------
        num_channels = min(C_act, self.max_channels)
        if num_channels < C_act:
            # Keep channels with highest mean activation (most informative)
            mean_act = act_flat.mean(dim=2)  # (B, C_act)
            # Average importance across the batch, pick top-k indices
            importance = mean_act.mean(dim=0)  # (C_act,)
            _, top_idx = importance.topk(num_channels)
            top_idx = top_idx.sort().values
            act_norm = act_norm[:, top_idx]  # (B, num_channels, H, W)
        else:
            top_idx = None

        # --- Step 3: Score each channel -----------------------------------
        # For each channel k, mask the input and get the target-class score
        scores = torch.zeros(B, num_channels, device=inp.device)

        with torch.no_grad():
            for k in range(num_channels):
                mask = act_norm[:, k : k + 1, :, :]  # (B, 1, H, W)
                masked_input = inp * mask  # (B, C_in, H, W)
                out_k = self.model(masked_input)  # (B, num_classes)
                # Softmax score for the target class
                probs = F.softmax(out_k, dim=1)
                scores[:, k] = probs.gather(1, target_cls.unsqueeze(1)).squeeze(1)

        # --- Step 4: Weighted combination ---------------------------------
        # scores -> (B, num_channels, 1, 1) as weights
        weights = F.relu(scores).unsqueeze(-1).unsqueeze(-1)

        # Use original (un-normalised) upsampled activations for the combination
        if top_idx is not None:
            act_selected = F.interpolate(
                act[:, top_idx], size=(H_in, W_in), mode="bilinear", align_corners=False
            )
        else:
            act_selected = F.interpolate(
                act, size=(H_in, W_in), mode="bilinear", align_corners=False
            )

        cam = (weights * act_selected).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = self._normalise(cam)
        return cam.detach()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CAM_REGISTRY: dict[str, type] = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPP,
    "gradcampp": GradCAMPP,
    "layercam": LayerCAM,
    "scorecam": ScoreCAM,
}


def get_cam(
    method_name: str,
    model: nn.Module,
    target_layer: nn.Module,
    **kwargs,
) -> _BaseCAM:
    """Factory function to instantiate a CAM method by name.

    Parameters
    ----------
    method_name : str
        One of ``'gradcam'``, ``'gradcam++'`` / ``'gradcampp'``,
        ``'layercam'``, ``'scorecam'``.
    model : nn.Module
        The classification model.
    target_layer : nn.Module
        The layer to compute CAMs from (see ``models.get_target_layers``).
    **kwargs
        Extra keyword arguments forwarded to the CAM constructor
        (e.g. ``max_channels`` for ScoreCAM).

    Returns
    -------
    _BaseCAM instance
    """
    key = method_name.lower().strip()
    if key not in _CAM_REGISTRY:
        raise ValueError(
            f"Unknown CAM method '{method_name}'. "
            f"Choose from: {list(_CAM_REGISTRY.keys())}"
        )
    cls = _CAM_REGISTRY[key]
    return cls(model=model, target_layer=target_layer, **kwargs)
