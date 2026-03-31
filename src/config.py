"""
Configuration dataclasses for AIDE adversarial attack experiments.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AttackConfig:
    """Configuration for a single adversarial attack."""

    epsilon: float
    alpha: float
    num_steps: int
    attack_name: str

    # MI-FGSM specific
    decay_factor: Optional[float] = None

    # AIDE-specific extensions
    momentum: bool = False
    multi_scale: bool = False
    adaptive: bool = False
    temperature: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Top-level configuration for an experiment run."""

    dataset: str
    model_name: str
    attack_config: AttackConfig
    num_eval_images: int = 1000
    batch_size: int = 64
    device: str = "cuda"
    cam_method: str = "gradcam"
    cam_layer: str = "auto"


# Standard epsilon sweep values (L-inf, in [0,1] scale).
EPSILON_VALUES = [2 / 255, 4 / 255, 8 / 255, 16 / 255]


def get_default_configs() -> dict[str, AttackConfig]:
    """Return a dictionary of standard attack configurations.

    All use L-inf eps = 8/255 unless otherwise noted.
    """
    eps = 8 / 255
    alpha = 2 / 255

    return {
        # --- Baselines ---
        "PGD-20": AttackConfig(
            epsilon=eps,
            alpha=alpha,
            num_steps=20,
            attack_name="pgd",
        ),
        "MI-FGSM-20": AttackConfig(
            epsilon=eps,
            alpha=alpha,
            num_steps=20,
            attack_name="mifgsm",
            decay_factor=1.0,
        ),
        "FGSM": AttackConfig(
            epsilon=eps,
            alpha=eps,  # single-step: alpha == epsilon
            num_steps=1,
            attack_name="fgsm",
        ),
        # --- AIDE variants ---
        "AIDE-Base": AttackConfig(
            epsilon=eps,
            alpha=alpha,
            num_steps=20,
            attack_name="aide_base",
        ),
        "AIDE-Momentum": AttackConfig(
            epsilon=eps,
            alpha=alpha,
            num_steps=20,
            attack_name="aide_momentum",
            momentum=True,
        ),
        "AIDE-MultiScale": AttackConfig(
            epsilon=eps,
            alpha=alpha,
            num_steps=20,
            attack_name="aide_multiscale",
            multi_scale=True,
        ),
        "AIDE-Adaptive": AttackConfig(
            epsilon=eps,
            alpha=alpha,
            num_steps=20,
            attack_name="aide_adaptive",
            adaptive=True,
        ),
        "AIDE-Soft": AttackConfig(
            epsilon=eps,
            alpha=alpha,
            num_steps=20,
            attack_name="aide_soft",
            temperature=0.1,
        ),
    }
