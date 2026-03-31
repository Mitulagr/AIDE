#!/usr/bin/env python3
"""
AIDE Full Experiment Suite
==========================
Runs all experiments from the research plan:
  1. Main results table (all methods × models × datasets × epsilons)
  2. Pareto frontier (ASR vs LPIPS)
  3. Attention drift visualization
  4. Ablation study
  5. Defense evasion
  6. Transferability
  7. CAM method comparison
  8. Convergence analysis

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/run_all.py --experiment main
    CUDA_VISIBLE_DEVICES=2 python experiments/run_all.py --experiment all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import AttackConfig, EPSILON_VALUES, get_default_configs
from src.data import get_dataset, get_correctly_classified_subset, CIFAR10_CLASSES
from torch.utils.data import TensorDataset
from src.models import get_model, get_target_layers, load_or_train_model
from src.cam import get_cam
from src.attacks.pgd import pgd_attack
from src.attacks.fgsm import fgsm_attack
from src.attacks.mifgsm import mifgsm_attack
from src.attacks.aide import (
    aide_base_attack, aide_momentum_attack, aide_multiscale_attack,
    aide_adaptive_attack, aide_soft_attack,
)
from src.metrics import MetricsAccumulator, compute_mean_observed_dissimilarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _unpack_subset(model, dataset, device, num_images):
    """Get correctly classified images as (images_tensor, labels_tensor) on CPU."""
    tds = get_correctly_classified_subset(model, dataset, device, num_images=num_images)
    return tds.tensors[0], tds.tensors[1]


# ============================================================
# Attack dispatcher
# ============================================================

def run_attack(attack_name, model, images, labels, epsilon, alpha, num_steps,
               device, target_layers, cam_method="gradcam",
               record_drift=False, record_loss=False):
    """Dispatch to the appropriate attack function."""
    tl = target_layers[0] if target_layers else None

    kwargs = dict(record_loss=record_loss)

    if attack_name == "FGSM":
        return fgsm_attack(model, images, labels, epsilon, device)

    elif attack_name == "PGD-20":
        return pgd_attack(model, images, labels, epsilon, alpha, num_steps, device, **kwargs)

    elif attack_name == "MI-FGSM-20":
        return mifgsm_attack(model, images, labels, epsilon, alpha, num_steps, device,
                             decay_factor=1.0, **kwargs)

    elif attack_name == "AIDE-Base":
        return aide_base_attack(model, images, labels, epsilon, alpha, num_steps, device,
                                tl, cam_method=cam_method,
                                record_drift=record_drift, **kwargs)

    elif attack_name == "AIDE-Momentum":
        return aide_momentum_attack(model, images, labels, epsilon, alpha, num_steps, device,
                                    tl, cam_method=cam_method, decay_factor=1.0,
                                    record_drift=record_drift, **kwargs)

    elif attack_name == "AIDE-MultiScale":
        return aide_multiscale_attack(model, images, labels, epsilon, alpha, num_steps, device,
                                     target_layers, cam_method=cam_method,
                                     record_drift=record_drift, **kwargs)

    elif attack_name == "AIDE-Adaptive":
        return aide_adaptive_attack(model, images, labels, epsilon, alpha, num_steps, device,
                                    tl, cam_method=cam_method, stall_patience=3,
                                    record_drift=record_drift, **kwargs)

    elif attack_name == "AIDE-Soft":
        return aide_soft_attack(model, images, labels, epsilon, alpha, num_steps, device,
                                tl, cam_method=cam_method, temperature=0.1,
                                record_drift=record_drift, **kwargs)

    # Ablation variants
    elif attack_name == "Static-Inverse":
        # Static mask: compute GradCAM once on clean image, use same mask for all steps
        return _static_inverse_attack(model, images, labels, epsilon, alpha, num_steps,
                                      device, tl, cam_method, record_loss=record_loss)

    elif attack_name == "Dynamic-Direct":
        # Opposite of AIDE: perturb WHERE model looks (like AG2)
        return _dynamic_direct_attack(model, images, labels, epsilon, alpha, num_steps,
                                      device, tl, cam_method, record_loss=record_loss)

    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def _static_inverse_attack(model, images, labels, epsilon, alpha, num_steps,
                           device, target_layer, cam_method, record_loss=False):
    """Ablation: Static inverse saliency mask (compute once, reuse)."""
    images, labels = images.to(device), labels.to(device)

    # Compute CAM on clean image ONCE
    cam_obj = get_cam(cam_method, model, target_layer)
    cam = cam_obj.compute(images)
    cam_obj.remove_hooks()
    inverse_mask = 1.0 - cam

    adv = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, 0.0, 1.0)
    losses = [] if record_loss else None

    for _ in range(num_steps):
        adv_input = adv.clone().detach().requires_grad_(True)
        outputs = model(adv_input)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        grad = adv_input.grad.detach()

        if record_loss:
            losses.append(loss.item())

        adv = adv + alpha * grad.sign() * inverse_mask
        delta = torch.clamp(adv - images, -epsilon, epsilon)
        adv = torch.clamp(images + delta, 0.0, 1.0).detach()

    if record_loss:
        return adv, losses
    return adv


def _dynamic_direct_attack(model, images, labels, epsilon, alpha, num_steps,
                           device, target_layer, cam_method, record_loss=False):
    """Ablation: Dynamic DIRECT saliency (perturb where model looks, like AG2)."""
    images, labels = images.to(device), labels.to(device)

    adv = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, 0.0, 1.0)
    losses = [] if record_loss else None

    for _ in range(num_steps):
        # CAM on current adversarial
        cam_obj = get_cam(cam_method, model, target_layer)
        cam = cam_obj.compute(adv.detach().clone().requires_grad_(True))
        cam_obj.remove_hooks()
        direct_mask = cam.detach()  # HIGH where model attends

        adv_input = adv.clone().detach().requires_grad_(True)
        outputs = model(adv_input)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        grad = adv_input.grad.detach()

        if record_loss:
            losses.append(loss.item())

        adv = adv + alpha * grad.sign() * direct_mask
        delta = torch.clamp(adv - images, -epsilon, epsilon)
        adv = torch.clamp(images + delta, 0.0, 1.0).detach()

    if record_loss:
        return adv, losses
    return adv


# ============================================================
# Experiment 1: Main Results Table
# ============================================================

def experiment_main_results(device, datasets=None, model_names=None,
                            attack_names=None, epsilons=None, num_images=1000):
    """Run main results: all methods × models × datasets × epsilons."""
    if datasets is None:
        datasets = ["cifar10"]
    if model_names is None:
        model_names = ["resnet50", "vgg19", "densenet121"]
    if attack_names is None:
        attack_names = ["FGSM", "PGD-20", "MI-FGSM-20", "AIDE-Base",
                        "AIDE-Momentum", "AIDE-MultiScale"]
    if epsilons is None:
        epsilons = EPSILON_VALUES

    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")
        dataset = get_dataset(ds_name)

        for model_name in model_names:
            print(f"\n--- Model: {model_name} ---")
            model = load_or_train_model(model_name, ds_name, device)
            target_layers = get_target_layers(model, model_name)

            # Get correctly classified subset
            clean_images, clean_labels = _unpack_subset(
                model, dataset, device, num_images
            )
            print(f"  Collected {len(clean_labels)} correctly classified images")

            for eps in epsilons:
                eps_int = round(eps * 255)
                print(f"\n  Epsilon: {eps_int}/255")

                alpha = max(eps / 10, 1.0 / 255)  # Standard: eps/10 or at least 1/255
                if eps_int <= 4:
                    alpha = eps / 5  # Smaller epsilon needs relatively larger steps
                num_steps = 20

                for atk_name in attack_names:
                    print(f"    {atk_name}...", end=" ", flush=True)
                    t0 = time.time()

                    metrics = MetricsAccumulator(device)
                    batch_size = 32 if "AIDE" in atk_name else 64

                    n_batches = (len(clean_labels) + batch_size - 1) // batch_size
                    for bi in range(n_batches):
                        start = bi * batch_size
                        end = min(start + batch_size, len(clean_labels))
                        imgs = clean_images[start:end].to(device)
                        lbls = clean_labels[start:end].to(device)

                        adv = run_attack(atk_name, model, imgs, lbls, eps, alpha,
                                         num_steps, device, target_layers)
                        # Handle tuple returns
                        if isinstance(adv, tuple):
                            adv = adv[0]

                        metrics.update(imgs, adv, lbls, model)

                    result = metrics.compute()
                    elapsed = time.time() - t0
                    result["time_seconds"] = elapsed

                    key = f"{ds_name}/{model_name}/eps{eps_int}/{atk_name}"
                    all_results[key] = result

                    print(f"ASR={result['asr_mean']:.1f}% "
                          f"LPIPS={result['lpips_mean']:.4f} "
                          f"SSIM={result['ssim_mean']:.4f} "
                          f"[{elapsed:.1f}s]")

    # Save results
    results_path = RESULTS_DIR / "main_results.json"
    _save_json(all_results, results_path)
    print(f"\nResults saved to {results_path}")

    # Generate table
    _print_main_table(all_results)
    return all_results


def _print_main_table(results):
    """Pretty-print the main results table."""
    print(f"\n{'='*90}")
    print("MAIN RESULTS TABLE")
    print(f"{'='*90}")

    # Group by dataset/model/epsilon
    groups = {}
    for key, val in results.items():
        parts = key.split("/")
        ds, model, eps_str, atk = parts[0], parts[1], parts[2], parts[3]
        group_key = f"{ds}/{model}/{eps_str}"
        if group_key not in groups:
            groups[group_key] = {}
        groups[group_key][atk] = val

    for group_key, attacks in groups.items():
        print(f"\n{group_key}")
        print(f"{'Attack':<20} {'ASR%':>8} {'LPIPS':>10} {'SSIM':>10} {'L2':>10} {'Time':>8}")
        print("-" * 70)
        for atk_name, r in attacks.items():
            print(f"{atk_name:<20} {r['asr_mean']:>7.1f}% "
                  f"{r['lpips_mean']:>9.4f} {r['ssim_mean']:>9.4f} "
                  f"{r.get('l2_mean', 0):>9.4f} {r.get('time_seconds', 0):>7.1f}s")


# ============================================================
# Experiment 2: Pareto Frontier
# ============================================================

def experiment_pareto_frontier(device, num_images=1000):
    """Plot ASR vs LPIPS across all methods and epsilons."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Pareto Frontier (ASR vs LPIPS)")
    print("=" * 70)

    model = load_or_train_model("resnet50", "cifar10", device)
    target_layers = get_target_layers(model, "resnet50")
    dataset = get_dataset("cifar10")
    clean_images, clean_labels = _unpack_subset(model, dataset, device, num_images)

    attacks = ["PGD-20", "MI-FGSM-20", "AIDE-Base", "AIDE-Momentum", "AIDE-Soft"]
    colors = {"PGD-20": "#e74c3c", "MI-FGSM-20": "#f39c12", "AIDE-Base": "#3498db",
              "AIDE-Momentum": "#2ecc71", "AIDE-Soft": "#9b59b6"}
    markers = {"PGD-20": "o", "MI-FGSM-20": "s", "AIDE-Base": "^",
               "AIDE-Momentum": "D", "AIDE-Soft": "v"}

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    pareto_data = {}

    for atk_name in attacks:
        asr_list, lpips_list = [], []

        for eps in EPSILON_VALUES:
            eps_int = round(eps * 255)
            alpha = max(eps / 10, 1.0 / 255)
            if eps_int <= 4:
                alpha = eps / 5

            print(f"  {atk_name} @ eps={eps_int}/255...", end=" ", flush=True)
            metrics = MetricsAccumulator(device)

            batch_size = 32 if "AIDE" in atk_name else 64
            n_batches = (len(clean_labels) + batch_size - 1) // batch_size
            for bi in range(n_batches):
                start = bi * batch_size
                end = min(start + batch_size, len(clean_labels))
                imgs = clean_images[start:end].to(device)
                lbls = clean_labels[start:end].to(device)

                adv = run_attack(atk_name, model, imgs, lbls, eps, alpha, 20,
                                 device, target_layers)
                if isinstance(adv, tuple):
                    adv = adv[0]
                metrics.update(imgs, adv, lbls, model)

            r = metrics.compute()
            asr_list.append(r["asr_mean"])
            lpips_list.append(r["lpips_mean"])
            print(f"ASR={r['asr_mean']:.1f}% LPIPS={r['lpips_mean']:.4f}")

        pareto_data[atk_name] = {"asr": asr_list, "lpips": lpips_list,
                                  "epsilons": [round(e * 255) for e in EPSILON_VALUES]}

        ax.plot(lpips_list, asr_list, f"-{markers[atk_name]}",
                color=colors[atk_name], label=atk_name, markersize=8, linewidth=2)

        # Annotate epsilon values
        for i, eps_int in enumerate([round(e * 255) for e in EPSILON_VALUES]):
            ax.annotate(f"ε={eps_int}", (lpips_list[i], asr_list[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("LPIPS (lower = more imperceptible)", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("Pareto Frontier: ASR vs Perceptual Quality", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pareto_frontier.png", dpi=200, bbox_inches="tight")
    plt.close()

    _save_json(pareto_data, RESULTS_DIR / "pareto_data.json")
    print(f"Saved Pareto frontier to {RESULTS_DIR}/pareto_frontier.png")
    return pareto_data


# ============================================================
# Experiment 3: Attention Drift Visualization
# ============================================================

def experiment_attention_drift(device, num_images=5):
    """Hero figure: show GradCAM drift across PGD steps."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Attention Drift Visualization")
    print("=" * 70)

    model = load_or_train_model("resnet50", "cifar10", device)
    target_layers = get_target_layers(model, "resnet50")
    dataset = get_dataset("cifar10")

    vis_steps = [0, 4, 9, 14, 19]
    collected = 0

    all_clean, all_adv, all_drift, all_masks, all_labels = [], [], [], [], []

    for i in range(len(dataset)):
        if collected >= num_images:
            break
        img, lbl = dataset[i]
        img = img.unsqueeze(0).to(device)
        lbl_t = torch.tensor([lbl]).to(device)

        with torch.no_grad():
            if model(img).argmax(1).item() != lbl:
                continue

        adv, drift = aide_base_attack(
            model, img, lbl_t, 8 / 255, 2 / 255, 20, device,
            target_layers[0], record_drift=True
        )

        all_clean.append(img.cpu().numpy()[0])
        all_adv.append(adv.cpu().numpy()[0])
        all_drift.append(drift)
        all_labels.append(lbl)
        collected += 1
        print(f"  Collected image {collected}/{num_images} (class: {CIFAR10_CLASSES[lbl]})")

    # Plot
    n_cols = 2 + len(vis_steps) + 1  # clean + cams + adv + perturbation
    fig, axes = plt.subplots(num_images, n_cols, figsize=(n_cols * 2.5, num_images * 2.5))
    if num_images == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Clean"] + [f"CAM Step {s+1}" for s in vis_steps] + ["Adversarial", "Perturbation (10×)"]

    for row in range(num_images):
        clean_img = np.transpose(all_clean[row], (1, 2, 0))
        adv_img = np.transpose(all_adv[row], (1, 2, 0))
        pert = np.clip(np.abs(adv_img - clean_img) * 10, 0, 1)

        axes[row, 0].imshow(np.clip(clean_img, 0, 1))
        axes[row, 0].set_ylabel(CIFAR10_CLASSES[all_labels[row]], fontsize=10)

        for ci, step in enumerate(vis_steps):
            cam = all_drift[row][step][0, 0]
            axes[row, 1 + ci].imshow(np.clip(clean_img, 0, 1), alpha=0.4)
            axes[row, 1 + ci].imshow(cam, cmap="jet", alpha=0.6, vmin=0, vmax=1)

        axes[row, -2].imshow(np.clip(adv_img, 0, 1))
        axes[row, -1].imshow(pert)

    for col in range(n_cols):
        axes[0, col].set_title(col_titles[col], fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("AIDE Attention Drift: GradCAM evolves as AIDE perturbs non-salient regions",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "attention_drift_hero.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved attention drift figure to {RESULTS_DIR}/attention_drift_hero.png")


# ============================================================
# Experiment 4: Ablation Study
# ============================================================

def experiment_ablation(device, num_images=1000):
    """Ablation: Static vs Dynamic, Inverse vs Direct, variants."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Ablation Study")
    print("=" * 70)

    model = load_or_train_model("resnet50", "cifar10", device)
    target_layers = get_target_layers(model, "resnet50")
    dataset = get_dataset("cifar10")
    clean_images, clean_labels = _unpack_subset(model, dataset, device, num_images)

    ablation_attacks = [
        "Static-Inverse",    # Your original DG-FGSM approach (static mask)
        "Dynamic-Direct",    # AG2-like (perturb where model looks)
        "PGD-20",            # No masking at all
        "AIDE-Base",         # Dynamic inverse (core method)
        "AIDE-MultiScale",   # + multi-layer CAM
        "AIDE-Momentum",     # + momentum
        "AIDE-Soft",         # + temperature softmax
        "AIDE-Adaptive",     # + stall detection
    ]

    eps = 8 / 255
    alpha = 2 / 255
    results = {}

    for atk_name in ablation_attacks:
        print(f"  {atk_name}...", end=" ", flush=True)
        metrics = MetricsAccumulator(device)

        batch_size = 32 if atk_name not in ("PGD-20",) else 64
        n_batches = (len(clean_labels) + batch_size - 1) // batch_size
        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, len(clean_labels))
            imgs = clean_images[start:end].to(device)
            lbls = clean_labels[start:end].to(device)

            adv = run_attack(atk_name, model, imgs, lbls, eps, alpha, 20,
                             device, target_layers)
            if isinstance(adv, tuple):
                adv = adv[0]
            metrics.update(imgs, adv, lbls, model)

        r = metrics.compute()
        results[atk_name] = r
        print(f"ASR={r['asr_mean']:.1f}% LPIPS={r['lpips_mean']:.4f} SSIM={r['ssim_mean']:.4f}")

    _save_json(results, RESULTS_DIR / "ablation_results.json")

    # Print table
    print(f"\n{'Attack':<20} {'Dynamic?':>8} {'Inverse?':>8} {'ASR%':>8} {'LPIPS':>10} {'SSIM':>10}")
    print("-" * 70)
    ablation_meta = {
        "Static-Inverse": ("No", "Yes"), "Dynamic-Direct": ("Yes", "No"),
        "PGD-20": ("N/A", "N/A"), "AIDE-Base": ("Yes", "Yes"),
        "AIDE-MultiScale": ("Yes", "Yes"), "AIDE-Momentum": ("Yes", "Yes"),
        "AIDE-Soft": ("Yes", "Yes"), "AIDE-Adaptive": ("Yes", "Yes"),
    }
    for atk_name, r in results.items():
        dyn, inv = ablation_meta.get(atk_name, ("?", "?"))
        print(f"{atk_name:<20} {dyn:>8} {inv:>8} {r['asr_mean']:>7.1f}% "
              f"{r['lpips_mean']:>9.4f} {r['ssim_mean']:>9.4f}")

    return results


# ============================================================
# Experiment 5: Defense Evasion
# ============================================================

def experiment_defense_evasion(device, num_images=500):
    """Evaluate how well different attacks evade saliency-based detection."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Defense Evasion")
    print("=" * 70)

    model = load_or_train_model("resnet50", "cifar10", device)
    target_layers = get_target_layers(model, "resnet50")
    dataset = get_dataset("cifar10")
    clean_images, clean_labels = _unpack_subset(model, dataset, device, num_images)

    attacks = ["PGD-20", "AIDE-Base", "AIDE-Momentum", "Dynamic-Direct"]
    eps, alpha = 8 / 255, 2 / 255
    results = {}

    for atk_name in attacks:
        print(f"\n  {atk_name}:")
        cam_sims = []  # Cosine similarity between clean and adv GradCAMs
        detection_scores = []  # L2 of GradCAM difference (proxy for detection)

        batch_size = 16
        n_batches = (len(clean_labels) + batch_size - 1) // batch_size
        for bi in tqdm(range(n_batches), desc=f"    {atk_name}"):
            start = bi * batch_size
            end = min(start + batch_size, len(clean_labels))
            imgs = clean_images[start:end].to(device)
            lbls = clean_labels[start:end].to(device)

            # Generate adversarial
            adv = run_attack(atk_name, model, imgs, lbls, eps, alpha, 20,
                             device, target_layers)
            if isinstance(adv, tuple):
                adv = adv[0]

            # Compute GradCAM on clean
            cam_clean_obj = get_cam("gradcam", model, target_layers[0])
            cam_clean = cam_clean_obj.compute(imgs)
            cam_clean_obj.remove_hooks()

            # Compute GradCAM on adversarial
            cam_adv_obj = get_cam("gradcam", model, target_layers[0])
            cam_adv = cam_adv_obj.compute(adv)
            cam_adv_obj.remove_hooks()

            # Cosine similarity per image
            B = imgs.shape[0]
            for b in range(B):
                c_flat = cam_clean[b].flatten()
                a_flat = cam_adv[b].flatten()
                cos_sim = F.cosine_similarity(c_flat.unsqueeze(0), a_flat.unsqueeze(0)).item()
                cam_sims.append(cos_sim)
                l2_diff = (c_flat - a_flat).norm(2).item()
                detection_scores.append(l2_diff)

        results[atk_name] = {
            "cam_similarity_mean": float(np.mean(cam_sims)),
            "cam_similarity_std": float(np.std(cam_sims)),
            "detection_score_mean": float(np.mean(detection_scores)),
            "detection_score_std": float(np.std(detection_scores)),
        }
        print(f"    CAM Similarity: {np.mean(cam_sims):.4f}±{np.std(cam_sims):.4f}")
        print(f"    Detection Score (L2): {np.mean(detection_scores):.4f}±{np.std(detection_scores):.4f}")

    # Feature squeezing defense
    print("\n  Feature Squeezing Defense:")
    for atk_name in attacks:
        detected = 0
        total = 0
        batch_size = 32
        n_batches = (len(clean_labels) + batch_size - 1) // batch_size

        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, len(clean_labels))
            imgs = clean_images[start:end].to(device)
            lbls = clean_labels[start:end].to(device)

            adv = run_attack(atk_name, model, imgs, lbls, eps, alpha, 20,
                             device, target_layers)
            if isinstance(adv, tuple):
                adv = adv[0]

            # Feature squeezing: reduce bit depth to 4 bits
            squeezed = torch.round(adv * 15) / 15

            with torch.no_grad():
                pred_orig = model(adv).argmax(1)
                pred_squeezed = model(squeezed).argmax(1)

            # If predictions differ, it's detected as adversarial
            detected += (pred_orig != pred_squeezed).sum().item()
            total += imgs.shape[0]

        det_rate = detected / total * 100
        results[atk_name]["feature_squeeze_detection_rate"] = det_rate
        print(f"    {atk_name}: Detection rate = {det_rate:.1f}%")

    _save_json(results, RESULTS_DIR / "defense_evasion_results.json")
    return results


# ============================================================
# Experiment 6: Transferability
# ============================================================

def experiment_transferability(device, num_images=1000):
    """Generate on ResNet-50, test transfer to other models."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Transferability (Black-Box)")
    print("=" * 70)

    source_model = load_or_train_model("resnet50", "cifar10", device)
    source_layers = get_target_layers(source_model, "resnet50")
    dataset = get_dataset("cifar10")

    # Get images correctly classified by ALL models
    target_model_names = ["vgg19", "densenet121", "mobilenetv2"]
    target_models = {}
    for name in target_model_names:
        target_models[name] = load_or_train_model(name, "cifar10", device)

    clean_images, clean_labels = _unpack_subset(source_model, dataset, device, num_images)

    attacks = ["PGD-20", "MI-FGSM-20", "AIDE-Base", "AIDE-Momentum"]
    eps, alpha = 8 / 255, 2 / 255
    results = {}

    for atk_name in attacks:
        print(f"\n  Generating with {atk_name}...")

        all_adv = []
        batch_size = 32 if "AIDE" in atk_name else 64
        n_batches = (len(clean_labels) + batch_size - 1) // batch_size

        for bi in tqdm(range(n_batches), desc=f"    {atk_name}"):
            start = bi * batch_size
            end = min(start + batch_size, len(clean_labels))
            imgs = clean_images[start:end].to(device)
            lbls = clean_labels[start:end].to(device)

            adv = run_attack(atk_name, source_model, imgs, lbls, eps, alpha, 20,
                             device, source_layers)
            if isinstance(adv, tuple):
                adv = adv[0]
            all_adv.append(adv.cpu())

        all_adv = torch.cat(all_adv, dim=0)

        # Test on source model (white-box ASR)
        with torch.no_grad():
            wb_correct = 0
            for bi in range(0, len(clean_labels), 64):
                end = min(bi + 64, len(clean_labels))
                pred = source_model(all_adv[bi:end].to(device)).argmax(1)
                wb_correct += (pred == clean_labels[bi:end].to(device)).sum().item()
        wb_asr = (1 - wb_correct / len(clean_labels)) * 100

        transfer_results = {"white_box_asr": wb_asr}

        # Test on target models
        for tgt_name, tgt_model in target_models.items():
            with torch.no_grad():
                correct = 0
                for bi in range(0, len(clean_labels), 64):
                    end = min(bi + 64, len(clean_labels))
                    pred = tgt_model(all_adv[bi:end].to(device)).argmax(1)
                    correct += (pred == clean_labels[bi:end].to(device)).sum().item()
            transfer_asr = (1 - correct / len(clean_labels)) * 100
            transfer_results[f"transfer_{tgt_name}"] = transfer_asr

        results[atk_name] = transfer_results
        print(f"    WB ASR: {wb_asr:.1f}%, " +
              ", ".join(f"{k}: {v:.1f}%" for k, v in transfer_results.items() if k != "white_box_asr"))

    _save_json(results, RESULTS_DIR / "transferability_results.json")

    # Print table
    print(f"\n{'Attack':<20} {'WB ASR':>10} " +
          " ".join(f"{n:>12}" for n in target_model_names))
    print("-" * 70)
    for atk, r in results.items():
        line = f"{atk:<20} {r['white_box_asr']:>9.1f}%"
        for n in target_model_names:
            line += f" {r.get(f'transfer_{n}', 0):>11.1f}%"
        print(line)

    return results


# ============================================================
# Experiment 7: CAM Method Comparison
# ============================================================

def experiment_cam_comparison(device, num_images=500):
    """Compare GradCAM, GradCAM++, LayerCAM in AIDE."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: CAM Method Comparison")
    print("=" * 70)

    model = load_or_train_model("resnet50", "cifar10", device)
    target_layers = get_target_layers(model, "resnet50")
    dataset = get_dataset("cifar10")
    clean_images, clean_labels = _unpack_subset(model, dataset, device, num_images)

    cam_methods = ["gradcam", "gradcam++", "layercam"]
    eps, alpha = 8 / 255, 2 / 255
    results = {}

    for cm in cam_methods:
        print(f"  AIDE-Base with {cm}...", end=" ", flush=True)
        metrics = MetricsAccumulator(device)

        batch_size = 32
        n_batches = (len(clean_labels) + batch_size - 1) // batch_size
        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, len(clean_labels))
            imgs = clean_images[start:end].to(device)
            lbls = clean_labels[start:end].to(device)

            adv = aide_base_attack(model, imgs, lbls, eps, alpha, 20, device,
                                   target_layers[0], cam_method=cm)
            if isinstance(adv, tuple):
                adv = adv[0]
            metrics.update(imgs, adv, lbls, model)

        r = metrics.compute()
        results[cm] = r
        print(f"ASR={r['asr_mean']:.1f}% LPIPS={r['lpips_mean']:.4f} SSIM={r['ssim_mean']:.4f}")

    _save_json(results, RESULTS_DIR / "cam_comparison_results.json")
    return results


# ============================================================
# Experiment 8: Convergence Analysis
# ============================================================

def experiment_convergence(device, num_images=200):
    """Plot loss curves over iterations for PGD vs AIDE variants."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Convergence Analysis")
    print("=" * 70)

    model = load_or_train_model("resnet50", "cifar10", device)
    target_layers = get_target_layers(model, "resnet50")
    dataset = get_dataset("cifar10")
    clean_images, clean_labels = _unpack_subset(model, dataset, device, num_images)

    attacks = ["PGD-20", "AIDE-Base", "AIDE-Momentum", "AIDE-Adaptive"]
    colors = {"PGD-20": "#e74c3c", "AIDE-Base": "#3498db",
              "AIDE-Momentum": "#2ecc71", "AIDE-Adaptive": "#f39c12"}
    eps, alpha = 8 / 255, 2 / 255

    all_losses = {}

    for atk_name in attacks:
        print(f"  {atk_name}...", end=" ", flush=True)
        step_losses = [[] for _ in range(20)]

        batch_size = 32
        n_batches = (len(clean_labels) + batch_size - 1) // batch_size
        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, len(clean_labels))
            imgs = clean_images[start:end].to(device)
            lbls = clean_labels[start:end].to(device)

            result = run_attack(atk_name, model, imgs, lbls, eps, alpha, 20,
                                device, target_layers, record_loss=True)
            if isinstance(result, tuple):
                _, losses = result[0], result[1]
            else:
                continue

            for s, l in enumerate(losses):
                step_losses[s].append(l)

        mean_losses = [np.mean(sl) if sl else 0 for sl in step_losses]
        all_losses[atk_name] = mean_losses
        print(f"Final loss: {mean_losses[-1]:.4f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for atk_name, losses in all_losses.items():
        ax.plot(range(1, 21), losses, "-o", color=colors.get(atk_name, "#333"),
                label=atk_name, markersize=4, linewidth=2)

    ax.set_xlabel("PGD Step", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Convergence: Loss per Step (ε=8/255, CIFAR-10, ResNet-50)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "convergence_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()

    _save_json({k: v for k, v in all_losses.items()}, RESULTS_DIR / "convergence_data.json")
    print(f"Saved convergence plot to {RESULTS_DIR}/convergence_analysis.png")
    return all_losses


# ============================================================
# Utilities
# ============================================================

def _save_json(data, path):
    """Save dict to JSON, handling numpy types."""
    def convert(o):
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)


def update_progress(experiment_name, status, details=""):
    """Append to RESEARCH_PROGRESS.md."""
    progress_file = ROOT / "RESEARCH_PROGRESS.md"
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    with open(progress_file, "a") as f:
        f.write(f"\n**{timestamp} - {experiment_name}:** {status}")
        if details:
            f.write(f" {details}")
        f.write("\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AIDE Experiment Suite")
    parser.add_argument("--experiment", type=str, default="main",
                        choices=["main", "pareto", "drift", "ablation", "defense",
                                 "transfer", "cam", "convergence", "all"],
                        help="Which experiment to run")
    parser.add_argument("--num-images", type=int, default=1000,
                        help="Number of images to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (set via CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")

    start = time.time()

    if args.experiment in ("main", "all"):
        update_progress("Experiment 1", "STARTED", "Main results table")
        experiment_main_results(device, num_images=args.num_images)
        update_progress("Experiment 1", "COMPLETED")

    if args.experiment in ("pareto", "all"):
        update_progress("Experiment 2", "STARTED", "Pareto frontier")
        experiment_pareto_frontier(device, num_images=args.num_images)
        update_progress("Experiment 2", "COMPLETED")

    if args.experiment in ("drift", "all"):
        update_progress("Experiment 3", "STARTED", "Attention drift visualization")
        experiment_attention_drift(device, num_images=5)
        update_progress("Experiment 3", "COMPLETED")

    if args.experiment in ("ablation", "all"):
        update_progress("Experiment 4", "STARTED", "Ablation study")
        experiment_ablation(device, num_images=args.num_images)
        update_progress("Experiment 4", "COMPLETED")

    if args.experiment in ("defense", "all"):
        update_progress("Experiment 5", "STARTED", "Defense evasion")
        experiment_defense_evasion(device, num_images=min(args.num_images, 500))
        update_progress("Experiment 5", "COMPLETED")

    if args.experiment in ("transfer", "all"):
        update_progress("Experiment 6", "STARTED", "Transferability")
        experiment_transferability(device, num_images=args.num_images)
        update_progress("Experiment 6", "COMPLETED")

    if args.experiment in ("cam", "all"):
        update_progress("Experiment 7", "STARTED", "CAM comparison")
        experiment_cam_comparison(device, num_images=min(args.num_images, 500))
        update_progress("Experiment 7", "COMPLETED")

    if args.experiment in ("convergence", "all"):
        update_progress("Experiment 8", "STARTED", "Convergence analysis")
        experiment_convergence(device, num_images=min(args.num_images, 200))
        update_progress("Experiment 8", "COMPLETED")

    total = time.time() - start
    print(f"\nTotal runtime: {total / 60:.1f} minutes")
    update_progress("Suite", "DONE", f"Total: {total/60:.1f} minutes")


if __name__ == "__main__":
    main()
