#!/usr/bin/env python3
"""
AIDE ImageNet (ImageNette) Experiments
======================================
Runs AIDE experiments on 224x224 images where spatial masking has
a much larger effect than on 32x32 CIFAR-10.

Uses ImageNette (10-class ImageNet subset) with pretrained models.

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/run_imagenet.py --num-images 500
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
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cam import get_cam
from src.attacks.pgd import pgd_attack
from src.attacks.fgsm import fgsm_attack
from src.attacks.mifgsm import mifgsm_attack
from src.attacks.aide import (
    aide_base_attack, aide_momentum_attack, aide_multiscale_attack,
)
from src.metrics import MetricsAccumulator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = ROOT / "results" / "imagenet"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ImageNette class names for visualization
IMAGENETTE_CLASSES = [
    "tench", "springer", "cassette", "chainsaw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]

# ImageNette folder names -> ImageNet class indices (for pretrained models)
IMAGENETTE_TO_IMAGENET = {
    "n01440764": 0,    "n02102040": 217,  "n02979186": 482,
    "n03000684": 491,  "n03028079": 497,  "n03394916": 566,
    "n03417042": 569,  "n03425413": 571,  "n03445777": 574,
    "n03888257": 701,
}


# ============================================================
# Data loading
# ============================================================

def load_imagenette(data_root, num_images=500, device="cuda:0"):
    """Load ImageNette val set, return (images, imagenet_labels) on CPU.

    Labels are remapped from ImageFolder indices (0-9) to actual ImageNet
    class indices so pretrained models can classify correctly.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    ds = datasets.ImageFolder(
        os.path.join(data_root, "imagenette2-320", "val"),
        transform=transform,
    )

    # Build label remapping: folder_idx -> imagenet_idx
    folder_to_inet = {}
    for folder_name, folder_idx in ds.class_to_idx.items():
        folder_to_inet[folder_idx] = IMAGENETTE_TO_IMAGENET[folder_name]

    return ds, folder_to_inet


def get_correctly_classified(model, dataset, folder_to_inet, device, num_images=500):
    """Collect correctly classified images with remapped labels."""
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    images_list, labels_list, local_labels_list = [], [], []
    collected = 0

    with torch.no_grad():
        for imgs, lbls in loader:
            # Remap labels to ImageNet indices
            inet_lbls = torch.tensor([folder_to_inet[l.item()] for l in lbls])
            imgs_gpu = imgs.to(device)
            preds = model(imgs_gpu).argmax(1).cpu()
            mask = preds == inet_lbls

            if mask.any():
                images_list.append(imgs[mask])
                labels_list.append(inet_lbls[mask])
                local_labels_list.append(lbls[mask])  # 0-9 for visualization
                collected += mask.sum().item()

            if collected >= num_images:
                break

    images = torch.cat(images_list)[:num_images]
    labels = torch.cat(labels_list)[:num_images]
    local_labels = torch.cat(local_labels_list)[:num_images]
    return images, labels, local_labels


# ============================================================
# Attack dispatcher
# ============================================================

def run_attack(name, model, images, labels, eps, alpha, steps, device,
               target_layers, record_loss=False, record_drift=False):
    tl = target_layers[0] if target_layers else None
    kw = dict(record_loss=record_loss)

    if name == "FGSM":
        return fgsm_attack(model, images, labels, eps, device)
    elif name == "PGD-20":
        return pgd_attack(model, images, labels, eps, alpha, steps, device, **kw)
    elif name == "MI-FGSM-20":
        return mifgsm_attack(model, images, labels, eps, alpha, steps, device,
                             decay_factor=1.0, **kw)
    elif name == "AIDE-Base":
        return aide_base_attack(model, images, labels, eps, alpha, steps, device,
                                tl, cam_method="gradcam",
                                record_drift=record_drift, **kw)
    elif name == "AIDE-MultiScale":
        return aide_multiscale_attack(model, images, labels, eps, alpha, steps, device,
                                     target_layers, cam_method="gradcam",
                                     record_drift=record_drift, **kw)
    elif name == "AIDE-Momentum":
        return aide_momentum_attack(model, images, labels, eps, alpha, steps, device,
                                    tl, cam_method="gradcam", decay_factor=1.0,
                                    record_drift=record_drift, **kw)
    else:
        raise ValueError(f"Unknown attack: {name}")


# ============================================================
# Experiment 1: Main Results Table
# ============================================================

def experiment_main_results(device, num_images=500):
    """Run main results on ImageNette with pretrained models."""
    print("\n" + "=" * 70)
    print("ImageNet EXPERIMENT 1: Main Results Table")
    print("=" * 70)

    model_configs = [
        ("resnet50", models.resnet50),
        ("densenet121", models.densenet121),
    ]

    attack_names = ["FGSM", "PGD-20", "MI-FGSM-20", "AIDE-Base", "AIDE-MultiScale"]
    epsilons = [2 / 255, 4 / 255, 8 / 255, 16 / 255]
    all_results = {}

    data_root = str(ROOT / "data")
    ds, folder_to_inet = load_imagenette(data_root)

    for model_name, model_factory in model_configs:
        print(f"\n--- Model: {model_name} (pretrained ImageNet) ---")
        model = model_factory(weights="DEFAULT").to(device).eval()

        # Get target layers for CAM
        if model_name == "resnet50":
            target_layers = [model.layer4[-1]]  # 7x7 for 224x224 input
        elif model_name == "densenet121":
            target_layers = [model.features.denseblock4]
        else:
            target_layers = []

        # Collect correctly classified images
        clean_images, clean_labels, local_labels = get_correctly_classified(
            model, ds, folder_to_inet, device, num_images
        )
        print(f"  Collected {len(clean_labels)} correctly classified images")

        for eps in epsilons:
            eps_int = round(eps * 255)
            alpha = max(eps / 10, 1.0 / 255)
            if eps_int <= 4:
                alpha = eps / 5
            num_steps = 20

            print(f"\n  Epsilon: {eps_int}/255")

            for atk_name in attack_names:
                print(f"    {atk_name}...", end=" ", flush=True)
                t0 = time.time()
                metrics = MetricsAccumulator(device)

                # Smaller batches for 224x224 images (GPU memory)
                batch_size = 8 if "AIDE" in atk_name else 16
                n_batches = (len(clean_labels) + batch_size - 1) // batch_size

                for bi in range(n_batches):
                    start = bi * batch_size
                    end = min(start + batch_size, len(clean_labels))
                    imgs = clean_images[start:end].to(device)
                    lbls = clean_labels[start:end].to(device)

                    adv = run_attack(atk_name, model, imgs, lbls, eps, alpha,
                                     num_steps, device, target_layers)
                    if isinstance(adv, tuple):
                        adv = adv[0]
                    metrics.update(imgs, adv, lbls, model)

                result = metrics.compute()
                elapsed = time.time() - t0
                result["time_seconds"] = elapsed

                key = f"imagenette/{model_name}/eps{eps_int}/{atk_name}"
                all_results[key] = result

                print(f"ASR={result['asr_mean']:.1f}% "
                      f"LPIPS={result['lpips_mean']:.4f} "
                      f"SSIM={result['ssim_mean']:.4f} "
                      f"[{elapsed:.1f}s]")

        del model
        torch.cuda.empty_cache()

    _save_json(all_results, RESULTS_DIR / "main_results.json")
    _print_table(all_results)
    return all_results


# ============================================================
# Experiment 2: Attention Drift Visualization (224x224 hero)
# ============================================================

def experiment_attention_drift(device, num_images=5):
    """Hero figure at 224x224 - attention drift is much more visible."""
    print("\n" + "=" * 70)
    print("ImageNet EXPERIMENT 2: Attention Drift Visualization (224x224)")
    print("=" * 70)

    model = models.resnet50(weights="DEFAULT").to(device).eval()
    target_layers = [model.layer4[-1]]

    data_root = str(ROOT / "data")
    ds, folder_to_inet = load_imagenette(data_root)

    vis_steps = [0, 4, 9, 14, 19]  # steps 1, 5, 10, 15, 20
    all_clean, all_adv, all_drift, all_labels = [], [], [], []
    collected = 0

    # Pick diverse classes
    seen_classes = set()
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    for img, lbl in loader:
        if collected >= num_images:
            break

        inet_lbl = folder_to_inet[lbl.item()]
        if lbl.item() in seen_classes:
            continue

        img_gpu = img.to(device)
        lbl_t = torch.tensor([inet_lbl]).to(device)

        with torch.no_grad():
            if model(img_gpu).argmax(1).item() != inet_lbl:
                continue

        adv, drift = aide_base_attack(
            model, img_gpu, lbl_t, 8 / 255, 2 / 255, 20, device,
            target_layers[0], record_drift=True
        )

        all_clean.append(img.numpy()[0])
        all_adv.append(adv.cpu().numpy()[0])
        all_drift.append(drift)
        all_labels.append(lbl.item())
        seen_classes.add(lbl.item())
        collected += 1
        print(f"  Collected {collected}/{num_images} ({IMAGENETTE_CLASSES[lbl.item()]})")

    # Plot
    n_cols = 2 + len(vis_steps) + 1
    fig, axes = plt.subplots(num_images, n_cols, figsize=(n_cols * 3, num_images * 3))
    if num_images == 1:
        axes = axes[np.newaxis, :]

    col_titles = (["Clean"] +
                  [f"CAM Step {s+1}" for s in vis_steps] +
                  ["Adversarial", "Perturbation (10x)"])

    for row in range(num_images):
        clean_img = np.transpose(all_clean[row], (1, 2, 0))
        adv_img = np.transpose(all_adv[row], (1, 2, 0))
        pert = np.clip(np.abs(adv_img - clean_img) * 10, 0, 1)

        axes[row, 0].imshow(np.clip(clean_img, 0, 1))
        axes[row, 0].set_ylabel(IMAGENETTE_CLASSES[all_labels[row]], fontsize=11)

        for ci, step in enumerate(vis_steps):
            cam = all_drift[row][step][0, 0]
            axes[row, 1 + ci].imshow(np.clip(clean_img, 0, 1), alpha=0.4)
            axes[row, 1 + ci].imshow(cam, cmap="jet", alpha=0.6, vmin=0, vmax=1)

        axes[row, -2].imshow(np.clip(adv_img, 0, 1))
        axes[row, -1].imshow(pert)

    for col in range(n_cols):
        axes[0, col].set_title(col_titles[col], fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        "AIDE Attention Drift on ImageNette (224x224): GradCAM evolves as AIDE perturbs non-salient regions",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "attention_drift_224.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {RESULTS_DIR}/attention_drift_224.png")


# ============================================================
# Experiment 3: Perturbation heatmap comparison
# ============================================================

def experiment_perturbation_comparison(device, num_images=5):
    """Side-by-side PGD vs AIDE perturbation overlaid on GradCAM."""
    print("\n" + "=" * 70)
    print("ImageNet EXPERIMENT 3: Perturbation vs Saliency Comparison")
    print("=" * 70)

    model = models.resnet50(weights="DEFAULT").to(device).eval()
    target_layers = [model.layer4[-1]]

    data_root = str(ROOT / "data")
    ds, folder_to_inet = load_imagenette(data_root)

    all_data = []
    seen_classes = set()
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    for img, lbl in loader:
        if len(all_data) >= num_images:
            break
        inet_lbl = folder_to_inet[lbl.item()]
        if lbl.item() in seen_classes:
            continue

        img_gpu = img.to(device)
        lbl_t = torch.tensor([inet_lbl]).to(device)

        with torch.no_grad():
            if model(img_gpu).argmax(1).item() != inet_lbl:
                continue

        # Compute clean GradCAM
        cam_obj = get_cam("gradcam", model, target_layers[0])
        cam_clean = cam_obj.compute(img_gpu).cpu().numpy()[0, 0]
        cam_obj.remove_hooks()

        # PGD attack
        adv_pgd = pgd_attack(model, img_gpu, lbl_t, 8/255, 2/255, 20, device)
        if isinstance(adv_pgd, tuple):
            adv_pgd = adv_pgd[0]

        # AIDE attack
        adv_aide = aide_base_attack(model, img_gpu, lbl_t, 8/255, 2/255, 20, device,
                                     target_layers[0])
        if isinstance(adv_aide, tuple):
            adv_aide = adv_aide[0]

        all_data.append({
            "clean": img.numpy()[0],
            "cam_clean": cam_clean,
            "adv_pgd": adv_pgd.cpu().numpy()[0],
            "adv_aide": adv_aide.cpu().numpy()[0],
            "label": lbl.item(),
        })
        seen_classes.add(lbl.item())
        print(f"  Collected {len(all_data)}/{num_images} ({IMAGENETTE_CLASSES[lbl.item()]})")

    # Plot: clean | GradCAM | PGD pert | AIDE pert | PGD pert×CAM | AIDE pert×CAM
    n_cols = 6
    fig, axes = plt.subplots(num_images, n_cols, figsize=(n_cols * 3, num_images * 3))
    if num_images == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Clean Image", "GradCAM (clean)",
                  "PGD Perturbation (5x)", "AIDE Perturbation (5x)",
                  "PGD Pert. × Saliency", "AIDE Pert. × Saliency"]

    for row, d in enumerate(all_data):
        clean = np.transpose(d["clean"], (1, 2, 0))
        pert_pgd = np.abs(np.transpose(d["adv_pgd"], (1, 2, 0)) - clean)
        pert_aide = np.abs(np.transpose(d["adv_aide"], (1, 2, 0)) - clean)

        # Perturbation magnitude per pixel
        pert_pgd_mag = pert_pgd.mean(axis=2)
        pert_aide_mag = pert_aide.mean(axis=2)

        # Overlap with saliency
        cam = d["cam_clean"]
        # Resize cam to match image
        from scipy.ndimage import zoom
        if cam.shape != pert_pgd_mag.shape:
            cam = zoom(cam, (pert_pgd_mag.shape[0] / cam.shape[0],
                             pert_pgd_mag.shape[1] / cam.shape[1]))

        overlap_pgd = pert_pgd_mag * cam
        overlap_aide = pert_aide_mag * cam

        axes[row, 0].imshow(np.clip(clean, 0, 1))
        axes[row, 0].set_ylabel(IMAGENETTE_CLASSES[d["label"]], fontsize=11)

        axes[row, 1].imshow(np.clip(clean, 0, 1), alpha=0.3)
        axes[row, 1].imshow(cam, cmap="jet", alpha=0.7, vmin=0, vmax=1)

        axes[row, 2].imshow(np.clip(pert_pgd * 5, 0, 1))
        axes[row, 3].imshow(np.clip(pert_aide * 5, 0, 1))

        # Show overlap: perturbation in salient regions (red = bad for imperceptibility)
        axes[row, 4].imshow(overlap_pgd, cmap="hot", vmin=0, vmax=overlap_pgd.max() + 1e-8)
        axes[row, 5].imshow(overlap_aide, cmap="hot", vmin=0, vmax=overlap_pgd.max() + 1e-8)

    for col in range(n_cols):
        axes[0, col].set_title(col_titles[col], fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        "AIDE redirects perturbation AWAY from salient regions (224x224 ImageNette)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "perturbation_vs_saliency.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {RESULTS_DIR}/perturbation_vs_saliency.png")

    # Compute saliency overlap scores
    print("\n  Saliency overlap scores (lower = perturbation avoids salient regions):")
    pgd_overlaps, aide_overlaps = [], []
    for d in all_data:
        clean = np.transpose(d["clean"], (1, 2, 0))
        pert_pgd = np.abs(np.transpose(d["adv_pgd"], (1, 2, 0)) - clean).mean(2)
        pert_aide = np.abs(np.transpose(d["adv_aide"], (1, 2, 0)) - clean).mean(2)
        cam = d["cam_clean"]
        if cam.shape != pert_pgd.shape:
            cam = zoom(cam, (pert_pgd.shape[0] / cam.shape[0],
                             pert_pgd.shape[1] / cam.shape[1]))
        pgd_overlaps.append((pert_pgd * cam).sum() / (pert_pgd.sum() + 1e-8))
        aide_overlaps.append((pert_aide * cam).sum() / (pert_aide.sum() + 1e-8))
    print(f"    PGD saliency overlap:  {np.mean(pgd_overlaps):.4f}")
    print(f"    AIDE saliency overlap: {np.mean(aide_overlaps):.4f}")
    print(f"    Reduction: {(1 - np.mean(aide_overlaps)/np.mean(pgd_overlaps))*100:.1f}%")


# ============================================================
# Experiment 4: Quantitative saliency overlap
# ============================================================

def experiment_saliency_overlap(device, num_images=500):
    """Measure perturbation-saliency overlap across many images."""
    print("\n" + "=" * 70)
    print("ImageNet EXPERIMENT 4: Saliency Overlap (Quantitative)")
    print("=" * 70)

    model = models.resnet50(weights="DEFAULT").to(device).eval()
    target_layers = [model.layer4[-1]]

    data_root = str(ROOT / "data")
    ds, folder_to_inet = load_imagenette(data_root)
    clean_images, clean_labels, _ = get_correctly_classified(
        model, ds, folder_to_inet, device, num_images
    )
    print(f"  {len(clean_labels)} images")

    attacks = ["PGD-20", "AIDE-Base", "AIDE-MultiScale"]
    eps, alpha = 8 / 255, 2 / 255
    results = {}

    for atk_name in attacks:
        print(f"\n  {atk_name}:")
        overlaps = []
        batch_size = 8 if "AIDE" in atk_name else 16
        n_batches = (len(clean_labels) + batch_size - 1) // batch_size

        for bi in tqdm(range(n_batches), desc=f"    {atk_name}"):
            start = bi * batch_size
            end = min(start + batch_size, len(clean_labels))
            imgs = clean_images[start:end].to(device)
            lbls = clean_labels[start:end].to(device)

            # Get clean GradCAM
            cam_obj = get_cam("gradcam", model, target_layers[0])
            cam_maps = cam_obj.compute(imgs)  # (B, 1, H_cam, W_cam)
            cam_obj.remove_hooks()

            # Upsample CAM to image size
            cam_up = F.interpolate(cam_maps, size=(224, 224), mode="bilinear",
                                   align_corners=False)  # (B, 1, 224, 224)

            # Attack
            adv = run_attack(atk_name, model, imgs, lbls, eps, alpha, 20,
                             device, target_layers)
            if isinstance(adv, tuple):
                adv = adv[0]

            # Perturbation magnitude
            pert = (adv - imgs).abs().mean(dim=1, keepdim=True)  # (B, 1, 224, 224)

            # Saliency overlap: weighted average of saliency at perturbed pixels
            # = sum(pert * cam) / sum(pert)
            for b in range(imgs.shape[0]):
                p = pert[b, 0].cpu().numpy()
                c = cam_up[b, 0].detach().cpu().numpy()
                if p.sum() > 1e-8:
                    overlap = (p * c).sum() / p.sum()
                    overlaps.append(float(overlap))

        results[atk_name] = {
            "saliency_overlap_mean": float(np.mean(overlaps)),
            "saliency_overlap_std": float(np.std(overlaps)),
        }
        print(f"    Saliency overlap: {np.mean(overlaps):.4f} +/- {np.std(overlaps):.4f}")

    _save_json(results, RESULTS_DIR / "saliency_overlap.json")

    # Print comparison
    if "PGD-20" in results and "AIDE-Base" in results:
        pgd_o = results["PGD-20"]["saliency_overlap_mean"]
        aide_o = results["AIDE-Base"]["saliency_overlap_mean"]
        print(f"\n  AIDE vs PGD saliency overlap reduction: "
              f"{(1 - aide_o / pgd_o) * 100:.1f}%")

    return results


# ============================================================
# Utilities
# ============================================================

def _save_json(data, path):
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


def _print_table(results):
    print(f"\n{'=' * 90}")
    print("IMAGENET RESULTS TABLE")
    print(f"{'=' * 90}")
    groups = {}
    for key, val in results.items():
        parts = key.split("/")
        group_key = f"{parts[0]}/{parts[1]}/{parts[2]}"
        atk = parts[3]
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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AIDE ImageNet Experiments")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["main", "drift", "pertcomp", "overlap", "all"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")

    start = time.time()

    if args.experiment in ("main", "all"):
        experiment_main_results(device, num_images=args.num_images)

    if args.experiment in ("drift", "all"):
        experiment_attention_drift(device, num_images=5)

    if args.experiment in ("pertcomp", "all"):
        experiment_perturbation_comparison(device, num_images=5)

    if args.experiment in ("overlap", "all"):
        experiment_saliency_overlap(device, num_images=args.num_images)

    total = time.time() - start
    print(f"\nTotal runtime: {total / 60:.1f} minutes")


if __name__ == "__main__":
    main()
