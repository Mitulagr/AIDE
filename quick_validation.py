"""
AIDE Quick Validation (Section 10 of Research Plan)
=====================================================
AIDE-Base vs Vanilla PGD on CIFAR-10 + ResNet-50 at eps=8/255
Metrics: ASR, LPIPS, SSIM + Attention Drift Visualization
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import lpips
from pytorch_msssim import ssim as compute_ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time

# ============================================================
# Config
# ============================================================
DEVICE = torch.device("cuda:0")  # Will be GPU 2 via CUDA_VISIBLE_DEVICES=2
EPSILON = 8.0 / 255.0
ALPHA = 2.0 / 255.0  # Standard step size for PGD-20
NUM_STEPS = 20
NUM_EVAL_IMAGES = 1000  # Evaluate on 1000 correctly-classified images
BATCH_SIZE = 64
NUM_VIS_IMAGES = 5  # For attention drift visualization
RESULTS_DIR = "/home/mac/mitrix69/research/AIDE/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Epsilon: {EPSILON:.4f} ({8}/255)")
print(f"Alpha: {ALPHA:.4f} ({2}/255)")
print(f"Steps: {NUM_STEPS}")
print(f"Eval images: {NUM_EVAL_IMAGES}")

# ============================================================
# Data: CIFAR-10 (no normalization - we work in [0,1] space)
# ============================================================
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(
    root='/home/mac/mitrix69/research/AIDE/data', train=False,
    download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False, num_workers=2
)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# ============================================================
# Model: ResNet-50 fine-tuned on CIFAR-10
# ============================================================
def get_resnet50_cifar10():
    """Load pretrained ResNet-50, adapt for CIFAR-10 (32x32, 10 classes)."""
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    # Adapt first conv for 32x32 input (smaller kernel, no aggressive downsampling)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def train_resnet50_cifar10(model, device, epochs=10):
    """Fine-tune ResNet-50 on CIFAR-10."""
    print("Fine-tuning ResNet-50 on CIFAR-10...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='/home/mac/mitrix69/research/AIDE/data', train=True,
        download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4
    )

    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        scheduler.step()
        acc = 100.0 * correct / total
        print(f"  Epoch {epoch+1}: Loss={running_loss/len(trainloader):.4f}, Acc={acc:.2f}%")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='/home/mac/mitrix69/research/AIDE/data', train=False,
                download=True, transform=transforms.ToTensor()
            ), batch_size=256, shuffle=False, num_workers=4
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100.0 * correct / total
    print(f"  Test accuracy: {test_acc:.2f}%")
    return model, test_acc


# ============================================================
# GradCAM for ResNet-50 (differentiable, works with CIFAR-10)
# ============================================================
class GradCAM:
    """GradCAM that hooks into a target layer and computes saliency maps.

    Returns a normalized [0,1] heatmap of shape (H, W) for each image.
    Designed to work within the PGD loop without breaking autograd for the attack gradient.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        self._hooks = []

        # Register hooks on target layer
        self._hooks.append(
            target_layer.register_forward_hook(self._save_features)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradients)
        )

    def _save_features(self, module, input, output):
        self.feature_maps = output

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x, class_idx=None):
        """Compute GradCAM heatmap. Returns tensor of shape (B, 1, H, W) in [0,1]."""
        B, C, H, W = x.shape

        # Forward
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        # Backward for the target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        for i in range(B):
            one_hot[i, class_idx[i]] = 1.0
        output.backward(gradient=one_hot, retain_graph=False)

        # Compute CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)  # (B, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)

        # Normalize per-image to [0, 1]
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam  # (B, 1, H, W)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ============================================================
# Attack Implementations
# ============================================================
def pgd_attack(model, images, labels, epsilon, alpha, num_steps, device):
    """Standard PGD-Linf attack (Madry et al.)."""
    images = images.to(device)
    labels = labels.to(device)

    # Random start
    adv = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, 0.0, 1.0)

    for step in range(num_steps):
        adv_input = adv.clone().detach().requires_grad_(True)
        outputs = model(adv_input)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        grad = adv_input.grad.detach()
        adv = adv + alpha * grad.sign()

        # Project onto epsilon-ball
        delta = torch.clamp(adv - images, -epsilon, epsilon)
        adv = torch.clamp(images + delta, 0.0, 1.0).detach()

    return adv


def aide_base_attack(model, images, labels, epsilon, alpha, num_steps, device,
                     target_layer, record_drift=False):
    """AIDE-Base: PGD with per-step inverse-saliency masking via GradCAM.

    At each step:
    1. Compute GradCAM on current adversarial image
    2. Invert: W = 1 - S (high weight where model doesn't attend)
    3. Mask gradient with W before taking PGD step
    """
    images = images.to(device)
    labels = labels.to(device)

    # Random start
    adv = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv, 0.0, 1.0)

    drift_maps = [] if record_drift else None

    for step in range(num_steps):
        # --- Step 1: Compute GradCAM on current adversarial image ---
        # We need a separate forward/backward for CAM (detached from attack grad)
        adv_for_cam = adv.clone().detach().requires_grad_(True)
        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam(adv_for_cam)  # (B, 1, H, W) in [0,1]
        grad_cam.remove_hooks()
        cam = cam.detach()

        if record_drift:
            drift_maps.append(cam.cpu().numpy())

        # --- Step 2: Inverse saliency mask ---
        inverse_mask = 1.0 - cam  # (B, 1, H, W) - high where model ignores

        # --- Step 3: Compute attack gradient ---
        adv_input = adv.clone().detach().requires_grad_(True)
        outputs = model(adv_input)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        grad = adv_input.grad.detach()

        # --- Step 4: Apply mask AFTER sign (critical: sign destroys mask magnitude) ---
        # sign(g * W) = sign(g) when W > 0 always, so mask must be post-sign
        adv = adv + alpha * grad.sign() * inverse_mask

        # --- Step 6: Project onto epsilon-ball ---
        delta = torch.clamp(adv - images, -epsilon, epsilon)
        adv = torch.clamp(images + delta, 0.0, 1.0).detach()

    if record_drift:
        return adv, drift_maps
    return adv


# ============================================================
# Evaluation
# ============================================================
def evaluate_attack(model, attack_fn, testloader, device, num_images, lpips_fn,
                    attack_name="Attack"):
    """Run attack on num_images correctly-classified samples.

    Returns dict with ASR, mean LPIPS, mean SSIM, and per-image stats.
    """
    model.eval()

    total_correct_clean = 0
    total_attacked = 0
    total_fooled = 0
    all_lpips = []
    all_ssim = []

    pbar = tqdm(total=num_images, desc=f"Evaluating {attack_name}")

    for images, labels in testloader:
        if total_attacked >= num_images:
            break

        images, labels = images.to(device), labels.to(device)

        # Check if model classifies correctly
        with torch.no_grad():
            clean_pred = model(images).argmax(dim=1)

        if clean_pred.item() != labels.item():
            continue  # Skip misclassified images

        # Attack
        adv_images = attack_fn(images, labels)

        # Check if attack succeeded
        with torch.no_grad():
            adv_pred = model(adv_images).argmax(dim=1)

        fooled = (adv_pred != labels).item()
        total_fooled += int(fooled)
        total_attacked += 1

        # Compute LPIPS (expects images in [-1, 1])
        img_lpips = images * 2.0 - 1.0
        adv_lpips = adv_images * 2.0 - 1.0
        with torch.no_grad():
            lp = lpips_fn(img_lpips, adv_lpips).item()
        all_lpips.append(lp)

        # Compute SSIM
        with torch.no_grad():
            ss = compute_ssim(images, adv_images, data_range=1.0, size_average=True).item()
        all_ssim.append(ss)

        pbar.update(1)

    pbar.close()

    asr = total_fooled / total_attacked * 100.0
    mean_lpips = np.mean(all_lpips)
    std_lpips = np.std(all_lpips)
    mean_ssim = np.mean(all_ssim)
    std_ssim = np.std(all_ssim)

    return {
        'asr': asr,
        'lpips_mean': mean_lpips,
        'lpips_std': std_lpips,
        'ssim_mean': mean_ssim,
        'ssim_std': std_ssim,
        'num_images': total_attacked,
        'num_fooled': total_fooled,
        'all_lpips': all_lpips,
        'all_ssim': all_ssim,
    }


# ============================================================
# Attention Drift Visualization
# ============================================================
def visualize_attention_drift(model, testloader, device, target_layer, num_images=5):
    """Generate the 'hero figure' showing attention drift over PGD steps."""
    model.eval()

    vis_steps = [0, 4, 9, 14, 19]  # Steps to visualize (0-indexed)
    collected = 0

    all_clean = []
    all_adv = []
    all_drift = []
    all_labels = []

    for images, labels in testloader:
        if collected >= num_images:
            break
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            if model(images).argmax(dim=1).item() != labels.item():
                continue

        adv, drift_maps = aide_base_attack(
            model, images, labels, EPSILON, ALPHA, NUM_STEPS, device,
            target_layer, record_drift=True
        )

        all_clean.append(images.cpu().numpy()[0])
        all_adv.append(adv.cpu().numpy()[0])
        all_drift.append(drift_maps)  # list of (1,1,H,W) arrays per step
        all_labels.append(labels.item())
        collected += 1
        print(f"  Collected drift visualization for image {collected}/{num_images}")

    # Plot: rows = images, cols = [clean, step0_cam, step4_cam, ..., step19_cam, adv, perturbation]
    n_vis_steps = len(vis_steps)
    n_cols = 2 + n_vis_steps + 1  # clean + cams + adv + perturbation
    fig, axes = plt.subplots(num_images, n_cols, figsize=(n_cols * 2.5, num_images * 2.5))

    if num_images == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Clean'] + [f'CAM Step {s+1}' for s in vis_steps] + ['Adversarial', 'Perturbation (10x)']

    for row in range(num_images):
        clean_img = np.transpose(all_clean[row], (1, 2, 0))
        adv_img = np.transpose(all_adv[row], (1, 2, 0))
        pert = np.clip(np.abs(adv_img - clean_img) * 10, 0, 1)

        # Clean image
        axes[row, 0].imshow(clean_img)
        axes[row, 0].set_ylabel(CIFAR10_CLASSES[all_labels[row]], fontsize=10)

        # CAM at selected steps
        for ci, step in enumerate(vis_steps):
            cam = all_drift[row][step][0, 0]  # (H, W)
            axes[row, 1 + ci].imshow(clean_img, alpha=0.4)
            axes[row, 1 + ci].imshow(cam, cmap='jet', alpha=0.6, vmin=0, vmax=1)

        # Adversarial image
        axes[row, -2].imshow(np.clip(adv_img, 0, 1))

        # Perturbation (magnified)
        axes[row, -1].imshow(pert)

    for col in range(n_cols):
        axes[0, col].set_title(col_titles[col], fontsize=9)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('AIDE Attention Drift: GradCAM shifts across PGD steps\n'
                 '(Model attention drifts toward perturbed background regions)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'attention_drift.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved attention drift visualization to {RESULTS_DIR}/attention_drift.png")


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()

    # --- Load or train model ---
    model_path = os.path.join(RESULTS_DIR, 'resnet50_cifar10.pth')
    model = get_resnet50_cifar10()

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)
        model.eval()
        # Quick accuracy check
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in torch.utils.data.DataLoader(
                testset, batch_size=256, shuffle=False, num_workers=4
            ):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        print(f"Model test accuracy: {100.0*correct/total:.2f}%")
    else:
        model, test_acc = train_resnet50_cifar10(model, DEVICE, epochs=15)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

    # --- Target layer for GradCAM ---
    # For 32x32 CIFAR-10: layer3 gives 8x8 spatial maps (layer4 gives 4x4 = too coarse)
    target_layer = model.layer3[-1]

    # --- LPIPS metric ---
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)

    # --- Define attack functions ---
    def pgd_fn(images, labels):
        return pgd_attack(model, images, labels, EPSILON, ALPHA, NUM_STEPS, DEVICE)

    def aide_fn(images, labels):
        return aide_base_attack(model, images, labels, EPSILON, ALPHA, NUM_STEPS, DEVICE,
                                target_layer)

    # --- Evaluate PGD ---
    print("\n" + "=" * 60)
    print("Evaluating Vanilla PGD")
    print("=" * 60)
    pgd_results = evaluate_attack(
        model, pgd_fn, testloader, DEVICE, NUM_EVAL_IMAGES, lpips_fn,
        attack_name="PGD-20"
    )

    # --- Evaluate AIDE ---
    print("\n" + "=" * 60)
    print("Evaluating AIDE-Base")
    print("=" * 60)
    aide_results = evaluate_attack(
        model, aide_fn, testloader, DEVICE, NUM_EVAL_IMAGES, lpips_fn,
        attack_name="AIDE-Base"
    )

    # --- Print results ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'PGD-20':>15} {'AIDE-Base':>15}")
    print("-" * 50)
    print(f"{'ASR (%)' :<20} {pgd_results['asr']:>14.2f}% {aide_results['asr']:>14.2f}%")
    print(f"{'LPIPS (↓ better)':<20} {pgd_results['lpips_mean']:>12.4f}±{pgd_results['lpips_std']:.4f}  "
          f"{aide_results['lpips_mean']:>8.4f}±{aide_results['lpips_std']:.4f}")
    print(f"{'SSIM (↑ better)':<20} {pgd_results['ssim_mean']:>12.4f}±{pgd_results['ssim_std']:.4f}  "
          f"{aide_results['ssim_mean']:>8.4f}±{aide_results['ssim_std']:.4f}")

    # --- Viability check ---
    print("\n" + "=" * 60)
    print("VIABILITY CHECK")
    print("=" * 60)
    aide_asr_ratio = aide_results['asr'] / max(pgd_results['asr'], 1e-8) * 100
    lpips_improvement = (pgd_results['lpips_mean'] - aide_results['lpips_mean']) / pgd_results['lpips_mean'] * 100
    ssim_improvement = (aide_results['ssim_mean'] - pgd_results['ssim_mean']) / pgd_results['ssim_mean'] * 100

    print(f"AIDE ASR / PGD ASR: {aide_asr_ratio:.1f}% (need ≥90%)")
    print(f"LPIPS improvement: {lpips_improvement:+.1f}% (need negative = lower LPIPS)")
    print(f"SSIM improvement:  {ssim_improvement:+.1f}% (need positive = higher SSIM)")

    viable = aide_asr_ratio >= 90.0 and aide_results['lpips_mean'] < pgd_results['lpips_mean']
    print(f"\n>>> PAPER VIABLE: {'YES ✓' if viable else 'NEEDS RETHINKING ✗'} <<<")

    if not viable:
        if aide_asr_ratio < 90.0:
            print(f"  - ASR too low: AIDE={aide_results['asr']:.1f}% vs PGD={pgd_results['asr']:.1f}%")
        if aide_results['lpips_mean'] >= pgd_results['lpips_mean']:
            print(f"  - LPIPS not better: AIDE={aide_results['lpips_mean']:.4f} vs PGD={pgd_results['lpips_mean']:.4f}")

    # --- Save results ---
    results = {
        'config': {
            'epsilon': 8, 'alpha': 2, 'steps': NUM_STEPS,
            'num_images': NUM_EVAL_IMAGES, 'model': 'ResNet-50'
        },
        'pgd': {k: v for k, v in pgd_results.items() if k not in ('all_lpips', 'all_ssim')},
        'aide': {k: v for k, v in aide_results.items() if k not in ('all_lpips', 'all_ssim')},
        'viability': {
            'aide_asr_ratio': aide_asr_ratio,
            'lpips_improvement_pct': lpips_improvement,
            'ssim_improvement_pct': ssim_improvement,
            'viable': bool(viable),
        },
        'runtime_seconds': time.time() - start_time,
    }

    results_path = os.path.join(RESULTS_DIR, 'quick_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else bool(o) if isinstance(o, np.bool_) else o)
    print(f"\nResults saved to {results_path}")

    # --- Attention drift visualization ---
    print("\n" + "=" * 60)
    print("Generating Attention Drift Visualization")
    print("=" * 60)
    visualize_attention_drift(model, testloader, DEVICE, target_layer, NUM_VIS_IMAGES)

    # --- Distribution comparison plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(pgd_results['all_lpips'], bins=30, alpha=0.7, label='PGD', color='#e74c3c')
    ax1.hist(aide_results['all_lpips'], bins=30, alpha=0.7, label='AIDE', color='#3498db')
    ax1.set_xlabel('LPIPS (lower = more imperceptible)')
    ax1.set_ylabel('Count')
    ax1.set_title('LPIPS Distribution')
    ax1.legend()

    ax2.hist(pgd_results['all_ssim'], bins=30, alpha=0.7, label='PGD', color='#e74c3c')
    ax2.hist(aide_results['all_ssim'], bins=30, alpha=0.7, label='AIDE', color='#3498db')
    ax2.set_xlabel('SSIM (higher = more similar)')
    ax2.set_ylabel('Count')
    ax2.set_title('SSIM Distribution')
    ax2.legend()

    plt.suptitle(f'PGD vs AIDE Perceptual Quality (ε={8}/255, {NUM_EVAL_IMAGES} images)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'distribution_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to {RESULTS_DIR}/distribution_comparison.png")

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")


if __name__ == '__main__':
    main()
