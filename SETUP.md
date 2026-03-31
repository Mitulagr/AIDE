# AIDE - Setup Guide

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 6000 Ada, 48GB VRAM)
- ~2GB disk space for datasets, ~1GB for model checkpoints

## 1. Activate virtual environment

```bash
source /home/mac/mitrix69/load/.venv/bin/activate
cd /path/to/AIDE
```

## 2. Install dependencies

Install PyTorch with the CUDA version matching your driver. Check your CUDA version with `nvidia-smi`.

```bash
# For CUDA 12.8 (adjust URL for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

## 3. Download datasets

CIFAR-10 downloads automatically on first run. For ImageNette (224x224 ImageNet subset):

```bash
python -c "
from torchvision.datasets.utils import download_and_extract_archive
download_and_extract_archive(
    'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz',
    'data/', filename='imagenette2-320.tgz'
)
"
```

## 4. Run experiments

CIFAR-10 models are trained automatically on first run (~2 min per model).
ImageNet models use pretrained weights (no training needed).

```bash
# Quick validation (AIDE-Base vs PGD, ~10 min)
CUDA_VISIBLE_DEVICES=0 python quick_validation.py

# Full CIFAR-10 experiment suite
CUDA_VISIBLE_DEVICES=0 python experiments/run_all.py --experiment all --num-images 1000

# Individual experiments
CUDA_VISIBLE_DEVICES=0 python experiments/run_all.py --experiment main      # Main results table
CUDA_VISIBLE_DEVICES=0 python experiments/run_all.py --experiment pareto    # ASR vs LPIPS Pareto
CUDA_VISIBLE_DEVICES=0 python experiments/run_all.py --experiment drift     # Attention drift figure
CUDA_VISIBLE_DEVICES=0 python experiments/run_all.py --experiment ablation  # Ablation study
CUDA_VISIBLE_DEVICES=0 python experiments/run_all.py --experiment defense   # Defense evasion
CUDA_VISIBLE_DEVICES=0 python experiments/run_all.py --experiment transfer  # Transferability
CUDA_VISIBLE_DEVICES=0 python experiments/run_all.py --experiment cam       # CAM method comparison
CUDA_VISIBLE_DEVICES=0 python experiments/run_all.py --experiment convergence

# ImageNet (ImageNette 224x224) experiments
CUDA_VISIBLE_DEVICES=0 python experiments/run_imagenet.py --num-images 500
```

## Project structure

```
AIDE/
├── src/
│   ├── attacks/          # FGSM, PGD, MI-FGSM, AIDE variants
│   ├── cam.py            # GradCAM, GradCAM++, LayerCAM, ScoreCAM
│   ├── config.py         # Attack and experiment configs
│   ├── data.py           # Dataset loading (CIFAR-10/100, ImageNet)
│   ├── metrics.py        # ASR, LPIPS, SSIM, PSNR, L2, attention drift
│   └── models.py         # Model loading, CIFAR adaptation, training
├── experiments/
│   ├── run_all.py        # Full CIFAR-10 experiment suite (8 experiments)
│   └── run_imagenet.py   # ImageNet (ImageNette) experiments
├── results/              # Output JSON, plots, model checkpoints
├── quick_validation.py   # Quick viability check
└── requirements.txt
```
