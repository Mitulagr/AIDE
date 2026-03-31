"""
Model loading, CIFAR adaptation, and training utilities for AIDE experiments.

Supports: resnet50, vgg19, densenet121, mobilenetv2, swin_t, convnext_t
Datasets: cifar10 (10 classes), cifar100 (100 classes), imagenet (1000 classes)
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

_NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet": 1000,
}

_CIFAR_DATASETS = {"cifar10", "cifar100"}

# Map user-facing names to torchvision factory functions
_MODEL_FACTORIES = {
    "resnet50": models.resnet50,
    "vgg19": models.vgg19,
    "densenet121": models.densenet121,
    "mobilenetv2": models.mobilenet_v2,
    "swin_t": models.swin_t,
    "convnext_t": models.convnext_tiny,
}


# ---------------------------------------------------------------------------
# CIFAR adaptation helpers
# ---------------------------------------------------------------------------


def _adapt_resnet_for_cifar(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace conv1 (7x7, stride 2) with 3x3 stride 1, remove maxpool,
    and swap the fc head for the correct number of classes."""
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    if model.fc.out_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _adapt_vgg_for_cifar(model: nn.Module, num_classes: int) -> nn.Module:
    """Shrink the classifier head; keep the feature extractor as-is (the
    adaptive avgpool already handles arbitrary spatial sizes)."""
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, num_classes),
    )
    return model


def _adapt_densenet_for_cifar(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the initial 7x7 conv with 3x3 and remove the initial maxpool.
    Swap classifier head."""
    model.features.conv0 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.features.pool0 = nn.Identity()
    if model.classifier.out_features != num_classes:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def _adapt_mobilenetv2_for_cifar(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace initial conv (stride 2) with stride 1, swap classifier."""
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        3, first_conv.out_channels,
        kernel_size=3, stride=1, padding=1, bias=False,
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _adapt_swin_for_cifar(model: nn.Module, num_classes: int) -> nn.Module:
    """Swap the classification head for the right number of classes.
    Swin-T's patch embedding already adapts to different input sizes."""
    if model.head.out_features != num_classes:
        model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def _adapt_convnext_for_cifar(model: nn.Module, num_classes: int) -> nn.Module:
    """Swap classification head."""
    head = model.classifier[-1]
    if head.out_features != num_classes:
        model.classifier[-1] = nn.Linear(head.in_features, num_classes)
    return model


_CIFAR_ADAPTORS = {
    "resnet50": _adapt_resnet_for_cifar,
    "vgg19": _adapt_vgg_for_cifar,
    "densenet121": _adapt_densenet_for_cifar,
    "mobilenetv2": _adapt_mobilenetv2_for_cifar,
    "swin_t": _adapt_swin_for_cifar,
    "convnext_t": _adapt_convnext_for_cifar,
}


# ---------------------------------------------------------------------------
# Public: get_model
# ---------------------------------------------------------------------------


def get_model(
    name: str,
    dataset: str = "cifar10",
    device: str = "cuda:0",
    pretrained: bool = True,
) -> nn.Module:
    """Load a model and return it in eval mode on *device*.

    For CIFAR datasets the architecture is adapted for 32x32 inputs and the
    classifier head is resized to the correct number of classes.  For ImageNet
    the standard pretrained weights are used as-is.

    Parameters
    ----------
    name : str
        Model name (resnet50, vgg19, densenet121, mobilenetv2, swin_t, convnext_t).
    dataset : str
        Target dataset (cifar10, cifar100, imagenet).
    device : str
        Torch device string.
    pretrained : bool
        Whether to load pretrained (ImageNet) weights before adaptation.

    Returns
    -------
    nn.Module  (in eval mode, on *device*)
    """
    name = name.lower().strip()
    dataset = dataset.lower().strip()

    if name not in _MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(_MODEL_FACTORIES.keys())}"
        )
    if dataset not in _NUM_CLASSES:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from: {list(_NUM_CLASSES.keys())}"
        )

    num_classes = _NUM_CLASSES[dataset]
    factory = _MODEL_FACTORIES[name]

    # Load with or without pretrained ImageNet weights
    weights = "DEFAULT" if pretrained else None
    model = factory(weights=weights)

    # Adapt architecture for CIFAR (32x32) inputs
    if dataset in _CIFAR_DATASETS:
        adaptor = _CIFAR_ADAPTORS[name]
        model = adaptor(model, num_classes)

    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Public: get_target_layers (for GradCAM and variants)
# ---------------------------------------------------------------------------


def get_target_layers(
    model: nn.Module,
    model_name: str,
    layer_spec: str = "auto",
) -> list[nn.Module]:
    """Return a list of target layers suitable for CAM methods.

    Parameters
    ----------
    model : nn.Module
        The model instance.
    model_name : str
        Architecture name (must match the names used by ``get_model``).
    layer_spec : str
        ``'auto'`` selects the canonical last convolutional / block layer for
        each architecture.  Otherwise an attribute path like
        ``'layer4.-1'`` may be provided (dot-separated, negative indices OK).

    Returns
    -------
    list[nn.Module]
    """
    model_name = model_name.lower().strip()

    if layer_spec != "auto":
        return [_resolve_layer(model, layer_spec)]

    # Detect whether the model has been adapted for CIFAR (32x32).
    is_cifar = _is_cifar_adapted(model, model_name)

    if model_name == "resnet50":
        # For CIFAR-adapted models layer4 produces 1x1 maps which are
        # useless for spatial CAM, so use layer3 instead.
        if is_cifar:
            return [model.layer3[-1]]
        return [model.layer4[-1]]

    if model_name == "vgg19":
        # features[34] is the last Conv2d (512-ch) before the final maxpool.
        return [model.features[34]]

    if model_name == "densenet121":
        return [model.features.denseblock4]

    if model_name == "mobilenetv2":
        return [model.features[-1]]

    if model_name == "swin_t":
        return [model.features[-1][-1].norm1]

    if model_name == "convnext_t":
        return [model.features[-1][-1]]

    raise ValueError(f"No auto target-layer mapping for model '{model_name}'.")


def _resolve_layer(model: nn.Module, path: str) -> nn.Module:
    """Walk a dot-separated attribute path (supports negative int indices)."""
    obj = model
    for part in path.split("."):
        try:
            idx = int(part)
            obj = list(obj.children())[idx]
        except ValueError:
            obj = getattr(obj, part)
    return obj


def _is_cifar_adapted(model: nn.Module, model_name: str) -> bool:
    """Heuristic: check if the model was adapted for 32x32 CIFAR inputs."""
    if model_name == "resnet50":
        return model.conv1.kernel_size == (3, 3) and isinstance(model.maxpool, nn.Identity)
    if model_name == "densenet121":
        return isinstance(model.features.pool0, nn.Identity)
    if model_name == "mobilenetv2":
        return model.features[0][0].stride == (1, 1)
    # For VGG / Swin / ConvNeXt the spatial adaptation doesn't change
    # the optimal CAM layer so we default to False (ImageNet path).
    return False


# ---------------------------------------------------------------------------
# Public: training utilities
# ---------------------------------------------------------------------------


def train_cifar_model(
    model: nn.Module,
    dataset_name: str,
    device: str,
    epochs: int = 15,
    save_path: Optional[str] = None,
) -> tuple[nn.Module, float]:
    """Fine-tune *model* on a CIFAR dataset.

    Uses SGD with cosine annealing, standard data augmentation (random crop
    with padding=4, random horizontal flip), and returns (model, test_accuracy).

    Parameters
    ----------
    model : nn.Module
        Model already on *device* (may be in eval mode; will be set to train).
    dataset_name : str
        ``'cifar10'`` or ``'cifar100'``.
    device : str
        Torch device string.
    epochs : int
        Number of training epochs.
    save_path : str, optional
        If provided, the model state dict is saved here after training.

    Returns
    -------
    tuple[nn.Module, float]
        The trained model (in eval mode) and the final test accuracy.
    """
    dataset_name = dataset_name.lower().strip()
    data_root = "/home/mac/mitulagr/research/AIDE/data"
    os.makedirs(data_root, exist_ok=True)

    dataset_cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100

    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_set = dataset_cls(root=data_root, train=True, download=True, transform=train_transform)
    test_set = dataset_cls(root=data_root, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"loss {avg_loss:.4f} | "
            f"train acc {train_acc:.2f}% | "
            f"lr {scheduler.get_last_lr()[0]:.6f}"
        )

    # Evaluate on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = 100.0 * correct / total
    print(f"  Test accuracy: {test_acc:.2f}%")

    # Save if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  Model saved to {save_path}")

    return model, test_acc


def load_or_train_model(
    name: str,
    dataset: str,
    device: str,
    models_dir: str = "/home/mac/mitulagr/research/AIDE/results/models",
) -> nn.Module:
    """Load a saved CIFAR-adapted model if it exists; otherwise train and save.

    For ImageNet, the pretrained model is returned directly (no fine-tuning).

    Parameters
    ----------
    name : str
        Model architecture name.
    dataset : str
        Dataset name.
    device : str
        Torch device string.
    models_dir : str
        Directory to search for / save trained checkpoints.

    Returns
    -------
    nn.Module  (in eval mode on *device*)
    """
    dataset = dataset.lower().strip()

    if dataset == "imagenet":
        return get_model(name, dataset=dataset, device=device, pretrained=True)

    # Build path for saved checkpoint
    os.makedirs(models_dir, exist_ok=True)
    filename = f"{name}_{dataset}.pth"
    save_path = os.path.join(models_dir, filename)

    # Also check legacy location (results/ root)
    legacy_path = os.path.join(
        "/home/mac/mitulagr/research/AIDE/results", filename
    )

    model = get_model(name, dataset=dataset, device=device, pretrained=True)

    # Try to load existing checkpoint
    for path in (save_path, legacy_path):
        if os.path.isfile(path):
            print(f"Loading saved model from {path}")
            state_dict = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            return model

    # No checkpoint found -- train from scratch
    print(f"No saved checkpoint found for {name}/{dataset}. Training...")
    model, test_acc = train_cifar_model(
        model, dataset, device, epochs=15, save_path=save_path
    )
    return model
