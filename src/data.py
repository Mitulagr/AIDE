"""
Data loading utilities for AIDE adversarial attack experiments.

Datasets are stored under ``data_root`` (default: project ``data/`` dir).
All images are returned in [0, 1] float32 tensor space.
"""

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Class name lists (useful for human-readable logging / visualisation)
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",
]

# ---------------------------------------------------------------------------
# Default data root
# ---------------------------------------------------------------------------

_DEFAULT_DATA_ROOT = "/home/mac/mitrix69/research/AIDE/data"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_dataset(
    name: str,
    split: str = "test",
    transform: Optional[transforms.Compose] = None,
    data_root: str = _DEFAULT_DATA_ROOT,
) -> datasets.VisionDataset:
    """Load a vision dataset by name.

    Parameters
    ----------
    name : str
        One of ``'cifar10'``, ``'cifar100'``, ``'imagenet'``.
    split : str
        ``'train'`` or ``'test'`` (``'val'`` for ImageNet).
    transform : optional
        Custom torchvision transform.  When *None* a sensible default is used
        (ToTensor for CIFAR, Resize+CenterCrop+ToTensor for ImageNet).
    data_root : str
        Root directory where datasets are / will be downloaded.

    Returns
    -------
    torchvision.datasets.VisionDataset
    """
    name = name.lower().strip()
    os.makedirs(data_root, exist_ok=True)

    if name == "cifar10":
        default_transform = transforms.Compose([transforms.ToTensor()])
        return datasets.CIFAR10(
            root=data_root,
            train=(split == "train"),
            download=True,
            transform=transform or default_transform,
        )

    if name == "cifar100":
        default_transform = transforms.Compose([transforms.ToTensor()])
        return datasets.CIFAR100(
            root=data_root,
            train=(split == "train"),
            download=True,
            transform=transform or default_transform,
        )

    if name == "imagenet":
        default_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        imagenet_root = os.path.join(data_root, "imagenet")
        imagenet_split = "val" if split == "test" else split

        # Prefer the official ImageNet class when the folder structure matches.
        # Fall back to a plain ImageFolder otherwise.
        split_dir = os.path.join(imagenet_root, imagenet_split)
        if os.path.isdir(split_dir):
            try:
                return datasets.ImageNet(
                    root=imagenet_root,
                    split=imagenet_split,
                    transform=transform or default_transform,
                )
            except (RuntimeError, FileNotFoundError):
                return datasets.ImageFolder(
                    root=split_dir,
                    transform=transform or default_transform,
                )
        raise FileNotFoundError(
            f"ImageNet split directory not found: {split_dir}. "
            "Please download ImageNet manually and place it under "
            f"{imagenet_root}/{{train,val}}/."
        )

    raise ValueError(f"Unknown dataset: '{name}'. Choose from: cifar10, cifar100, imagenet.")


def get_dataloader(
    dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """Convenience wrapper around ``torch.utils.data.DataLoader``."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def get_correctly_classified_subset(
    model: torch.nn.Module,
    dataset,
    device: str,
    num_images: int = 1000,
) -> TensorDataset:
    """Return a ``TensorDataset`` of ``(images, labels)`` the model classifies
    correctly, capped at *num_images*.

    The model is set to eval mode internally.  Images and labels are returned
    on CPU so downstream code can move them as needed.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    correct_images: list[torch.Tensor] = []
    correct_labels: list[torch.Tensor] = []
    collected = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        mask = preds == labels

        if mask.any():
            correct_images.append(images[mask].cpu())
            correct_labels.append(labels[mask].cpu())
            collected += mask.sum().item()

        if collected >= num_images:
            break

    all_images = torch.cat(correct_images, dim=0)[:num_images]
    all_labels = torch.cat(correct_labels, dim=0)[:num_images]
    return TensorDataset(all_images, all_labels)
