import torch
import torch.nn.functional as F


def fgsm_attack(model, images, labels, epsilon, device):
    """Single-step FGSM attack (Goodfellow et al., 2014).

    Args:
        model: classifier in eval mode
        images: (B, C, H, W) in [0,1]
        labels: (B,) true labels
        epsilon: Linf budget
        device: torch device
    Returns:
        adv_images: (B, C, H, W) in [0,1], detached
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    loss.backward()

    # Single step in the sign of the gradient
    adv_images = images + epsilon * images.grad.sign()
    adv_images = torch.clamp(adv_images, 0.0, 1.0)

    return adv_images.detach()
