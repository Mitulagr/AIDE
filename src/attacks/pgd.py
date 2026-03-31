import torch
import torch.nn.functional as F


def pgd_attack(model, images, labels, epsilon, alpha, num_steps, device,
               record_loss=False):
    """PGD-Linf attack (Madry et al., 2018).

    Args:
        model: classifier in eval mode
        images: (B, C, H, W) in [0,1]
        labels: (B,) true labels
        epsilon: Linf budget
        alpha: step size per iteration
        num_steps: number of PGD steps
        device: torch device
        record_loss: if True, also return list of loss values per step
    Returns:
        adv_images: (B, C, H, W) if record_loss=False
        (adv_images, losses): if record_loss=True
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # Random start within the epsilon-ball
    adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

    losses = []

    for _ in range(num_steps):
        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        if record_loss:
            losses.append(loss.item())

        loss.backward()

        # Sign gradient ascent step
        adv_images = adv_images + alpha * adv_images.grad.sign()

        # Project back into the epsilon-ball around the original images
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + perturbation, 0.0, 1.0).detach()

    if record_loss:
        return adv_images.detach(), losses
    return adv_images.detach()
