import torch
import torch.nn.functional as F


def mifgsm_attack(model, images, labels, epsilon, alpha, num_steps, device,
                  decay_factor=1.0, record_loss=False):
    """MI-FGSM attack (Dong et al., 2018) - Momentum Iterative FGSM.

    Args:
        model: classifier
        images: (B, C, H, W) in [0,1]
        labels: (B,) true labels
        epsilon, alpha, num_steps: standard PGD params
        decay_factor: momentum decay (default 1.0)
        device: torch device
        record_loss: if True, also return list of loss values
    Returns:
        adv_images or (adv_images, losses)
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # Random start within the epsilon-ball
    adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

    momentum = torch.zeros_like(images)
    losses = []

    for _ in range(num_steps):
        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        if record_loss:
            losses.append(loss.item())

        loss.backward()

        grad = adv_images.grad.detach()

        # Accumulate momentum on L1-normalized gradient
        grad_norm = grad / (grad.abs().sum(dim=(1, 2, 3), keepdim=True) + 1e-12)
        momentum = decay_factor * momentum + grad_norm

        # Step using the sign of the accumulated momentum
        adv_images = adv_images + alpha * momentum.sign()

        # Project back into the epsilon-ball around the original images
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + perturbation, 0.0, 1.0).detach()

    if record_loss:
        return adv_images.detach(), losses
    return adv_images.detach()
