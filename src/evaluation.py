import math

import torch

from .training import calc_loss_loader


@torch.no_grad()
def evaluate_loss(model, data_loader, device= torch.device, num_batches= None) -> float:

    loss = calc_loss_loader(
        data_loader=data_loader,
        model=model,
        device=device,
        num_batches=num_batches,
    )

    return loss


def calculate_perplexity(loss: float) -> float:
    return math.exp(loss)