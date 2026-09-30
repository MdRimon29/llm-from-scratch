import torch
import torch.nn as nn

from .generation import generate_text


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    start_context,
    tokenizer,
    eval_freq: int = 100,
    eval_iter: int = 5,
    sample_every: int = 2,
):

    train_losses = []
    val_losses = []
    track_tokens_seen = []

    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):

        model.train()

        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)

            loss.backward()

            optimizer.step()

            tokens_seen += (input_batch.numel())

            global_step += 1

            if global_step % eval_freq == 0:

                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)

                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)

                train_losses.append(train_loss)

                val_losses.append(val_loss)

                track_tokens_seen.append(tokens_seen)

                print(
                    f"Epoch {epoch + 1}, "
                    f"Step {global_step}: "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        if (epoch + 1) % sample_every == 0:
            print(
                f"\n--- Sample after epoch {epoch + 1} ---"
            )

            generate_and_print_sample(
                model,
                tokenizer,
                device,
                start_context,
            )


    return (train_losses, val_losses, track_tokens_seen)


def generate_and_print_sample(
    model,
    tokenizer,
    device,
    start_context,
):
    model.eval()

    context_size = model.cfg.context_length

    encoded = tokenizer.encode_tensor(
        start_context,
        device=device,
    )

    with torch.no_grad():
        token_ids = generate_text(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )

    decoded_text = tokenizer.decode_tensor(token_ids)

    print(decoded_text.replace("\n", " "))

    model.train()