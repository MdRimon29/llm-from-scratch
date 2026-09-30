from pathlib import Path

import torch

from src.config import GPTConfig
from src.dataset import create_dataloader
from src.model import GPTModel
from src.tokenizer import GPT2Tokenizer
from src.training import train_model
from src.utils import get_device, set_seed


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "raw" / "the-verdict.txt"


def main():
    set_seed(123)

    device = get_device()

    print(f"Using device: {device}")

    config = GPTConfig(
        vocab_size=50257,
        context_length=256,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
    )

    tokenizer = GPT2Tokenizer()

    text = DATA_PATH.read_text(encoding="utf-8")

    split_idx = int(len(text) * 0.90)

    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_loader = create_dataloader(
        text=train_text,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=config.context_length,
        stride=config.context_length,
        shuffle=True,
        drop_last=True,
    )

    val_loader = create_dataloader(
        text=val_text,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=config.context_length,
        stride=config.context_length,
        shuffle=False,
        drop_last=False,
    )

    model = GPTModel(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        weight_decay=0.1,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        sample_every=2,
    )

    checkpoint_dir = ROOT_DIR / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        checkpoint_dir / "gpt_model.pth",
    )

    print("Training complete.")
    print("Model saved to checkpoints/gpt_model.pth")


if __name__ == "__main__":
    main()