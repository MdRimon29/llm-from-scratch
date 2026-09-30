from pathlib import Path

from src.config import GPTConfig
from src.dataset import create_dataloader
from src.evaluation import calculate_perplexity, evaluate_loss
from src.model import GPTModel
from src.tokenizer import GPT2Tokenizer
from src.utils import get_device


ROOT_DIR = Path(__file__).resolve().parents[1]


def main():
    device = get_device()

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

    text = (ROOT_DIR/ "data"/ "raw"/ "the-verdict.txt").read_text(encoding="utf-8")

    split_idx = int(len(text) * 0.90)

    val_text = text[split_idx:]

    val_loader = create_dataloader(
        text=val_text,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=config.context_length,
        stride=config.context_length,
        shuffle=False,
        drop_last=False,
    )

    model = GPTModel(config)

    checkpoint_path = (ROOT_DIR/ "checkpoints"/ "gpt_model.pth")

    model.load_state_dict(
        __import__("torch").load(
            checkpoint_path,
            map_location=device,
        )
    )

    model.to(device)

    loss = evaluate_loss(model, val_loader, device)

    perplexity = calculate_perplexity(loss)

    print(f"Validation loss: {loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()