from pathlib import Path

import torch

from src.config import GPTConfig
from src.generation import generate_text
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

    model = GPTModel(config)

    checkpoint_path = (ROOT_DIR/ "checkpoints"/ "gpt_model.pth")

    state_dict = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    prompt = "Every effort moves"

    input_ids = tokenizer.encode_tensor(prompt, device=device)

    output_ids = generate_text(model=model, idx=input_ids, max_new_tokens=50, context_size=config.context_length)

    output_text = tokenizer.decode_tensor(output_ids)

    print(output_text)


if __name__ == "__main__":
    main()