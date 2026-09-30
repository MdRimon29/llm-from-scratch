import tiktoken
import torch


class GPT2Tokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def encode_tensor(self, text: str, device: torch.device | str = "cpu") -> torch.Tensor:
        token_ids = self.encode(text)

        return torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    def decode_tensor(self, token_ids: torch.Tensor) -> str:
        return self.decode(token_ids.squeeze(0).tolist())