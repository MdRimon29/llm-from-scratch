import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))

            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(text: str, tokenizer, batch_size, max_length, stride, 
                      shuffle= True, drop_last= True, num_workers= 0) -> DataLoader:

    dataset = GPTDataset(
        text=text,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )