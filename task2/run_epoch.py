from enum import Enum
import time
from typing import Optional

from tqdm.auto import tqdm
import torch
import torch.nn as nn

from transformer import PositionalEncoding
from dataset import SimpleDataset, BrainDataset, BigBrainDataset, UltraDuperBigBrainDataset


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


def get_gpt2_model(dataset: SimpleDataset, d_model: int = 1024, nhead: int = 8) -> torch.nn.Module:
    return nn.Sequential(
        nn.Embedding(num_embeddings=len(dataset.vocab), embedding_dim=d_model, padding_idx=0),
        PositionalEncoding(d_model, max_len=dataset.max_length),
        nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
    )


BATCH_SIZE = 4


def run_epoch(data_mode: DataMode, data_path: str = "data.hf", k: Optional[int] = None) -> None:
    if data_mode == DataMode.BRAIN:
        dataset = BrainDataset(data_path)
    elif data_mode == DataMode.BIG_BRAIN:
        dataset = BigBrainDataset(data_path)
    else:
        assert k is not None, f'k is None'
        dataset = UltraDuperBigBrainDataset(data_path, k=k)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loader = dataset.create_loader(batch_size=BATCH_SIZE)
    model = get_gpt2_model(dataset).to()
    times = []
    for batch in tqdm(loader):
        start = time.time()
        outputs = model(batch.to(device))
        if device != "cpu":
            torch.cuda.synchronize()
        times.append(time.time() - start)
