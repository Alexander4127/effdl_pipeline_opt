from enum import Enum
import json
import numpy as np
import pandas as pd
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


def run_epoch(
        data_mode: DataMode,
        data_path: str = "data.hf",
        k: Optional[int] = None,
        warmup_steps: int = 150) -> list:
    start = time.time()
    if data_mode == DataMode.BRAIN:
        dataset = BrainDataset(data_path)
    elif data_mode == DataMode.BIG_BRAIN:
        dataset = BigBrainDataset(data_path)
    else:
        assert k is not None, f'k is None'
        dataset = UltraDuperBigBrainDataset(data_path, k=k)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loader = dataset.create_loader(batch_size=BATCH_SIZE)
    init_time = time.time() - start
    model = get_gpt2_model(dataset).to()
    times = []
    for batch in tqdm(loader, desc=f"{DataMode}-{k}"):
        start = time.time()
        outputs = model(batch.to(device))
        if device != "cpu":
            torch.cuda.synchronize()
        times.append(time.time() - start)
    times = times[warmup_steps:]
    return [init_time, np.minimum(times), np.maximum(times), np.mean(times), np.median(times)]


def add_result(foo, desc, df, **kwargs):
    res = foo(**kwargs)
    print(f'{desc}:', res)
    df.loc[desc] = res


if __name__ == "__main__":
    df = pd.DataFrame(columns=["Init", "Min", "Max", "Mean", "Med"])
    add_result(run_epoch, "Brain", df, data_mode=DataMode.BRAIN)
    add_result(run_epoch, "Big Brain", df, data_mode=DataMode.BRAIN)
    for k in [1, 5, 10, 20, 50]:
        add_result(run_epoch, f"Ultra Big Brain. k = {k}", df, data_mode=DataMode.ULTRA_DUPER_BIG_BRAIN, k=k)
