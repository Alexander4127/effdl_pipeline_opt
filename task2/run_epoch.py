import argparse
from enum import Enum
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


def get_model(dataset: SimpleDataset, d_model: int = 1024, nhead: int = 8) -> torch.nn.Module:
    class GPT2Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model
            self.embed = nn.Embedding(
                num_embeddings=len(dataset.vocab),
                embedding_dim=d_model,
                padding_idx=dataset.vocab["<pad>"]
            )
            self.pos_enc = PositionalEncoding(d_model, max_len=dataset.max_length)
            self.decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)

        def forward(self, seqs: torch.Tensor) -> torch.Tensor:
            embeds = self.embed(seqs)
            pos_enc = self.pos_enc(embeds)
            return self.decoder(pos_enc, pos_enc)

    return GPT2Model()


def run_epoch(
        data_mode: DataMode,
        data_path: str = "data.hf",
        batch_size: int = 32,
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
    loader = dataset.create_loader(batch_size=batch_size)
    init_time = time.time() - start
    model = get_model(dataset).to(device)
    times, seqs_len, seqs_diff = [], [], []
    for seqs, seqs_len_ in tqdm(loader, desc=f"{data_mode.name}-{k}"):
        seqs_len += seqs_len_
        seqs_diff.append(max(seqs_len_) - min(seqs_len_))
        start = time.time()
        outputs = model(seqs.to(device))
        if device != "cpu":
            torch.cuda.synchronize()
        times.append(time.time() - start)
    times = times[warmup_steps:]
    return [
        init_time, np.min(times), np.max(times), np.mean(times), np.median(times),
        np.mean(seqs_len[warmup_steps:]), np.mean(seqs_diff[warmup_steps:])
    ]


def add_result(foo, desc, df, **kwargs):
    res = foo(**kwargs)
    print(f'{desc}:', res)
    df.loc[desc] = res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()

    df = pd.DataFrame(columns=["Init", "Min", "Max", "Mean", "Med", "SeqLen", "SeqDiff"])
    add_result(run_epoch, "Brain", df, data_mode=DataMode.BRAIN, batch_size=args.batch_size)
    add_result(run_epoch, "Big Brain", df, data_mode=DataMode.BIG_BRAIN, batch_size=args.batch_size)
    for k in [1, 5, 10, 20, 50, 640]:
        add_result(run_epoch, f"Ultra Big Brain. k = {k}", df,
                   data_mode=DataMode.ULTRA_DUPER_BIG_BRAIN, k=k, batch_size=args.batch_size)
    df.to_csv("result.csv")
