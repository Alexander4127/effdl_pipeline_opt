from abc import abstractmethod
from collections import Counter, defaultdict
import json
import numpy as np
from pathlib import Path
from random import shuffle
from typing import Optional

from datasets import load_from_disk
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Sampler
import torchtext
from torchtext.vocab import vocab as Vocab


MAX_LENGTH = 640


def build_vocab(data, tokenizer):
    counter = Counter()
    for line in data:
        counter.update(tokenizer(line["text"]))
    vocab = Vocab(counter, min_freq=2000, specials=['<pad>', '<unk>', '<bos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab


class SimpleDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.data = load_from_disk(data_path)
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        if Path("vocab.pth").exists():
            self.vocab = torch.load("vocab.pth")
        else:
            self.vocab = build_vocab(self.data, self.tokenizer)
            torch.save(self.vocab, "vocab.pth")
        if Path("upd_data.hf").exists():
            self.data = load_from_disk("upd_data.hf")
        else:
            self.data = self.data.map(
                lambda line: {
                    "tokens": self.vocab.forward(
                        ["<bos>"] + self.tokenizer(line["text"])[:max_length - 2] + ["<eos>"]
                    ),
                    "text": line["text"]
                }
            )
            self.data.save_to_disk("upd_data.hf")

    def __getitem__(self, item: int):
        return torch.tensor(self.data[item]["tokens"])

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def create_loader(self, batch_size):
        raise NotImplementedError()


class BrainDataset(SimpleDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        super().__init__(data_path, max_length)

    def create_loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, collate_fn=Collator(max_length=self.max_length))


class BigBrainDataset(SimpleDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        super().__init__(data_path, max_length)

    def create_loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, collate_fn=Collator(max_length=None))


class UltraDuperBigBrainDataset(SimpleDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, k: int = 1):
        super().__init__(data_path, max_length)
        self.k = k

    def __getitem__(self, lst_idx: list[int]):
        assert type(lst_idx) == list, f'Unexpected type of lst_idx = {type(lst_idx)}'
        return [torch.tensor(self.data[idx]["tokens"]) for idx in lst_idx]

    def create_loader(self, batch_size):
        sampler = UltraDuperBigBrainBatchSampler(self, batch_size, self.k)
        return DataLoader(self, batch_sampler=sampler, collate_fn=Collator(max_length=None))


class Collator:
    def __init__(self, max_length: Optional[int] = MAX_LENGTH):
        """
        Collate func for Brain and BigBrain datasets
        :param max_length: maximum sequence length to pad to (for "Brain" approach only)
        """
        self.max_length = max_length

    def __call__(self, batch: list[torch.Tensor]):
        """
            Pad each sequence of the incoming sequences list
            :param batch: a list of the objects received from the dataset by __getitem__
            :return: padded sequences
        """
        lengths = torch.tensor([len(seq) for seq in batch])
        max_length = self.max_length if self.max_length is not None else max(lengths)
        seqs = torch.zeros([max_length, len(batch)], dtype=torch.int32)
        for idx, seq in enumerate(batch):
            seqs[:len(seq), idx] = seq
        return seqs


class UltraDuperBigBrainBatchSampler(Sampler):
    def __init__(self, dataset: SimpleDataset, batch_size: int, k: int):
        super().__init__(dataset)
        self.length_mapper = defaultdict(set)
        self.batch_size = batch_size
        for idx, line in enumerate(dataset.data):
            self.length_mapper[len(line["tokens"])].add(idx)

        for length in self.length_mapper.keys():
            for tmp_len in range(length - k + 1, length):
                if tmp_len in self.length_mapper:
                    assert len(self.length_mapper[tmp_len] & self.length_mapper[length]) == 0
                    self.length_mapper[tmp_len] |= self.length_mapper[length]

        self.length_mapper = {k_: list(v) for k_, v in self.length_mapper.items()}

    def __len__(self):
        return len(self.length_mapper)

    def __iter__(self):
        lst_ks = list(self.length_mapper.keys())
        shuffle(lst_ks)
        for k in lst_ks:
            length = min(self.batch_size, len(self.length_mapper[k]))
            yield list(np.random.choice(self.length_mapper[k], size=length, replace=False))
