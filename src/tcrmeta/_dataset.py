"""Fixed-length tokenization of (CDR1 + CDR2 + CDR2.5 + CDR3) TCR
sequences for the ESM2 base encoder.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_MAX_LEN = 48


class TCRDatasetFixedLen(Dataset):
    def __init__(self, df: pd.DataFrame, token2idx: Dict[str, int], max_len: int = DEFAULT_MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.t2i = token2idx
        self.max_len = max_len
        self.cls = token2idx["[CLS]"]
        self.pad = token2idx["[PAD]"]
        self.eos = token2idx["[EOS]"]
        self.unk = token2idx["[UNK]"]
        self.gap = token2idx["X"]  # separator between CDR segments

    def _encode_seq(self, s: str) -> List[int]:
        return [self.t2i.get(a, self.unk) for a in s]

    def __getitem__(self, i) -> Tuple[Tuple[str, str], List[int]]:
        r = self.df.iloc[i]
        v_seq = r.cdr1 + "X" + r.cdr2 + "X" + r.cdr2_5
        tokens = (
            [self.cls]
            + self._encode_seq(v_seq)
            + [self.gap]
            + self._encode_seq(r.cdr3aa)
            + [self.eos]
        )
        if len(tokens) < self.max_len:
            tokens = tokens + [self.pad] * (self.max_len - len(tokens))
        else:
            tokens = tokens[: self.max_len]
        key = (r.cdr3aa, r.v_gene)
        return key, tokens

    def __len__(self) -> int:
        return len(self.df)


def fixedlen_collate(batch):
    keys, seqs = zip(*batch)
    input_ids = torch.tensor(seqs, dtype=torch.long)
    return keys, input_ids
