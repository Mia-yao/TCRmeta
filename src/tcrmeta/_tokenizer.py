"""ESM2 tokenizer -> amino-acid token id lookup, built once and cached."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

from transformers import EsmTokenizer

from ._weights import ESM2_BACKBONE

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWYX")


@lru_cache(maxsize=1)
def build_token2idx(backbone_name: str = ESM2_BACKBONE) -> Dict[str, int]:
    tokenizer = EsmTokenizer.from_pretrained(backbone_name)
    token2idx = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS}
    token2idx["[PAD]"] = tokenizer.convert_tokens_to_ids("<pad>")
    token2idx["[MASK]"] = tokenizer.convert_tokens_to_ids("<mask>")
    token2idx["[UNK]"] = tokenizer.convert_tokens_to_ids("<unk>")
    token2idx["[CLS]"] = tokenizer.convert_tokens_to_ids("<cls>")
    token2idx["[EOS]"] = tokenizer.convert_tokens_to_ids("<eos>")
    return token2idx
