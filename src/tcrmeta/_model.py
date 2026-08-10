"""Model architectures: the ESM2-backbone masked-LM encoder used during
pretraining (only its encoder + CLS embedding is used at inference time),
and the ResMLP Siamese projection head used during contrastive fine-tuning.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel

from ._weights import ESM2_BACKBONE


class TCRMaskedModel(nn.Module):
    """ESM2 encoder + masked-LM head. Only `encoder` (and its CLS-token
    embedding) is used downstream by TCRmeta; `lm_head` is retained so
    pretrained checkpoints load without a state_dict mismatch.
    """

    def __init__(self, backbone_name: str = ESM2_BACKBONE, vocab_size: int = 33):
        super().__init__()
        self.encoder = EsmModel.from_pretrained(backbone_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.lm_head = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, input_tokens, attention_mask=None, return_embedding=False):
        outputs = self.encoder(input_ids=input_tokens, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        logits = self.lm_head(sequence_output)
        return (logits, sequence_output) if return_embedding else logits


class ResMLPBlock(nn.Module):
    """Pre-norm residual MLP block: LayerNorm -> Linear -> GELU -> Dropout
    -> Linear -> Dropout, added back with a learnable (small-initialized)
    residual scale so the block starts close to identity.
    """

    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + self.res_scale * h


class ResMLPStack(nn.Module):
    """`depth` stacked ResMLPBlocks."""

    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.1, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([ResMLPBlock(dim, hidden, dropout) for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class SiameseOnEmb_ResMLP(nn.Module):
    """Contrastive fine-tuning projection head: ResMLP re-mixes the
    480-dim base embedding, then a linear projection + L2 normalization
    maps it onto the unit sphere in the antigen-aware embedding space.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 64,
        hidden: int = 768,
        dropout: float = 0.1,
        depth: int = 3,
    ):
        super().__init__()
        self.mlp = ResMLPStack(in_dim, hidden=hidden, dropout=dropout, depth=depth)
        self.proj = nn.Linear(in_dim, proj_dim, bias=False)
        # Learnable scalars carried over from contrastive training; unused
        # at inference (encode()/forward() distance is computed downstream),
        # kept only so pretrained state_dicts load without a mismatch.
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.m = nn.Parameter(torch.tensor(1.0))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        z = self.proj(x)
        return F.normalize(z, dim=-1)

    def forward(self, x1, x2):
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        d = torch.norm(z1 - z2, dim=-1) * self.scale
        return z1, z2, d


def load_state_dict_flexible(model: nn.Module, ckpt) -> None:
    """Load a checkpoint saved under any of the common key conventions
    (raw state_dict, {"model_state_dict": ...}, {"state_dict": ...},
    an EMA shadow dict, etc.).
    """
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        model.load_state_dict(ckpt, strict=True)
        return
    for key in ("model_state_dict", "ema_state_dict", "state_dict", "model"):
        if key in ckpt:
            model.load_state_dict(ckpt[key], strict=True)
            return
    if "ema_shadow" in ckpt:
        sd = model.state_dict()
        for k, v in ckpt["ema_shadow"].items():
            if k in sd:
                sd[k] = v
        model.load_state_dict(sd, strict=False)
        return
    raise RuntimeError(f"Unrecognized checkpoint format, top-level keys: {list(ckpt.keys())}")
