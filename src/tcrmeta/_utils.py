"""Shared, dependency-light utilities used across TCRmeta's embedding,
CSS, UMAP, and energy-shift modules.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Device handling
# --------------------------------------------------------------------------- #
def resolve_device(device: Optional[str] = "auto"):
    """Resolve a user-facing device string into a torch.device.

    Parameters
    ----------
    device:
        One of "auto" (default, uses CUDA if available), "cpu", "cuda",
        "cuda:0", "mps", or an already-constructed torch.device.

    Note: torch is imported lazily here (rather than at module level) so
    that the pure numpy/pandas utilities in this module (downsampling,
    column validation) can be used/tested without requiring torch to be
    installed.
    """
    import torch

    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# --------------------------------------------------------------------------- #
# Downsampling
# --------------------------------------------------------------------------- #
def downsample_multinomial(
    counts: np.ndarray,
    target: int,
    rng: Optional[np.random.Generator] = None,
    seed: int = 0,
) -> np.ndarray:
    """Multinomial downsampling of clone counts to a fixed target depth.

    If the repertoire's total count is already <= target, counts are
    returned unchanged (no upsampling).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total <= target:
        return counts.copy()
    p = counts / total
    return rng.multinomial(target, p)


def normalize_weights(counts: np.ndarray) -> Optional[np.ndarray]:
    """Normalize a count/weight vector to sum to 1. Returns None if the
    vector sums to <= 0 (nothing usable to weight by)."""
    counts = np.asarray(counts, dtype=np.float64)
    s = counts.sum()
    if s <= 0:
        return None
    return counts / s


def downsample_then_embed(
    df: pd.DataFrame,
    downsample: int,
    seed: int,
    device: str,
    embed_kwargs: dict,
) -> pd.DataFrame:
    """Downsample a repertoire's raw counts FIRST, then embed only the
    clones that survive downsampling — used by compute_css/plot_umap.

    Multinomial downsampling to `downsample` total reads keeps at most
    `downsample` unique (cdr3aa, v_gene) rows (each surviving row needs
    >=1 read). Embedding only those rows — instead of embedding the
    repertoire's full unique-clone set and downsampling afterward — can
    be a large speedup for repertoires with many more unique clones than
    `downsample`.

    Caveat: a handful of surviving rows can still get dropped afterward
    if their V gene has no CDR1/CDR2/CDR2.5 entry in TCRmeta's germline
    lookup table (attach_embeddings' internal dropna) — for most real
    repertoires this is rare/negligible, but it means results can differ
    very slightly from embedding-then-downsampling in the (uncommon)
    case where such unrecognized-V-gene rows exist and would otherwise
    have consumed some of the downsampling budget.

    Returns the downsampled + embedded dataframe with an added 'ds'
    float64 column (post-downsample weight per surviving row).

    Always embeds with embedding_type="final" (the antigen-aware,
    contrastively fine-tuned embedding) regardless of what's in
    embed_kwargs — compute_css/plot_umap's reference maps are built on
    that embedding space, so scoring against them requires it. Only
    embed_repertoire() itself exposes a choice of embedding_type.
    """
    from .embedding import attach_embeddings  # lazy: keeps this module

    # importable/testable without torch/transformers.
    df = validate_repertoire_columns(df)
    rng = np.random.default_rng(seed)
    counts = df["count"].to_numpy(dtype=np.int64)
    ds = downsample_multinomial(counts, downsample, rng=rng)
    keep = ds > 0
    df = df.loc[keep].copy()
    df["ds"] = ds[keep].astype(np.float64)
    embed_kwargs = {**embed_kwargs, "embedding_type": "final"}
    df_e = attach_embeddings(df, device=device, **embed_kwargs)
    return df_e


# --------------------------------------------------------------------------- #
# Repertoire column validation
# --------------------------------------------------------------------------- #
REQUIRED_COLUMNS = ("cdr3aa", "v_gene", "count")


def validate_repertoire_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and lightly normalize a TCR repertoire dataframe.

    TCRmeta requires the exact columns 'cdr3aa', 'v_gene', 'count' —
    there is no alias-guessing. If your data uses different column
    names (e.g. Adaptive's 'aminoAcid'/'vGeneName'/'count (templates/reads)',
    or AIRR's 'junction_aa'/'v_call'/'duplicate_count'), rename them
    yourself before calling into TCRmeta.

    Normalization performed (does not change column names):
      - cdr3aa is cast to str and given canonical IMGT flanks (leading
        C, trailing F) if missing.
      - v_gene is cast to str, allele suffix (e.g. '*01') stripped.
      - count is coerced to int; rows with count <= 0 are dropped.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Repertoire dataframe is missing required column(s): {missing}. "
            f"TCRmeta requires exactly these columns: {REQUIRED_COLUMNS}. "
            "Rename your columns before calling this function (no aliases are "
            "auto-detected)."
        )

    df = df.copy()
    df["cdr3aa"] = df["cdr3aa"].astype(str)
    df["cdr3aa"] = df["cdr3aa"].str.replace(r"^(?!C)", "C", regex=True)
    df["cdr3aa"] = df["cdr3aa"].str.replace(r"(?<!F)$", "F", regex=True)

    df["v_gene"] = df["v_gene"].astype(str).str.replace(r"\*.*$", "", regex=True).str.strip()

    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(np.int64)
    df = df[df["count"] > 0].reset_index(drop=True)
    return df


def aggregate_duplicate_clones(df: pd.DataFrame, sort: bool = True) -> pd.DataFrame:
    """Collapse duplicate (cdr3aa, v_gene) rows by summing 'count'.

    `sort` controls whether the result is reordered alphabetically by
    (cdr3aa, v_gene) (pandas groupby's default) or left in first-occurrence
    order. This matters for reproducibility: multinomial downsampling
    consumes the RNG stream in row order, so which convention to use
    depends on what the calling code's original (pre-packaging) script
    did. energy_shift's original script deduplicated with pandas'
    default sort=True before downsampling, so energy.py relies on the
    default here. CSS/UMAP's original scoring scripts never deduplicated
    rows at all (raw CSV order preserved throughout) — those code paths
    (attach_embeddings) intentionally skip calling this function.
    """
    agg_cols = {"count": "sum"}
    other_cols = [c for c in df.columns if c not in ("cdr3aa", "v_gene", "count")]
    for c in other_cols:
        agg_cols[c] = "first"
    return df.groupby(["cdr3aa", "v_gene"], as_index=False, sort=sort).agg(agg_cols)
