"""Task 4: energy_shift — weighted energy-distance shift between two TCR
repertoires, per V gene plus a frequency-weighted summary.

Matches the reference load_one_sample() / compute_pair_metrics_by_v_long()
/ collapse_one_pair() pipeline, with ONE deliberate deviation for speed:

1. Each repertoire is downsampled FIRST — a WHOLE-REPERTOIRE target
   depth (`downsample`), not per-V-gene — and only the clones that
   survive downsampling are then embedded (dropping any that still have
   no CDR1/CDR2/CDR2.5 lookup entry). The reference script does this in
   the opposite order (embed/drop-unmapped, THEN downsample); we embed
   only the downsampled survivors instead since embedding real repertoire
   data means running the actual encoder model, and at most `downsample`
   unique clones ever need it this way, vs. every unique clone in the
   (possibly much larger) raw repertoire. See downsample_then_embed's
   analogous docstring in _utils.py for the (rare) caveat this
   introduces vs. embed-then-downsample. Only after downsampling does
   the result get split by V gene, so a V gene's post-downsample clone
   count is only its proportional share of `downsample`, not
   `downsample` itself.

2. Critically, each repertoire's downsampling uses its OWN fresh
   `np.random.default_rng(seed)` — the *same* `seed` value for every
   repertoire, but a brand-new, independent generator each time (NOT one
   rng shared/advanced sequentially across the pair). This matches
   load_one_sample() being called independently per sample (typically
   once per sample, cached, and reused across many pairs): a given
   repertoire's downsampled result only depends on its own data and
   `seed`, never on which other repertoire it's being compared against.

3. Per shared V gene (with >= min_clones clones on both sides after
   downsampling), the (Szekely-Rizzo) weighted energy distance is
   computed:

    D(X, wX, Y, wY) = 2*sum(wX_i wY_j |Xi-Yj|) - sum(wX_i wX_j |Xi-Xj|)
                       - sum(wY_i wY_j |Yi-Yj|)

4. The whole-repertoire summary weights each V gene by its RAW
   post-downsample read depth — `v_weight = (depth_1 + depth_2) / 2`,
   where depth_i = sum(ds_count) for that V gene in repertoire i (NOT
   normalized by the repertoire's total depth, and NOT clone/row count)
   — then takes a weighted average of energy_distance across V genes.
   This matches collapse_one_pair() exactly (minus its jaccard column,
   which TCRmeta's energy_shift doesn't compute).

v_gene defaults to None, meaning all V genes present (with enough
clones) in both repertoires are scored — unlike compute_css/plot_umap,
this is cheap enough to not need restricting by default.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._utils import (
    downsample_multinomial,
    normalize_weights,
    resolve_device,
    validate_repertoire_columns,
)


def _euclidean_dist_matrix_numpy(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    # float32 throughout, matching the original script's torch.float32
    # distance computation exactly (rather than upcasting to float64,
    # which is more precise but won't bit-match the original's results).
    X = np.asarray(X, dtype=np.float32)
    Y = X if Y is None else np.asarray(Y, dtype=np.float32)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True)
    D2 = x2 + y2.T - 2.0 * (X @ Y.T)
    np.maximum(D2, 0, out=D2)
    return np.sqrt(D2, out=D2)


def _weighted_energy_distance_numpy(X: np.ndarray, wX: np.ndarray, Y: np.ndarray, wY: np.ndarray) -> float:
    """Weighted (Szekely-Rizzo) energy distance between two weighted
    point clouds, pure numpy. This is the reference implementation used
    on CPU, and for testing without torch installed.
    """
    wX = np.asarray(wX, dtype=np.float32)
    wY = np.asarray(wY, dtype=np.float32)
    Dxy = _euclidean_dist_matrix_numpy(X, Y)
    Dxx = _euclidean_dist_matrix_numpy(X)
    Dyy = _euclidean_dist_matrix_numpy(Y)
    term_xy = 2.0 * (wX[:, None] * wY[None, :] * Dxy).sum()
    term_xx = (wX[:, None] * wX[None, :] * Dxx).sum()
    term_yy = (wY[:, None] * wY[None, :] * Dyy).sum()
    return float(term_xy - term_xx - term_yy)


def _weighted_energy_distance_torch(X: np.ndarray, wX: np.ndarray, Y: np.ndarray, wY: np.ndarray, device) -> float:
    import torch

    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)
    wX_t = torch.as_tensor(wX, dtype=torch.float32, device=device)
    wY_t = torch.as_tensor(wY, dtype=torch.float32, device=device)

    Dxy = torch.cdist(X_t, Y_t, p=2)
    Dxx = torch.cdist(X_t, X_t, p=2)
    Dyy = torch.cdist(Y_t, Y_t, p=2)

    term_xy = 2.0 * (wX_t[:, None] * wY_t[None, :] * Dxy).sum()
    term_xx = (wX_t[:, None] * wX_t[None, :] * Dxx).sum()
    term_yy = (wY_t[:, None] * wY_t[None, :] * Dyy).sum()
    return float((term_xy - term_xx - term_yy).item())


def _pairwise_weighted_energy_distance(
    X1: np.ndarray, w1: np.ndarray, X2: np.ndarray, w2: np.ndarray, device
) -> float:
    """Weighted energy distance between two point clouds. Uses a torch
    GPU path when `device` is an actual accelerator (cuda/mps) — the
    O(n1*n2) distance matrices benefit from it — and falls back to the
    plain-numpy implementation on CPU, which avoids torch tensor/device
    overhead for the common case and needs no torch installation at all.
    """
    device_type = getattr(device, "type", str(device))
    if device_type in ("cuda", "mps"):
        return _weighted_energy_distance_torch(X1, w1, X2, w2, device)
    return _weighted_energy_distance_numpy(X1, w1, X2, w2)


def _prepare_for_energy(
    df: pd.DataFrame,
    downsample: int,
    seed: int,
    device: str,
    embed_kwargs: dict,
) -> pd.DataFrame:
    """Downsample ONE repertoire (independently of whatever it's being
    compared against) to `downsample` total reads, then embed only the
    clones that survive downsampling.

    Takes `seed` directly (not a shared rng object) and creates its own
    fresh `np.random.default_rng(seed)` right here — matching the
    reference load_one_sample(), where downsampling happens once per
    sample with a freshly-seeded generator, independent of any other
    sample. This means a given repertoire's downsampled result depends
    only on its own data and `seed`, never on which other repertoire
    it's being paired with in energy_shift() — important since the same
    repertoire may be reused across many calls/pairs and should
    downsample identically every time.

    Downsampling happens BEFORE embedding (the one deliberate deviation
    from the reference, for speed — see module docstring). Multinomial
    downsampling to `downsample` total reads keeps at most `downsample`
    unique (cdr3aa, v_gene) rows, so this embeds at most `downsample`
    clones instead of every unique clone in the (possibly much larger)
    raw repertoire.

    Returns the (possibly reduced) dataframe with an added 'ds_count'
    column (only rows that survived downsampling, i.e. ds_count > 0).

    Deliberately does NOT call aggregate_duplicate_clones: the reference
    load_one_sample() never deduplicates (cdr3aa, v_gene) rows before
    downsampling — raw per-row structure (including any genuine
    duplicate clone rows) is preserved throughout.

    Always embeds with embedding_type="final" (the antigen-aware,
    contrastively fine-tuned embedding) regardless of what's in
    embed_kwargs — energy_shift's weighted-energy-distance calculation
    is defined against that embedding space. Only embed_repertoire()
    itself exposes a choice of embedding_type.
    """
    from .embedding import attach_embeddings  # lazy: keeps the pure distance

    # math in this module importable/testable without torch/transformers.
    df = validate_repertoire_columns(df)

    rng = np.random.default_rng(seed)
    counts = df["count"].to_numpy(dtype=np.int64)
    counts_ds = downsample_multinomial(counts, downsample, rng=rng)
    keep = counts_ds > 0
    df = df.loc[keep].copy()
    df["ds_count"] = counts_ds[keep]

    embed_kwargs = {**embed_kwargs, "embedding_type": "final"}
    df_e = attach_embeddings(df, device=device, **embed_kwargs)
    return df_e


def _weighted_summary(per_gene_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-V-gene energy_distance table into one
    frequency-weighted summary row — matches the reference
    collapse_one_pair() exactly.

    Weight per V gene ('v_weight') is `(depth_1 + depth_2) / 2`, where
    depth_i = sum(ds_count) for that V gene in repertoire i — i.e. the
    RAW post-downsample read depth assigned to that V gene (NOT
    normalized by the repertoire's total depth, and NOT clone/row
    count). energy_weighted is then
    `(energy_distance * v_weight).sum() / v_weight.sum()`. Only rows
    with non-NaN energy_distance and v_weight > 0 are used (matching
    `dropna(subset=['energy', 'v_weight'])` + `df[df['v_weight'] > 0]`).
    """
    required = {"depth_1", "depth_2", "energy_distance"}
    if len(per_gene_df) == 0 or not required.issubset(per_gene_df.columns):
        return pd.DataFrame([{"energy_weighted": np.nan, "n_v_genes": 0}])

    df = per_gene_df.copy()
    df["v_weight"] = (df["depth_1"] + df["depth_2"]) / 2.0
    df = df.dropna(subset=["energy_distance", "v_weight"])
    df = df[df["v_weight"] > 0]
    if len(df) == 0:
        return pd.DataFrame([{"energy_weighted": np.nan, "n_v_genes": 0}])

    w = df["v_weight"].to_numpy(dtype=np.float64)
    e = df["energy_distance"].to_numpy(dtype=np.float64)
    energy_weighted = float((e * w).sum() / w.sum())

    return pd.DataFrame([{
        "energy_weighted": energy_weighted,
        "n_v_genes": int(len(df)),
    }])


def energy_shift(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    v_gene: Optional[Union[str, Sequence[str]]] = None,
    downsample: int = 10000,
    min_clones: int = 100,
    device: str = "auto",
    seed: int = 123,
    per_gene_path: Optional[str] = None,
    weighted_summary_path: Optional[str] = None,
    **embed_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the weighted energy-distance shift between two repertoires.

    Parameters
    ----------
    df1, df2:
        Repertoire dataframes ('cdr3aa', 'v_gene', 'count').
    v_gene:
        A V gene, list of V genes, or None (default) to use every V gene
        present in both repertoires with enough clones. Unlike
        compute_css/plot_umap, results across V genes are returned
        together in one table (computing all genes is cheap here).
    downsample:
        WHOLE-REPERTOIRE target depth for multinomial downsampling
        (default 10000), applied ONCE to each repertoire's full clone-
        count vector (every V gene combined) before splitting by V gene
        — matching the original script exactly. A given V gene's
        post-downsample clone count is only its proportional share of
        `downsample`, not `downsample` itself.
    min_clones:
        Minimum downsampled clones (per repertoire, per V gene, after
        the whole-repertoire downsample above) required for a V gene to
        be scored (default 100).

    Returns
    -------
    (per_gene_df, weighted_summary_df):
        per_gene_df: one row per V gene (v_gene, n_clones_1, n_clones_2,
        depth_1, depth_2, energy_distance). depth_i is the raw
        post-downsample read depth for that V gene in repertoire i, used
        (as (depth_1+depth_2)/2) to build the frequency-weighted
        summary.
        weighted_summary_df: single-row dataframe with 'energy_weighted'
        (weighted by (depth_1+depth_2)/2 across V genes) and
        'n_v_genes'.
    """
    dev = resolve_device(device)

    # Each repertoire is downsampled independently — its own fresh
    # rng(seed), not a shared/advanced one — so df1's result never
    # depends on df2 (or vice versa). See _prepare_for_energy's
    # docstring.
    df1_e = _prepare_for_energy(df1, downsample, seed, device, embed_kwargs)
    df2_e = _prepare_for_energy(df2, downsample, seed, device, embed_kwargs)

    empty_cols = ["v_gene", "n_clones_1", "n_clones_2", "depth_1", "depth_2", "energy_distance"]
    if len(df1_e) < 2 or len(df2_e) < 2:
        per_gene_df = pd.DataFrame(columns=empty_cols)
        weighted_summary_df = _weighted_summary(per_gene_df)
        return per_gene_df, weighted_summary_df

    if v_gene is None:
        vgenes = sorted(set(df1_e["v_gene"].unique()) & set(df2_e["v_gene"].unique()))
    elif isinstance(v_gene, str):
        vgenes = [v_gene]
    else:
        vgenes = list(v_gene)

    rows = []
    for vg in vgenes:
        sub1 = df1_e.loc[df1_e["v_gene"] == vg]
        sub2 = df2_e.loc[df2_e["v_gene"] == vg]
        n1, n2 = len(sub1), len(sub2)
        if n1 < min_clones or n2 < min_clones:
            continue
        w1 = normalize_weights(sub1["ds_count"].to_numpy())
        w2 = normalize_weights(sub2["ds_count"].to_numpy())
        if w1 is None or w2 is None:
            continue
        X1 = np.vstack(sub1["embedding"].values).astype(np.float32)
        X2 = np.vstack(sub2["embedding"].values).astype(np.float32)
        ed = _pairwise_weighted_energy_distance(X1, w1, X2, w2, dev)

        rows.append({
            "v_gene": vg,
            "n_clones_1": n1,
            "n_clones_2": n2,
            "depth_1": float(sub1["ds_count"].sum()),
            "depth_2": float(sub2["ds_count"].sum()),
            "energy_distance": ed,
        })

    per_gene_df = pd.DataFrame(rows)
    weighted_summary_df = _weighted_summary(per_gene_df)

    if per_gene_path is not None:
        per_gene_df.to_csv(per_gene_path, index=False)
    if weighted_summary_path is not None:
        weighted_summary_df.to_csv(weighted_summary_path, index=False)

    return per_gene_df, weighted_summary_df
