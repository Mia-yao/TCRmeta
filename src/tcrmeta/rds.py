"""Task 2: compute_rds — score a query repertoire against a reference
density map (Repertoire Dispersal Score / RDS).

For each V gene, the query repertoire's embeddings are projected through
the reference's fitted scaler + PCA, scored with the reference's KDE to
get a per-clone log-density, and compared against the reference's own
high-density (POS) and low-density (LZO) percentile thresholds:

    MLD     = weighted mean log-density
    POS_p   = weighted fraction of clones at/above the reference's p-th
              percentile log-density (matches the reference's dense/core region)
    LZO_q   = weighted fraction of clones at/below the reference's q-th
              percentile log-density (outlier / shifted-away-from-reference)
    RDS     = LZO_q / (POS_p + eps)   — higher means more of the repertoire
              has drifted into the reference's low-density tail.
    log_RDS = log10(RDS + eps)

Two outputs are always produced: a per-V-gene table (one row per gene
scored), and a whole-repertoire table pooling log-densities across all
scored V genes (weighted by clone count) into one aggregate score.

Performance note: the repertoire is downsampled (multinomial, whole
repertoire) BEFORE embedding — see _utils.downsample_then_embed — so at
most `downsample` unique clones are ever run through the encoder,
rather than every unique clone in the (possibly much larger) raw
repertoire.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._utils import downsample_then_embed
from ._weights import resolve_default_reference
from .reference import load_reference

ReferenceMap = Dict[str, dict]

_DEFAULT_REFERENCE_CACHE: Optional[ReferenceMap] = None


def load_default_reference() -> ReferenceMap:
    """Load (and cache) the reference map shipped with TCRmeta."""
    global _DEFAULT_REFERENCE_CACHE
    if _DEFAULT_REFERENCE_CACHE is None:
        _DEFAULT_REFERENCE_CACHE = load_reference(str(resolve_default_reference()))
    return _DEFAULT_REFERENCE_CACHE


def _resolve_reference(reference: Union[None, str, ReferenceMap]) -> ReferenceMap:
    if reference is None:
        return load_default_reference()
    if isinstance(reference, str):
        return load_reference(reference)
    return reference


def _score_vgene(
    emb: np.ndarray,
    w: np.ndarray,
    model: dict,
    pos_percentile: int,
    neg_percentile: int,
) -> dict:
    thr_pos = model.get(f"thr_{pos_percentile}")
    thr_neg = model.get(f"thr_{neg_percentile}")
    if thr_pos is None or thr_neg is None:
        raise ValueError(
            f"Reference model does not have precomputed thresholds for "
            f"percentiles {pos_percentile}/{neg_percentile}. Available: "
            f"{[k for k in model if k.startswith('thr_')]}"
        )
    expected_dim = getattr(model["scaler"], "n_features_in_", None)
    if expected_dim is not None and emb.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: this reference's V-gene models "
            f"were fit on {expected_dim}-dim embeddings, but the query "
            f"repertoire was embedded to {emb.shape[1]}-dim. compute_rds "
            f"always embeds with embedding_type='final' (64-dim), so this "
            f"usually means the reference map was built with a different, "
            f"incompatible embedding pipeline — rebuild it with "
            f"build_reference() from this version of TCRmeta."
        )
    X = model["pca"].transform(model["scaler"].transform(emb))
    ld = model["kde_model"].score_samples(X)
    wn = w / w.sum()
    mld = float(np.average(ld, weights=wn))
    pos = float(np.average((ld >= thr_pos).astype(float), weights=wn))
    lzo = float(np.average((ld <= thr_neg).astype(float), weights=wn))
    rds = lzo / (pos + 1e-10)
    return {
        "MLD": mld,
        "POS": pos,
        "LZO": lzo,
        "RDS": rds,
        "log_RDS": float(np.log10(rds + 1e-10)),
        "n_clones": int(len(ld)),
        "_logdens": ld,
        "_thr_pos": thr_pos,
        "_thr_neg": thr_neg,
    }


def _pool_whole_repertoire(scored_rows: List[dict]) -> dict:
    if len(scored_rows) == 0:
        return {"MLD": np.nan, "POS": np.nan, "LZO": np.nan, "RDS": np.nan,
                "log_RDS": np.nan, "n_clones": 0, "n_v_genes": 0}

    ld_cat = np.concatenate([r["_ld"] for r in scored_rows])
    w_cat = np.concatenate([r["_w"] for r in scored_rows])
    wn = w_cat / w_cat.sum()
    n_total = sum(r["n_clones"] for r in scored_rows)
    thr_pos = sum(r["_thr_pos"] * r["n_clones"] for r in scored_rows) / n_total
    thr_neg = sum(r["_thr_neg"] * r["n_clones"] for r in scored_rows) / n_total

    mld = float(np.average(ld_cat, weights=wn))
    pos = float(np.average((ld_cat >= thr_pos).astype(float), weights=wn))
    lzo = float(np.average((ld_cat <= thr_neg).astype(float), weights=wn))
    rds = lzo / (pos + 1e-10)
    return {
        "MLD": mld,
        "POS": pos,
        "LZO": lzo,
        "RDS": rds,
        "log_RDS": float(np.log10(rds + 1e-10)),
        "n_clones": int(len(ld_cat)),
        "n_v_genes": len(scored_rows),
    }


def compute_rds(
    df: pd.DataFrame,
    reference: Union[None, str, ReferenceMap] = None,
    v_gene: Optional[Union[str, Sequence[str]]] = None,
    downsample: int = 10000,
    min_clones: int = 2,
    pos_percentile: int = 85,
    neg_percentile: int = 15,
    device: str = "auto",
    seed: int = 42,
    per_gene_path: Optional[str] = None,
    whole_path: Optional[str] = None,
    **embed_kwargs,
):
    """Score a query repertoire's Repertoire Dispersal Score (RDS) against
    a reference density map.

    Parameters
    ----------
    df:
        Query repertoire dataframe ('cdr3aa', 'v_gene', 'count').
    reference:
        A reference map (as returned by build_reference()), a path to a
        pickled one, or None to use TCRmeta's shipped default reference.
    v_gene:
        A V gene, a list of V genes, or None (default) to score every V
        gene present in the reference.
    downsample:
        Multinomial downsampling target depth for the whole repertoire
        before per-V-gene scoring (default 10000).
    min_clones:
        Minimum clones required for a V gene to be scored.

    Returns
    -------
    (per_gene_df, whole_df):
        per_gene_df has one row per scored V gene (columns: v_gene, MLD,
        POS_<p>, LZO_<q>, RDS, log_RDS, n_clones).
        whole_df is a single-row dataframe pooling all scored V genes'
        densities (weighted by clone count) into one aggregate score.
    """
    # Downsample the raw repertoire FIRST, then embed only the surviving
    # clones (at most `downsample` unique rows) — much cheaper than
    # embedding the whole repertoire and downsampling afterward when the
    # repertoire has many more unique clones than `downsample`. See
    # downsample_then_embed()'s docstring for the (rare) caveat this
    # introduces vs. embed-then-downsample.
    df_e = downsample_then_embed(df, downsample, seed, device, embed_kwargs)
    if len(df_e) < 2:
        raise ValueError("Repertoire has fewer than 2 embeddable clones after downsampling.")

    ref_map = _resolve_reference(reference)

    if v_gene is None:
        vgenes_to_score = list(ref_map.keys())
    elif isinstance(v_gene, str):
        vgenes_to_score = [v_gene]
    else:
        vgenes_to_score = list(v_gene)

    vgene_arr = df_e["v_gene"].to_numpy()
    emb_all = np.vstack(df_e["embedding"].values).astype(np.float32)
    ds_all = df_e["ds"].to_numpy(dtype=np.float64)

    rows = []
    pool_inputs = []
    for vg in vgenes_to_score:
        if vg not in ref_map:
            continue
        mask = vgene_arr == vg
        if mask.sum() < min_clones:
            continue
        emb = emb_all[mask]
        w = ds_all[mask]
        scored = _score_vgene(emb, w, ref_map[vg], pos_percentile, neg_percentile)
        rows.append({
            "v_gene": vg,
            "MLD": scored["MLD"],
            f"POS_{pos_percentile}": scored["POS"],
            f"LZO_{neg_percentile}": scored["LZO"],
            "RDS": scored["RDS"],
            "log_RDS": scored["log_RDS"],
            "n_clones": scored["n_clones"],
        })
        pool_inputs.append({
            "_ld": scored["_logdens"], "_w": w,
            "_thr_pos": scored["_thr_pos"], "_thr_neg": scored["_thr_neg"],
            "n_clones": scored["n_clones"],
        })

    per_gene_df = pd.DataFrame(rows)
    whole_df = pd.DataFrame([_pool_whole_repertoire(pool_inputs)])

    if per_gene_path is not None:
        per_gene_df.to_csv(per_gene_path, index=False)
    if whole_path is not None:
        whole_df.to_csv(whole_path, index=False)

    return per_gene_df, whole_df
