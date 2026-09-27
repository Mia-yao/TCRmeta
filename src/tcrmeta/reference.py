"""build_reference(): fit a per-V-gene reference density + UMAP map from
a reference cohort of TCR repertoires. Each V gene's model bundles both
the KDE side (used by compute_css) and the UMAP side (used by plot_umap)
in a single object, since both are fit from the same pooled, weighted
reference embeddings.

This generalizes build_density_maps.py to take raw repertoire dataframes
directly (embedding + per-sample downsampling happen internally) instead
of a precomputed ref_data.pkl.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError as e:  # pragma: no cover
    raise ImportError("umap-learn is required for build_reference/plot_umap: pip install umap-learn") from e

from ._utils import downsample_multinomial

PERCENTILES = [5, 10, 15, 20, 25, 75, 80, 85, 90, 95]


def _prepare_reference_sample(
    df: pd.DataFrame,
    downsample: int,
    rng: np.random.Generator,
    device: str,
    embed_kwargs: dict,
) -> Optional[dict]:
    """Embed one reference-cohort repertoire and downsample its clone
    counts to a fixed depth, returning per-clone V gene labels, 64-dim
    embeddings, and downsampled weights.

    Always embeds with embedding_type="final" (the antigen-aware,
    contrastively fine-tuned embedding) regardless of what's in
    embed_kwargs, so every reference map is built in a single,
    consistent 64-dim embedding space matching compute_css/plot_umap.
    Only embed_repertoire() itself exposes a choice of embedding_type.
    """
    from .embedding import attach_embeddings  # lazy: keeps the pure

    # density/UMAP-fitting logic (_build_one_vgene) importable/testable
    # without torch/transformers installed.
    embed_kwargs = {**embed_kwargs, "embedding_type": "final"}
    df_e = attach_embeddings(df, device=device, **embed_kwargs)
    if len(df_e) < 2:
        return None
    counts = df_e["count"].to_numpy(dtype=np.int64)
    ds = downsample_multinomial(counts, downsample, rng=rng)
    keep = ds > 0
    if keep.sum() < 2:
        return None
    df_e = df_e.loc[keep].copy()
    ds = ds[keep]
    emb = np.vstack(df_e["embedding"].values).astype(np.float32)
    vgene = df_e["v_gene"].to_numpy()
    return {"emb": emb, "ds": ds.astype(np.int64), "vgene": vgene}


def _select_vgenes(samples: List[dict], v_genes: Optional[List[str]], freq_thr: float) -> List[str]:
    if v_genes is not None:
        return list(v_genes)
    total_by_vg: Dict[str, int] = {}
    grand_total = 0
    for s in samples:
        for vg, cnt in zip(s["vgene"], s["ds"]):
            total_by_vg[vg] = total_by_vg.get(vg, 0) + int(cnt)
            grand_total += int(cnt)
    if grand_total == 0:
        return []
    return [vg for vg, cnt in total_by_vg.items() if (cnt / grand_total) >= freq_thr]


def _build_one_vgene(
    vgene: str,
    samples: List[dict],
    pca_dim: int,
    max_umap: int,
    umap_neighbors: int,
    umap_min_dist: float,
    min_samples: int,
    min_clones: int,
    seed: int,
) -> Optional[dict]:
    emb_list, ds_list, n_samples_with_vg = [], [], 0
    for s in samples:
        mask = s["vgene"] == vgene
        if mask.sum() < 2:
            continue
        n_samples_with_vg += 1
        emb_list.append(s["emb"][mask])
        ds_list.append(s["ds"][mask])

    if n_samples_with_vg < min_samples or len(emb_list) == 0:
        return None
    all_emb = np.vstack(emb_list)
    all_ds = np.concatenate(ds_list)
    if all_emb.shape[0] < min_clones:
        return None

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(all_emb)

    actual_dim = min(pca_dim, emb_scaled.shape[0] - 1, emb_scaled.shape[1])
    pca = PCA(n_components=actual_dim, random_state=seed)
    emb_pca = pca.fit_transform(emb_scaled)

    n, d = emb_pca.shape
    bw = n ** (-1.0 / (d + 4))
    kde_model = KernelDensity(kernel="gaussian", bandwidth=bw)
    kde_model.fit(emb_pca, sample_weight=all_ds)
    ref_log_dens = kde_model.score_samples(emb_pca)

    N = emb_scaled.shape[0]
    if N > max_umap:
        rng = np.random.default_rng(seed)
        p = all_ds / all_ds.sum()
        sub = rng.choice(N, size=max_umap, replace=False, p=p)
        emb_for_umap = emb_scaled[sub]
        ds_for_umap = all_ds[sub]
    else:
        emb_for_umap = emb_scaled
        ds_for_umap = all_ds

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(umap_neighbors, emb_for_umap.shape[0] - 1),
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=seed,
        verbose=False,
    )
    umap2d_sub = reducer.fit_transform(emb_for_umap)
    w_sub = ds_for_umap / ds_for_umap.sum()
    ref_centroid_2d = np.average(umap2d_sub, weights=w_sub, axis=0)

    model = {
        "scaler": scaler,
        "pca": pca,
        "kde_model": kde_model,
        "ref_logdens": ref_log_dens.astype(np.float32),
        "umap_reducer": reducer,
        "ref_umap2d": umap2d_sub,
        "ref_ds_weights": ds_for_umap,
        "ref_centroid_2d": ref_centroid_2d,
        "n_clones": int(N),
        "n_clones_umap": int(emb_for_umap.shape[0]),
        "n_samples": n_samples_with_vg,
        "bw": bw,
        "pca_dim": actual_dim,
    }
    for p_val in PERCENTILES:
        model[f"thr_{p_val}"] = float(np.percentile(ref_log_dens, p_val))
    model["pos_threshold"] = model["thr_75"]
    return model


def build_reference(
    repertoires: List[pd.DataFrame],
    v_genes: Optional[List[str]] = None,
    freq_thr: float = 0.01,
    pca_dim: int = 30,
    max_umap: int = 10000,
    umap_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    downsample: int = 10000,
    min_samples: int = 20,
    min_clones: int = 50,
    n_jobs: int = -1,
    seed: int = 42,
    device: str = "auto",
    **embed_kwargs,
) -> Dict[str, dict]:
    """Build a per-V-gene reference density + UMAP map from a reference
    cohort, for use as `reference=` in compute_css()/plot_umap().

    Parameters
    ----------
    repertoires:
        A list of raw repertoire dataframes (one per reference-cohort
        sample), each with columns 'cdr3aa', 'v_gene', 'count'.
    v_genes:
        Explicit list of V genes to build models for. If None, all V
        genes whose pooled frequency across `repertoires` is >= freq_thr
        are used.
    downsample:
        Per-sample target depth for multinomial downsampling before
        pooling (default 10000).
    min_samples / min_clones:
        A V gene is only modeled if at least `min_samples` reference
        samples carry it and the pooled, downsampled clone count is at
        least `min_clones`.

    Returns
    -------
    dict mapping v_gene -> model dict (scaler/pca/kde_model/thresholds/
    umap_reducer/ref_umap2d/ref_ds_weights/ref_centroid_2d/...).
    """
    if len(repertoires) == 0:
        raise ValueError("`repertoires` is empty — need at least one reference sample.")

    rng = np.random.default_rng(seed)
    samples = []
    for df in repertoires:
        s = _prepare_reference_sample(df, downsample, rng, device, embed_kwargs)
        if s is not None:
            samples.append(s)
    if len(samples) == 0:
        raise ValueError("None of the provided repertoires had usable clones after embedding/downsampling.")

    selected_vgenes = _select_vgenes(samples, v_genes, freq_thr)
    if len(selected_vgenes) == 0:
        raise ValueError(
            "No V genes selected — either raise `freq_thr`-eligible genes weren't "
            "found, or pass an explicit `v_genes` list."
        )

    results = Parallel(n_jobs=n_jobs)(
        delayed(_build_one_vgene)(
            vg, samples, pca_dim, max_umap, umap_neighbors, umap_min_dist, min_samples, min_clones, seed
        )
        for vg in selected_vgenes
    )
    reference_map = {vg: model for vg, model in zip(selected_vgenes, results) if model is not None}
    if len(reference_map) == 0:
        raise ValueError(
            "No V gene met min_samples/min_clones thresholds — try lowering "
            "min_samples/min_clones or providing more reference repertoires."
        )
    return reference_map


def save_reference(reference_map: Dict[str, dict], path: str, compress: int = 3) -> None:
    """Save a reference map with joblib (not plain pickle) — joblib
    handles the numpy-array-heavy contents (KDE models, UMAP reducers,
    ref_umap2d arrays, etc.) more robustly, and its compressed format is
    what build_density_maps.py-style reference maps are typically saved
    as already.
    """
    joblib.dump(reference_map, path, compress=compress)


def load_reference(path: str) -> Dict[str, dict]:
    """Load a reference map saved with joblib.dump — this transparently
    handles both compressed and uncompressed joblib pickles (including
    ones produced outside TCRmeta, e.g. by build_density_maps.py-style
    scripts using `joblib.dump(vgene_models, out_path, compress=3)`).
    Plain `pickle.load` cannot read the compressed format and raises
    `UnpicklingError: invalid load key, 'x'` (the zlib stream's magic
    byte) if used instead.
    """
    return joblib.load(path)
