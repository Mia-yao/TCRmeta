"""Task 3: plot_umap — project a query repertoire into a reference's
per-V-gene UMAP space and plot its density against the reference's
(grey contours = reference, colored contours = query), matching the
manuscript figure style.

v_gene must be specified (a single gene, or a list) since each V gene
has its own independently-fit reference embedding space; scoring/
plotting "all V genes" by default would be slow and isn't meaningful
to pool across genes.

Performance note: the repertoire is downsampled (multinomial, whole
repertoire) BEFORE embedding — see _utils.downsample_then_embed — so at
most `downsample` unique clones are ever run through the encoder,
rather than every unique clone in the (possibly much larger) raw
repertoire.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from ._utils import downsample_then_embed
from .css import ReferenceMap, _resolve_reference

GRID_N = 300
REF_CONTOUR_LEVEL_FRACS = (0.3, 0.6)
QUERY_CONTOUR_PERCENTILES = (50, 70, 85, 95)


def _compute_kde_grid(xy: np.ndarray, weights: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> Optional[np.ndarray]:
    if xy is None or len(xy) < 10:
        return None
    try:
        w = weights / weights.sum()
        kde = gaussian_kde(xy.T, bw_method="scott", weights=w)
        return kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    except Exception:
        return None


def _plot_one_vgene(
    vg: str,
    emb_vg: np.ndarray,
    w_vg: np.ndarray,
    model: dict,
    color: str,
    grid_n: int,
    figsize: Tuple[float, float],
) -> Tuple[plt.Figure, pd.DataFrame]:
    expected_dim = getattr(model["scaler"], "n_features_in_", None)
    if expected_dim is not None and emb_vg.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: this reference's '{vg}' model was "
            f"fit on {expected_dim}-dim embeddings, but the query repertoire "
            f"was embedded to {emb_vg.shape[1]}-dim. plot_umap always embeds "
            f"with embedding_type='final' (64-dim), so this usually means the "
            f"reference map was built with a different, incompatible "
            f"embedding pipeline — rebuild it with build_reference() from "
            f"this version of TCRmeta."
        )
    X_scaled = model["scaler"].transform(emb_vg)
    X_umap = model["umap_reducer"].transform(X_scaled)

    ref_umap = model["ref_umap2d"]
    ref_w = model.get("ref_ds_weights", np.ones(len(ref_umap)))

    x0, x1 = np.percentile(ref_umap[:, 0], 1) - 1, np.percentile(ref_umap[:, 0], 99) + 1
    y0, y1 = np.percentile(ref_umap[:, 1], 1) - 1, np.percentile(ref_umap[:, 1], 99) + 1
    xx, yy = np.mgrid[x0:x1:complex(grid_n), y0:y1:complex(grid_n)]

    dens_ref = _compute_kde_grid(ref_umap, ref_w.astype(np.float64), xx, yy)
    dens_query = _compute_kde_grid(X_umap, w_vg, xx, yy)

    fig, ax = plt.subplots(figsize=figsize)
    if dens_ref is not None:
        ax.contourf(xx, yy, dens_ref, levels=20, cmap="Greys", alpha=0.35, zorder=0)
        ref_levels = [dens_ref.max() * t for t in REF_CONTOUR_LEVEL_FRACS]
        ax.contour(xx, yy, dens_ref, levels=ref_levels, colors="gray", linewidths=0.8, alpha=0.6, zorder=1)

    if dens_query is not None:
        valid = dens_query[dens_query > 0]
        if len(valid) > 4:
            levels_q = np.percentile(valid, list(QUERY_CONTOUR_PERCENTILES))
            ax.contourf(xx, yy, dens_query, levels=[levels_q[0], dens_query.max()],
                        colors=[color], alpha=0.15, zorder=2)
            ax.contour(xx, yy, dens_query, levels=levels_q, colors=[color],
                       linewidths=1.6, alpha=0.9, zorder=3)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("UMAP-1", fontsize=10)
    ax.set_ylabel("UMAP-2", fontsize=10)
    ax.set_title(vg, fontsize=12, fontweight="bold", color=color)
    fig.tight_layout()

    coords_df = pd.DataFrame({
        "v_gene": vg,
        "umap_1": X_umap[:, 0],
        "umap_2": X_umap[:, 1],
        "weight": w_vg,
    })
    return fig, coords_df


def plot_umap(
    df: pd.DataFrame,
    reference: Union[None, str, ReferenceMap] = None,
    v_gene: Union[str, Sequence[str]] = None,
    downsample: int = 10000,
    color: str = "#1565C0",
    grid_n: int = GRID_N,
    figsize: Tuple[float, float] = (5, 5),
    device: str = "auto",
    seed: int = 42,
    save_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    **embed_kwargs,
):
    """Plot a query repertoire's UMAP density against a reference map,
    for one or more V genes. v_gene is required.

    Returns
    -------
    If v_gene is a single string: (fig, coords_df) for that V gene.
    If v_gene is a list: dict {v_gene: (fig, coords_df)} — one figure
    per V gene, never pooled together.

    save_path is only valid when v_gene is a single string (exact file
    to write). save_dir writes one file per V gene named "<v_gene>.png"
    when v_gene is a list.
    """
    if v_gene is None:
        raise ValueError(
            "plot_umap requires v_gene to be specified (a string or a list of "
            "strings) — each V gene has its own reference embedding space, so "
            "there's no meaningful 'all V genes' default here."
        )

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
    vgene_arr = df_e["v_gene"].to_numpy()
    emb_all = np.vstack(df_e["embedding"].values).astype(np.float32)
    ds_all = df_e["ds"].to_numpy(dtype=np.float64)

    single_gene = isinstance(v_gene, str)
    vgene_list = [v_gene] if single_gene else list(v_gene)

    results: Dict[str, Tuple[plt.Figure, pd.DataFrame]] = {}
    for vg in vgene_list:
        if vg not in ref_map:
            raise ValueError(f"V gene '{vg}' not found in the reference map.")
        mask = vgene_arr == vg
        if mask.sum() < 10:
            raise ValueError(f"V gene '{vg}' has fewer than 10 clones in this repertoire after downsampling.")
        fig, coords_df = _plot_one_vgene(vg, emb_all[mask], ds_all[mask], ref_map[vg], color, grid_n, figsize)
        results[vg] = (fig, coords_df)

        if single_gene and save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        elif (not single_gene) and save_dir is not None:
            fig.savefig(f"{save_dir.rstrip('/')}/{vg}.png", dpi=300, bbox_inches="tight")

    if single_gene:
        return results[v_gene]
    return results
