"""TCRmeta: antigen-aware TCR repertoire embeddings and downstream
repertoire-level analysis (CSS scoring, UMAP projection, energy-distance
shift), built on a pretrained TCRmeta encoder + fine-tuning.
"""
from .embedding import embed_repertoire
from .reference import build_reference, load_reference, save_reference
from .css import compute_css, load_default_reference
from .umap_plot import plot_umap
from .energy import energy_shift

__all__ = [
    "embed_repertoire",
    "build_reference",
    "save_reference",
    "load_reference",
    "load_default_reference",
    "compute_css",
    "plot_umap",
    "energy_shift",
]

__version__ = "0.1.5"
