"""Model artifact + reference map download/caching.

All non-code artifacts (the fine-tuned base encoder checkpoint, the 7
projection-head checkpoints, the GPA rotation matrices, the V-gene CDR
lookup table, and the default shipped reference map) are hosted on the
Hugging Face Hub rather than bundled into the PyPI package, and are
downloaded once and cached locally (via huggingface_hub, which handles
its own on-disk cache).

Set the TCRMETA_HF_REPO environment variable to point at a different
Hub repo (e.g. a private mirror), and TCRMETA_WEIGHTS_DIR to bypass the
Hub entirely and load artifacts from a local directory instead (useful
offline, or before the Hub repo exists).

If the Hub repo is PRIVATE, authentication is required to download from
it. Either run `huggingface-cli login` once (huggingface_hub then reuses
that cached token automatically), or set the TCRMETA_HF_TOKEN
environment variable to an access token explicitly (useful in
non-interactive environments like CI or a shared cluster).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# TODO: replace with the actual Hugging Face Hub repo id once created,
# e.g. "mingyaopan/tcrmeta-weights". Can be overridden without a code
# change via the TCRMETA_HF_REPO environment variable.
DEFAULT_HF_REPO = os.environ.get("TCRMETA_HF_REPO", "mingyaopan/tcrmeta-weights")

# Filenames as they should exist in the Hub repo / local weights dir.
BASE_CKPT_FILE = "base_encoder/best_model.pth"
ROTATIONS_FILE = "finetune/rotations_gpa.pkl"
VGENE_CDR_FILE = "reference_data/vgene_cdr.csv"
PROJ_HEAD_FILES = [f"finetune/proj_head_{i}.pt" for i in range(7)]
DEFAULT_REFERENCE_FILE = "reference_maps/young_reference.pkl"

ESM2_BACKBONE = "facebook/esm2_t12_35M_UR50D"


def _local_weights_dir() -> Optional[Path]:
    d = os.environ.get("TCRMETA_WEIGHTS_DIR")
    return Path(d) if d else None


def get_cache_dir() -> Path:
    cache_dir = Path(os.environ.get("TCRMETA_CACHE_DIR", Path.home() / ".cache" / "tcrmeta"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def resolve_artifact(relative_path: str, repo_id: str = DEFAULT_HF_REPO) -> Path:
    """Return a local filesystem path for a named artifact, downloading
    (and caching) it from the Hugging Face Hub if it isn't available
    locally via TCRMETA_WEIGHTS_DIR.
    """
    local_dir = _local_weights_dir()
    if local_dir is not None:
        local_path = local_dir / relative_path
        if not local_path.exists():
            raise FileNotFoundError(
                f"TCRMETA_WEIGHTS_DIR is set to {local_dir}, but {relative_path} "
                "was not found under it."
            )
        return local_path

    # Common mistake: pointing TCRMETA_HF_REPO at a local filesystem path
    # (that's what TCRMETA_WEIGHTS_DIR is for) instead of a Hub repo id
    # ("namespace/repo_name"). Catch it here with a clear message rather
    # than letting huggingface_hub's HFValidationError surface instead.
    looks_like_local_path = repo_id.startswith(("/", "~", ".")) or os.path.exists(repo_id)
    if looks_like_local_path:
        raise ValueError(
            f"repo_id resolved to '{repo_id}', which looks like a local filesystem "
            "path, not a Hugging Face Hub repo id ('namespace/repo_name'). If you "
            "meant to load weights from a local directory, set the "
            "TCRMETA_WEIGHTS_DIR environment variable instead of TCRMETA_HF_REPO:\n"
            f'  os.environ["TCRMETA_WEIGHTS_DIR"] = "{repo_id}"\n'
            "TCRMETA_HF_REPO is only for overriding which Hub repo to download from."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to auto-download TCRmeta model weights. "
            "Install it with `pip install huggingface_hub`, or set TCRMETA_WEIGHTS_DIR "
            "to a local directory containing the weights."
        ) from e

    path = hf_hub_download(
        repo_id=repo_id,
        filename=relative_path,
        cache_dir=str(get_cache_dir()),
        token=os.environ.get("TCRMETA_HF_TOKEN"),  # None -> huggingface_hub
        # falls back to the cached `huggingface-cli login` token, if any.
    )
    return Path(path)


def resolve_base_checkpoint() -> Path:
    return resolve_artifact(BASE_CKPT_FILE)


def resolve_rotations() -> Path:
    return resolve_artifact(ROTATIONS_FILE)


def resolve_vgene_table() -> Path:
    return resolve_artifact(VGENE_CDR_FILE)


def resolve_projection_heads() -> list[Path]:
    return [resolve_artifact(f) for f in PROJ_HEAD_FILES]


def resolve_default_reference() -> Path:
    return resolve_artifact(DEFAULT_REFERENCE_FILE)
