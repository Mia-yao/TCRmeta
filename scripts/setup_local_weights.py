#!/usr/bin/env python3
"""One-time script to lay out your existing isilon checkpoint files into
the local directory structure TCRmeta expects (for TCRMETA_WEIGHTS_DIR),
so you can test the package without first uploading anything to the
Hugging Face Hub.

Uses symlinks (not copies) so it's instant and doesn't duplicate large
checkpoint files. If your filesystem doesn't support symlinks across the
paths involved, set SYMLINK = False to fall back to copying.

Run once:
    python scripts/setup_local_weights.py
Then in your notebook/script:
    os.environ["TCRMETA_WEIGHTS_DIR"] = "/mnt/isilon/.../tcrmeta_weights"
    (i.e. whatever you set WEIGHTS_DIR to below)
"""
import os
import shutil

SYMLINK = True  # set False to copy instead of symlink

# ---------------------------------------------------------------------------
# Where to build the local weights directory. Point TCRMETA_WEIGHTS_DIR at
# this same path afterward.
WEIGHTS_DIR = "/mnt/isilon/boli_lab/globus/project/mingyao/DeepCluster/tcrmeta_weights"

# ---------------------------------------------------------------------------
# Your existing files (as given).
VGENE_CDR = "/mnt/isilon/boli_lab/globus/project/mingyao/HLA_TCR_cross_reactivity/single_tcr_mhc_prediction/vgene_cdr1225.csv"
BASE_CKPT = "/mnt/isilon/boli_lab/globus/project/mingyao/DeepCluster/model_checkpoint/thers10_Markov_35M_twoside_perTCR/best_model.pth"
ROT_PATH = "/mnt/isilon/boli_lab/globus/project/mingyao/DeepCluster/data/fine_tune/rotations_gpa.pkl"
PROJ_PATHS = [
    "/mnt/isilon/boli_lab/globus/project/mingyao/DeepCluster/model_checkpoint/fisher_label_finetune_tune_10times_64dim/fold_7/best_bce_base.pt",
    "/mnt/isilon/boli_lab/globus/project/mingyao/DeepCluster/model_checkpoint/fisher_label_finetune_tune_10times_64dim/fold_8/best_bce_base.pt",
    "/mnt/isilon/boli_lab/globus/project/mingyao/DeepCluster/model_checkpoint/fisher_label_finetune_tune_10times_64dim/fold_5/best_bce_base.pt",
    "/mnt/isilon/boli_lab/globus/project/mingyao/DeepCluster/model_checkpoint/fisher_label_finetune_tune_10times_64dim/fold_1/best_purity10_base.pt",
    "/mnt/isilon/boli_lab/globus/project/mingyao/DeepCluster/model_checkpoint/fisher_label_finetune_tune_10times_64dim/fold_2/best_purity10_base.pt",
    "/mnt/isilon/boli_lab/globus/project/mingyao/DeepCluster/model_checkpoint/fisher_label_finetune_tune_10times_64dim/fold_3/best_purity10_base.pt",
    "/mnt/isilon/boli_lab/globus/project/mingyao/DeepCluster/model_checkpoint/fisher_label_finetune_tune_10times_64dim/fold_8/best_purity10_base.pt",
]
# IMPORTANT: this order must match the order rotations_gpa.pkl's R_list was
# fit in. If you're not sure, use whatever order you originally passed
# PROJ_PATHS in when generating rotations_gpa.pkl.

# Optional: your build_density_maps.py output. Leave as None if you don't
# have one yet / want to build it fresh with tm.build_reference() instead.
# This was referenced earlier as YOUNG_MODEL_PATH in lc_density_scoring.py —
# double check this is still the right file before relying on it.
DEFAULT_REFERENCE = (
    "/mnt/isilon/boli_lab/globus/project/mingyao/DeepCluster/distribution_shift/"
    "Tonon/reference_map/density_map_output/density_map_output_pca30_10000_thres/"
    "vgene_density_models.pkl"
)

# ---------------------------------------------------------------------------
# Target relative paths (must match src/tcrmeta/_weights.py exactly).
TARGETS = {
    "base_encoder/best_model.pth": BASE_CKPT,
    "finetune/rotations_gpa.pkl": ROT_PATH,
    "reference_data/vgene_cdr.csv": VGENE_CDR,
}
for i, p in enumerate(PROJ_PATHS):
    TARGETS[f"finetune/proj_head_{i}.pt"] = p
if DEFAULT_REFERENCE and os.path.exists(DEFAULT_REFERENCE):
    TARGETS["reference_maps/young_reference.pkl"] = DEFAULT_REFERENCE


def main():
    for rel_path, source in TARGETS.items():
        if not os.path.exists(source):
            print(f"SKIP (source not found): {source}")
            continue
        dest = os.path.join(WEIGHTS_DIR, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) or os.path.islink(dest):
            os.remove(dest)
        if SYMLINK:
            os.symlink(source, dest)
        else:
            shutil.copy2(source, dest)
        print(f"{'linked' if SYMLINK else 'copied'}: {dest} -> {source}")

    print(f"\nDone. Set:\n    os.environ['TCRMETA_WEIGHTS_DIR'] = {WEIGHTS_DIR!r}")
    if "reference_maps/young_reference.pkl" not in TARGETS:
        print(
            "\nNote: no default reference map was linked (DEFAULT_REFERENCE not "
            "found/set). compute_css()/plot_umap() will need reference=... passed "
            "explicitly, or build one fresh with tm.build_reference()."
        )


if __name__ == "__main__":
    main()
