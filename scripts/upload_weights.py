#!/usr/bin/env python3
"""One-time script to upload TCRmeta's model weights + default reference
map to a Hugging Face Hub repo, in the exact folder layout _weights.py
expects.

Steps before running this:
  1. pip install huggingface_hub
  2. Create a free account at https://huggingface.co/join (if you don't
     have one).
  3. Create an access token with WRITE permission:
     https://huggingface.co/settings/tokens -> "New token" -> role "Write".
  4. Log in from the terminal:  huggingface-cli login
     (paste the token when prompted). This caches the token locally so
     neither this script nor the package needs to hardcode it.

Then: fill in REPO_ID and the LOCAL_FILES paths below and run:
    python scripts/upload_weights.py
"""
from huggingface_hub import HfApi, create_repo

# ---------------------------------------------------------------------------
# 1. Fill this in: "<your-username>/<repo-name>", e.g. "mingyaopan/tcrmeta-weights"
REPO_ID = "CHANGE_ME/tcrmeta-weights"

# Set to True while the manuscript is under review, flip to False (or just
# change visibility in the Hub UI later) once you're ready to share publicly.
PRIVATE = True

# ---------------------------------------------------------------------------
# 2. Fill in the local paths to your actual files. Keys are the exact
# relative paths TCRmeta's _weights.py looks for inside the Hub repo —
# don't change the keys, only the values (your local file locations).
LOCAL_FILES = {
    "base_encoder/best_model.pth": "/path/to/your/best_model.pth",
    "finetune/rotations_gpa.pkl": "/path/to/your/rotations_gpa.pkl",
    "reference_data/vgene_cdr.csv": "/path/to/your/vgene_cdr1225.csv",
    "reference_maps/young_reference.pkl": "/path/to/your/vgene_density_models.pkl",
    # The 7 projection-head checkpoints, IN THE SAME ORDER the rotation
    # matrices (rotations_gpa.pkl's R_list) were fit against. Getting this
    # order wrong will silently misalign the ensemble.
    "finetune/proj_head_0.pt": "/path/to/fold_7/best_bce_base.pt",
    "finetune/proj_head_1.pt": "/path/to/fold_8/best_bce_base.pt",
    "finetune/proj_head_2.pt": "/path/to/fold_5/best_bce_base.pt",
    "finetune/proj_head_3.pt": "/path/to/fold_1/best_purity10_base.pt",
    "finetune/proj_head_4.pt": "/path/to/fold_2/best_purity10_base.pt",
    "finetune/proj_head_5.pt": "/path/to/fold_3/best_purity10_base.pt",
    "finetune/proj_head_6.pt": "/path/to/fold_8/best_purity10_base.pt",
}


def main():
    api = HfApi()
    create_repo(REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)

    for repo_path, local_path in LOCAL_FILES.items():
        print(f"Uploading {local_path} -> {REPO_ID}/{repo_path} ...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model",
        )
    print("Done. Now set:")
    print(f"    export TCRMETA_HF_REPO={REPO_ID}")
    print("or update DEFAULT_HF_REPO in src/tcrmeta/_weights.py to this value.")


if __name__ == "__main__":
    main()
