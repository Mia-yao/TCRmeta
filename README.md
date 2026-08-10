# TCRmeta

Antigen-aware TCR repertoire embeddings and downstream repertoire-level
analysis, built on a pretrained-ESM2 + contrastively fine-tuned encoder.

Given a bulk TCR repertoire, TCRmeta can:

1. **Embed** every clone into a 64-dim antigen-aware embedding space (`embed_repertoire`).
2. **Score** a repertoire's clonal shift score (CSS) against a reference cohort (`compute_css`).
3. **Project** a repertoire onto a reference UMAP map, per V gene (`plot_umap`).
4. **Compare** two repertoires via a frequency-weighted energy-distance shift (`energy_shift`).

Model weights and the default reference map are downloaded once from the
Hugging Face Hub and cached locally (`~/.cache/tcrmeta` by default).

## Install

```bash
pip install tcrmeta
```

GPU acceleration (embedding + energy-distance computation) is used
automatically if a CUDA (or MPS) device is available; pass `device="cpu"`
anywhere to force CPU.

## Input format

Every function takes a TCR repertoire as a `pandas.DataFrame` with
**exactly** these columns (no aliases are auto-detected — rename your own
columns first):

| column   | meaning                                  |
|----------|-------------------------------------------|
| `cdr3aa` | CDR3 amino acid sequence                  |
| `v_gene` | V gene (e.g. `TRBV6-4`, allele suffix ok)  |
| `count`  | clone count / templates / reads           |

Currently supports human TRB (beta chain) repertoires only.

## Quickstart

```python
import pandas as pd
import tcrmeta as tm

df = pd.read_csv("my_repertoire.csv")  # cdr3aa, v_gene, count

# 1. Embed
embeddings = tm.embed_repertoire(df)  # dict {(cdr3aa, v_gene): 64-dim np.ndarray}

# 2. CSS against the shipped default reference
per_gene_css, whole_css = tm.compute_css(df)  # v_gene=None -> scores all reference V genes

# 3. UMAP projection (v_gene required)
fig, coords = tm.plot_umap(df, v_gene="TRBV6-4")
fig.savefig("trbv6-4_umap.png")

# 4. Energy-distance shift between two repertoires
df2 = pd.read_csv("other_repertoire.csv")
per_gene_energy, weighted_summary = tm.energy_shift(df, df2)
```

### Building your own reference map

```python
reference_repertoires = [pd.read_csv(f) for f in my_reference_files]
reference = tm.build_reference(reference_repertoires)
tm.save_reference(reference, "my_reference.pkl")

per_gene_css, whole_css = tm.compute_css(df, reference="my_reference.pkl")
fig, coords = tm.plot_umap(df, reference=reference, v_gene="TRBV6-4")
```

### Keeping intermediate embeddings

```python
tm.embed_repertoire(df, save_path="my_repertoire_embeddings.pkl")
```

Pass the saved path back in via `embeddings=` (loaded with `pickle.load`)
to any downstream function to skip re-embedding, or just delete the file
if you don't need it.

## Configuration

| Env var               | Purpose                                              |
|------------------------|-------------------------------------------------------|
| `TCRMETA_HF_REPO`      | Hugging Face Hub repo id for weights/reference (placeholder until published) |
| `TCRMETA_WEIGHTS_DIR`  | Load weights from a local directory instead of the Hub |
| `TCRMETA_CACHE_DIR`    | Local cache directory (default `~/.cache/tcrmeta`)    |

See `scripts/upload_weights.py` for a one-time script to publish your
model weights and default reference map to the Hugging Face Hub.

## Development

```bash
pip install -e ".[dev]"
pytest
```
