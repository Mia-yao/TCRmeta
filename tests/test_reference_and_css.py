"""Exercise build_reference()'s per-V-gene fitting (_build_one_vgene) and
compute_css()'s scoring (_score_vgene / _pool_whole_repertoire) against
synthetic embeddings, bypassing the actual ESM2/ResMLP model (which
needs torch/transformers + real weights, unavailable in this sandbox).

These synthetic-embedding tests verify the *statistical logic* is
correct: a query repertoire drawn from the same distribution as the
reference should score as "unshifted" (CSS near 1, balanced POS/LZO),
while a query repertoire drawn from a clearly separated distribution
should score as strongly shifted (CSS >> 1, dominated by low-density
tail membership). This is the core claim the CSS metric is supposed to
support, so it's the most important thing to check.
"""
import numpy as np
import pytest

VGENE = "TRBV_TEST"
DIM = 10


def _make_samples(n_samples=25, n_clones_per_sample=200, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_samples):
        emb = rng.normal(loc=0.0, scale=1.0, size=(n_clones_per_sample, dim)).astype(np.float32)
        ds = rng.integers(1, 20, size=n_clones_per_sample).astype(np.int64)
        vgene = np.array([VGENE] * n_clones_per_sample)
        samples.append({"emb": emb, "ds": ds, "vgene": vgene})
    return samples


@pytest.fixture(scope="module")
def fitted_model(tcrmeta_reference):
    samples = _make_samples()
    model = tcrmeta_reference._build_one_vgene(
        VGENE, samples,
        pca_dim=5, max_umap=500, umap_neighbors=15, umap_min_dist=0.1,
        min_samples=20, min_clones=50, seed=42,
    )
    assert model is not None
    return model


def test_build_one_vgene_returns_expected_keys(fitted_model):
    for key in ("scaler", "pca", "kde_model", "ref_logdens", "umap_reducer",
                "ref_umap2d", "ref_ds_weights", "ref_centroid_2d",
                "thr_5", "thr_15", "thr_85", "thr_95", "n_clones", "n_samples", "bw", "pca_dim"):
        assert key in fitted_model
    assert fitted_model["n_samples"] == 25
    assert fitted_model["thr_5"] < fitted_model["thr_85"] < fitted_model["thr_95"]


def test_build_one_vgene_respects_min_samples(tcrmeta_reference):
    samples = _make_samples(n_samples=5)  # below default min_samples=20
    model = tcrmeta_reference._build_one_vgene(
        VGENE, samples, pca_dim=5, max_umap=500, umap_neighbors=15, umap_min_dist=0.1,
        min_samples=20, min_clones=50, seed=42,
    )
    assert model is None


def test_select_vgenes_explicit_list_bypasses_frequency_filter(tcrmeta_reference):
    samples = _make_samples(n_samples=3, n_clones_per_sample=5)
    selected = tcrmeta_reference._select_vgenes(samples, v_genes=["ANYTHING"], freq_thr=0.5)
    assert selected == ["ANYTHING"]


def test_select_vgenes_frequency_threshold(tcrmeta_reference):
    rng = np.random.default_rng(0)
    samples = [{
        "vgene": np.array(["A"] * 90 + ["B"] * 10),
        "ds": np.ones(100, dtype=np.int64),
        "emb": rng.normal(size=(100, 2)),
    }]
    # "A" is 90% of pooled counts, "B" is 10%.
    assert tcrmeta_reference._select_vgenes(samples, None, freq_thr=0.5) == ["A"]
    assert set(tcrmeta_reference._select_vgenes(samples, None, freq_thr=0.05)) == {"A", "B"}


def test_css_matched_distribution_scores_near_balanced(tcrmeta_css, fitted_model):
    rng = np.random.default_rng(123)
    emb_query = rng.normal(loc=0.0, scale=1.0, size=(300, DIM)).astype(np.float32)
    w_query = rng.integers(1, 20, size=300).astype(np.float64)
    scored = tcrmeta_css._score_vgene(emb_query, w_query, fitted_model, pos_percentile=85, neg_percentile=15)
    # In-distribution query: POS (top 15%) and LZO (bottom 15%) should both
    # land roughly near 15%, so CSS = LZO/POS should be in a modest range
    # around 1, not wildly skewed.
    assert 0.3 < scored["CSS"] < 3.0


def test_css_shifted_distribution_scores_much_higher(tcrmeta_css, fitted_model):
    rng = np.random.default_rng(456)
    # Shifted far from the reference's mean (0) -> should mostly fall in
    # the reference's low-density tail.
    emb_shifted = rng.normal(loc=8.0, scale=1.0, size=(300, DIM)).astype(np.float32)
    w_shifted = rng.integers(1, 20, size=300).astype(np.float64)
    scored_shifted = tcrmeta_css._score_vgene(emb_shifted, w_shifted, fitted_model, pos_percentile=85, neg_percentile=15)

    rng2 = np.random.default_rng(789)
    emb_matched = rng2.normal(loc=0.0, scale=1.0, size=(300, DIM)).astype(np.float32)
    w_matched = rng2.integers(1, 20, size=300).astype(np.float64)
    scored_matched = tcrmeta_css._score_vgene(emb_matched, w_matched, fitted_model, pos_percentile=85, neg_percentile=15)

    assert scored_shifted["LZO"] > 0.8  # nearly all clones in the low-density tail
    assert scored_shifted["POS"] < 0.05  # almost none in the high-density core
    assert scored_shifted["CSS"] > scored_matched["CSS"] * 5


def test_pool_whole_repertoire_matches_manual_weighted_average(tcrmeta_css):
    # Two "V genes" with known log-densities/weights/thresholds; verify the
    # pooled MLD/POS/LZO/CSS match a hand-computed weighted average.
    ld_a = np.array([1.0, 2.0, 3.0, 4.0])
    ld_b = np.array([5.0, 6.0])
    w_a = np.array([1.0, 1.0, 1.0, 1.0])
    w_b = np.array([1.0, 1.0])
    rows = [
        {"_ld": ld_a, "_w": w_a, "_thr_pos": 3.5, "_thr_neg": 1.5, "n_clones": 4},
        {"_ld": ld_b, "_w": w_b, "_thr_pos": 3.5, "_thr_neg": 1.5, "n_clones": 2},
    ]
    pooled = tcrmeta_css._pool_whole_repertoire(rows)

    ld_cat = np.concatenate([ld_a, ld_b])
    w_cat = np.concatenate([w_a, w_b])
    wn = w_cat / w_cat.sum()
    expected_mld = float(np.average(ld_cat, weights=wn))
    expected_pos = float(np.average((ld_cat >= 3.5).astype(float), weights=wn))
    expected_lzo = float(np.average((ld_cat <= 1.5).astype(float), weights=wn))

    assert pooled["n_clones"] == 6
    assert pooled["n_v_genes"] == 2
    assert np.isclose(pooled["MLD"], expected_mld)
    assert np.isclose(pooled["POS"], expected_pos)
    assert np.isclose(pooled["LZO"], expected_lzo)


def test_pool_whole_repertoire_empty_input(tcrmeta_css):
    pooled = tcrmeta_css._pool_whole_repertoire([])
    assert pooled["n_v_genes"] == 0
    assert np.isnan(pooled["CSS"])


def test_score_vgene_raises_clear_error_on_embedding_dim_mismatch(tcrmeta_css, fitted_model):
    # fitted_model was fit on DIM=10 embeddings (e.g. embedding_type="final").
    # Simulate calling compute_css with embedding_type="pretrained" (a
    # different, mismatched dimensionality) against that same reference.
    rng = np.random.default_rng(0)
    wrong_dim = DIM + 5
    emb_wrong = rng.normal(size=(50, wrong_dim)).astype(np.float32)
    w = np.ones(50)
    with pytest.raises(ValueError, match="[Ee]mbedding dimension mismatch"):
        tcrmeta_css._score_vgene(emb_wrong, w, fitted_model, pos_percentile=85, neg_percentile=15)
