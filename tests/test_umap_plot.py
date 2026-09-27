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


def test_plot_one_vgene_raises_clear_error_on_embedding_dim_mismatch(tcrmeta_umap_plot, fitted_model):
    rng = np.random.default_rng(0)
    wrong_dim = DIM + 5
    emb_wrong = rng.normal(size=(50, wrong_dim)).astype(np.float32)
    w = np.ones(50)
    with pytest.raises(ValueError, match="[Ee]mbedding dimension mismatch"):
        tcrmeta_umap_plot._plot_one_vgene(VGENE, emb_wrong, w, fitted_model, "#1565C0", 50, (5, 5))


def test_compute_kde_grid_returns_none_for_too_few_points(tcrmeta_umap_plot):
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    xy = np.random.default_rng(0).normal(size=(3, 2))  # < 10 points
    w = np.ones(3)
    assert tcrmeta_umap_plot._compute_kde_grid(xy, w, xx, yy) is None


def test_compute_kde_grid_shape_and_peak_near_data_center(tcrmeta_umap_plot):
    rng = np.random.default_rng(0)
    xy = rng.normal(loc=(2.0, -1.0), scale=0.3, size=(200, 2))
    w = np.ones(200)
    xi = np.linspace(-4, 6, 60)
    yi = np.linspace(-5, 3, 60)
    xx, yy = np.meshgrid(xi, yi)
    dens = tcrmeta_umap_plot._compute_kde_grid(xy, w, xx, yy)
    assert dens is not None
    assert dens.shape == xx.shape
    peak_idx = np.unravel_index(np.argmax(dens), dens.shape)
    peak_x, peak_y = xx[peak_idx], yy[peak_idx]
    assert abs(peak_x - 2.0) < 1.0
    assert abs(peak_y - (-1.0)) < 1.0
