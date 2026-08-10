import numpy as np


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
