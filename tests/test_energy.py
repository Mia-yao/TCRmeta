import numpy as np
import pandas as pd


def test_energy_distance_zero_for_identical_point_clouds(tcrmeta_energy):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    w = np.ones(50) / 50
    ed = tcrmeta_energy._weighted_energy_distance_numpy(X, w, X.copy(), w.copy())
    assert abs(ed) < 1e-8


def test_energy_distance_near_zero_for_same_distribution_different_samples(tcrmeta_energy):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 5))
    Y = rng.normal(size=(500, 5))
    w = np.ones(500) / 500
    ed = tcrmeta_energy._weighted_energy_distance_numpy(X, w, Y, w)
    # Two independent samples from the *same* distribution should give an
    # energy distance close to 0 (sampling noise only), not systematically
    # large in either direction.
    assert abs(ed) < 0.5


def test_energy_distance_large_for_separated_distributions(tcrmeta_energy):
    rng = np.random.default_rng(2)
    X = rng.normal(loc=0.0, size=(300, 5))
    Y = rng.normal(loc=10.0, size=(300, 5))
    w = np.ones(300) / 300
    ed = tcrmeta_energy._weighted_energy_distance_numpy(X, w, Y, w)
    assert ed > 5.0  # clearly separated clouds -> large positive energy distance


def test_dispatch_picks_numpy_path_on_cpu_device_string(tcrmeta_energy):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 3))
    w = np.ones(20) / 20
    # Passing a plain "cpu" string (not a torch.device) must not require
    # torch to be installed/imported.
    ed = tcrmeta_energy._pairwise_weighted_energy_distance(X, w, X.copy(), w.copy(), "cpu")
    assert abs(ed) < 1e-8


def test_weighted_summary_matches_manual_calculation(tcrmeta_energy):
    # v_weight = (depth_1 + depth_2) / 2 — the RAW post-downsample read
    # depth assigned to that V gene in each repertoire (NOT normalized
    # by repertoire total, NOT clone count) — matches the reference
    # collapse_one_pair() formula: (energy * v_weight).sum() / v_weight.sum().
    per_gene_df = pd.DataFrame({
        "v_gene": ["A", "B", "C"],
        "n_clones_1": [100, 50, 10],
        "n_clones_2": [100, 150, 10],
        "depth_1": [500.0, 300.0, 50.0],
        "depth_2": [500.0, 300.0, 50.0],
        "energy_distance": [2.0, 8.0, 999.0],
    })
    summary = tcrmeta_energy._weighted_summary(per_gene_df)

    weights = np.array([500.0, 300.0, 50.0])  # (depth_1+depth_2)/2 per row
    energies = np.array([2.0, 8.0, 999.0])
    expected_energy = float((energies * weights).sum() / weights.sum())

    assert summary.loc[0, "n_v_genes"] == 3
    assert np.isclose(summary.loc[0, "energy_weighted"], expected_energy)


def test_weighted_summary_drops_nan_energy_rows(tcrmeta_energy):
    per_gene_df = pd.DataFrame({
        "v_gene": ["A", "B"],
        "n_clones_1": [100, 100],
        "n_clones_2": [100, 100],
        "depth_1": [400.0, 600.0],
        "depth_2": [400.0, 600.0],
        "energy_distance": [np.nan, 3.0],
    })
    summary = tcrmeta_energy._weighted_summary(per_gene_df)
    assert summary.loc[0, "n_v_genes"] == 1
    assert np.isclose(summary.loc[0, "energy_weighted"], 3.0)


def test_weighted_summary_drops_zero_weight_rows(tcrmeta_energy):
    per_gene_df = pd.DataFrame({
        "v_gene": ["A", "B"],
        "n_clones_1": [0, 100],
        "n_clones_2": [0, 100],
        "depth_1": [0.0, 400.0],
        "depth_2": [0.0, 400.0],
        "energy_distance": [5.0, 3.0],
    })
    summary = tcrmeta_energy._weighted_summary(per_gene_df)
    assert summary.loc[0, "n_v_genes"] == 1
    assert np.isclose(summary.loc[0, "energy_weighted"], 3.0)


def test_weighted_summary_empty_input(tcrmeta_energy):
    summary = tcrmeta_energy._weighted_summary(
        pd.DataFrame(columns=["n_clones_1", "n_clones_2", "depth_1", "depth_2", "energy_distance"])
    )
    assert summary.loc[0, "n_v_genes"] == 0
    assert np.isnan(summary.loc[0, "energy_weighted"])
