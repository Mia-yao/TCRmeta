import numpy as np
import pandas as pd
import pytest


def test_downsample_multinomial_no_op_below_target(tcrmeta_utils):
    counts = np.array([10, 20, 30])
    rng = np.random.default_rng(0)
    out = tcrmeta_utils.downsample_multinomial(counts, target=1000, rng=rng)
    assert np.array_equal(out, counts)  # total (60) <= target -> unchanged


def test_downsample_multinomial_reduces_to_target(tcrmeta_utils):
    counts = np.array([1000, 2000, 3000, 4000])
    rng = np.random.default_rng(0)
    out = tcrmeta_utils.downsample_multinomial(counts, target=1000, rng=rng)
    assert out.sum() == 1000
    assert len(out) == len(counts)
    # Roughly proportional: the largest bucket should still get the most reads.
    assert out.argmax() == counts.argmax()


def test_downsample_multinomial_reproducible_with_same_rng_state(tcrmeta_utils):
    counts = np.array([500, 1500, 3000])
    out1 = tcrmeta_utils.downsample_multinomial(counts, target=1000, rng=np.random.default_rng(42))
    out2 = tcrmeta_utils.downsample_multinomial(counts, target=1000, rng=np.random.default_rng(42))
    assert np.array_equal(out1, out2)


def test_normalize_weights(tcrmeta_utils):
    w = tcrmeta_utils.normalize_weights(np.array([1, 1, 2]))
    assert w is not None
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, [0.25, 0.25, 0.5])


def test_normalize_weights_empty_returns_none(tcrmeta_utils):
    assert tcrmeta_utils.normalize_weights(np.array([0, 0])) is None


def test_validate_repertoire_columns_requires_exact_names(tcrmeta_utils):
    df = pd.DataFrame({"aminoAcid": ["CASSX"], "vGeneName": ["TRBV6-4"], "count": [5]})
    with pytest.raises(ValueError, match="missing required column"):
        tcrmeta_utils.validate_repertoire_columns(df)


def test_validate_repertoire_columns_normalizes_cdr3_and_vgene(tcrmeta_utils):
    df = pd.DataFrame({
        "cdr3aa": ["ASSXX", "CASSYYF", "ASSZZ"],  # first/third missing C/F flanks
        "v_gene": ["TRBV6-4*01", "TRBV6-1", "TRBV6-4*02"],
        "count": [5, 0, 3],  # second row should be dropped (count <= 0)
    })
    out = tcrmeta_utils.validate_repertoire_columns(df)
    assert len(out) == 2
    assert out["cdr3aa"].tolist() == ["CASSXXF", "CASSZZF"]
    assert out["v_gene"].tolist() == ["TRBV6-4", "TRBV6-4"]
    assert (out["count"] > 0).all()


def test_aggregate_duplicate_clones_sums_counts(tcrmeta_utils):
    df = pd.DataFrame({
        "cdr3aa": ["CASSAAF", "CASSAAF", "CASSBBF"],
        "v_gene": ["TRBV6-4", "TRBV6-4", "TRBV6-1"],
        "count": [10, 5, 7],
    })
    out = tcrmeta_utils.aggregate_duplicate_clones(df)
    out = out.set_index(["cdr3aa", "v_gene"])
    assert out.loc[("CASSAAF", "TRBV6-4"), "count"] == 15
    assert out.loc[("CASSBBF", "TRBV6-1"), "count"] == 7
