"""Tests for 06_embed_text: masking, chunking, claim vectors, and PCA fit on train only."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from utils.config import REPO_ROOT, get_path, load_config
from utils.text import chunk_words, mask_damage_numbers


def _script_06():
    spec = importlib.util.spec_from_file_location("embed_text", REPO_ROOT / "src" / "06_embed_text.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("text, expected", [
    ("Damage was estimated at $1.2 million.", "Damage was estimated at [AMOUNT]."),
    ("Property damage of 10.00K.", "Property damage of [AMOUNT]."),
    ("Losses near 3 million dollars.", "Losses near [AMOUNT]."),
    ("Over 1,200 homes and 30 mobile homes flooded.", "Over [NUM] homes and [NUM] mobile homes flooded."),
    ("The surge inundated about 300 structures.", "The surge inundated about [NUM] structures."),
    ("The storm damaged 45 businesses.", "The storm damaged [NUM] businesses."),
])
def test_mask_replaces_damage_figures(text, expected):
    assert mask_damage_numbers(text) == expected


def test_mask_keeps_storm_measurements():
    text = "Rainfall of 11.25 inches, gusts to 80 mph, 6 feet of surge on US 19 on September 26."
    assert mask_damage_numbers(text) == text


def test_chunks_never_exceed_limit_and_keep_all_words():
    text = "Short one. " + " ".join(["word"] * 400) + ". Last sentence here."
    chunks = chunk_words(text, 150)
    assert max(len(c.split()) for c in chunks) <= 150
    assert sum(len(c.split()) for c in chunks) == len(text.split())


def test_claim_vector_is_mean_of_pieces_and_zero_without_text():
    script = _script_06()
    vectors = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    X = script.claim_matrix(pd.Series([("a", "b"), ("a",), ()]), vectors, 2)
    np.testing.assert_allclose(X, [[0.5, 0.5], [1.0, 0.0], [0.0, 0.0]])


def test_pca_is_fit_on_train_rows_only():
    script = _script_06()
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, size=(200, 5))
    test = rng.normal(100, 1, size=(50, 5))  # far away: would move the PCA mean if used
    X = np.vstack([train, test])
    mask = np.r_[np.ones(200, bool), np.zeros(50, bool)]
    pca = script.fit_pca_train_only(X, mask, 3, seed=0)
    np.testing.assert_allclose(pca.mean_, train.mean(axis=0))
    assert pca.n_samples_ == 200


def test_claims_without_text_get_zero_components():
    script = _script_06()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    pca = script.fit_pca_train_only(X, np.ones(20, bool), 2, seed=0)
    has_text = np.r_[np.ones(15, bool), np.zeros(5, bool)]
    Z = script.pca_features(pca, X, has_text, "txt")
    assert (Z.iloc[15:] == 0).all().all() and list(Z.columns) == ["txt_pc01", "txt_pc02"]


def test_saved_pca_was_fit_on_train_claims_only():
    cfg = load_config()
    models = get_path(cfg, "models", create=False)
    claims_path = get_path(cfg, "interim", create=False) / "claims_with_text.parquet"
    if not (models / "pca_text.joblib").exists() or not claims_path.exists():
        pytest.skip("PCA not built (run src/06_embed_text.py)")
    import joblib
    claims = pd.read_parquet(claims_path, columns=["id", "split", "has_narrative", "quarantined"])
    if cfg["quarantine"]["enabled"]:
        claims = claims[~claims["quarantined"]]
    train_ids = set(claims.loc[(claims["split"] == "train") & claims["has_narrative"], "id"])
    for name in ("pca_text.joblib", "pca_text_masked.joblib"):
        saved = joblib.load(models / name)
        assert set(saved["fit_ids"]) == train_ids
        assert saved["pca"].n_samples_ == len(train_ids)
