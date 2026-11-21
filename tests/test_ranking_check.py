"""Tests for src/08b_ranking_check.py's pure helpers (rank-based deciles, Gini)."""

import importlib.util

import numpy as np
import pytest

from utils.config import REPO_ROOT


def _script_08b():
    spec = importlib.util.spec_from_file_location("ranking_check", REPO_ROOT / "src" / "08b_ranking_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assign_deciles_gives_equal_sized_groups_even_with_few_distinct_values():
    m = _script_08b()
    # Only 2 distinct predicted values (like B1's handful of zone means), 100 rows:
    # rank-based assignment must still produce exactly 10 claims per decile.
    pred = np.array([0.1] * 50 + [0.2] * 50)
    deciles = m.assign_deciles(pred)
    counts = deciles.value_counts()
    assert len(counts) == 10
    assert (counts == 10).all()
    # Lowest predicted value should land in early deciles, highest in late deciles
    assert deciles[:50].max() <= deciles[50:].min()


def test_gini_risk_ranking_zero_for_uninformative_ranking():
    m = _script_08b()
    rng = np.random.RandomState(0)
    actual = rng.uniform(0, 100, size=5000)
    pred = rng.uniform(0, 1, size=5000)  # unrelated to actual
    gini = m.gini_risk_ranking(actual, pred)
    assert abs(gini) < 0.05  # near zero: no separation when ranking is unrelated to actual


def test_gini_risk_ranking_positive_for_perfect_ranking():
    m = _script_08b()
    actual = np.array([1.0, 2.0, 5.0, 10.0, 50.0])
    pred = actual.copy()  # perfect ranking: predicted order == actual order
    gini = m.gini_risk_ranking(actual, pred)
    assert gini > 0.3  # clearly separates low- from high-loss claims


def test_gini_risk_ranking_negative_for_inverted_ranking():
    m = _script_08b()
    actual = np.array([1.0, 2.0, 5.0, 10.0, 50.0])
    pred = -actual  # worst-first: exactly backwards
    gini = m.gini_risk_ranking(actual, pred)
    assert gini < -0.3
