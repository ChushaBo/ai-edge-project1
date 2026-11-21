"""Tests for src/09_frequency.py's credibility-blending helpers (RUNBOOK 6.1)."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from utils.config import REPO_ROOT


def _script_09():
    spec = importlib.util.spec_from_file_location("frequency", REPO_ROOT / "src" / "09_frequency.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_county_from_geoid_takes_first_five_digits():
    m = _script_09()
    out = m.county_from_geoid(pd.Series(["120860059013", "120350601063"]))
    assert out.tolist() == ["12086", "12035"]


def test_credibility_blend_full_weight_on_large_n():
    m = _script_09()
    n, raw, prior = pd.Series([1000.0]), pd.Series([0.05]), pd.Series([0.01])
    blended, z = m.credibility_blend(n, raw, prior, k=10.0)
    assert z.iloc[0] > 0.99
    assert blended.iloc[0] == pytest.approx(0.05, abs=0.01)


def test_credibility_blend_falls_back_to_prior_when_no_claims():
    m = _script_09()
    n, raw, prior = pd.Series([0.0]), pd.Series([np.nan]), pd.Series([0.02])  # zero exposure: undefined raw rate
    blended, z = m.credibility_blend(n, raw, prior, k=50.0)
    assert z.iloc[0] == 0.0
    assert blended.iloc[0] == 0.02


def test_credibility_blend_partial_weight_matches_formula():
    m = _script_09()
    n, raw, prior, k = pd.Series([30.0]), pd.Series([0.1]), pd.Series([0.02]), 30.0
    blended, z = m.credibility_blend(n, raw, prior, k)
    expected_z = 30.0 / (30.0 + k)
    assert z.iloc[0] == pytest.approx(expected_z)
    assert blended.iloc[0] == pytest.approx(expected_z * 0.1 + (1 - expected_z) * 0.02)
