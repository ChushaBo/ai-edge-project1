"""Tests for src/08c_segment_breakdown.py's pure helpers."""

import importlib.util

import pandas as pd

from utils.config import REPO_ROOT


def _script_08c():
    spec = importlib.util.spec_from_file_location("segment_breakdown", REPO_ROOT / "src" / "08c_segment_breakdown.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_occupancy_label_splits_condo_from_other_residential():
    m = _script_08c()
    codes = pd.Series([1, 2, 3, 4, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    out = m.occupancy_label(codes)
    assert out.tolist() == [
        "single_family", "two_to_four_family", "other_residential", "commercial", "commercial",
        "single_family", "two_to_four_family", "other_residential", "single_family", "condo_or_RCBAP",
        "single_family", "commercial", "commercial", "commercial",
    ]


def test_occupancy_label_flags_unmapped_codes():
    m = _script_08c()
    out = m.occupancy_label(pd.Series([1, 99, None]))
    assert out.tolist() == ["single_family", "unmapped", pd.NA]


def test_flood_zone_combined_groups_x_and_d_as_other():
    m = _script_08c()
    out = m.flood_zone_combined(pd.Series(["V", "A", "AH_AO", "X", "D", "other", None]))
    assert out.tolist() == ["V", "A", "AH/AO", "X/other", "X/other", "X/other", "X/other"]
