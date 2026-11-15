"""Tests for utils.config."""

import pandas as pd
import pytest

from utils.config import DecisionMissingError, load_config, require, require_date


def test_require_stops_on_undecided_value():
    cfg = {"years": {"policies_year": None}}
    with pytest.raises(DecisionMissingError):
        require(cfg, "years.policies_year")
    with pytest.raises(KeyError):
        require(cfg, "years.nope")


def test_require_date_compares_with_utc_loss_dates():
    cfg = load_config()  # unquoted YAML dates load as datetime.date
    train_end = require_date(cfg, "splits.train_end")
    loss = pd.Series(pd.to_datetime(["2019-12-31", "2020-01-01"], utc=True))

    assert train_end == pd.Timestamp("2019-12-31", tz="UTC")
    assert (loss <= train_end).tolist() == [True, False]  # inclusive boundary, no TypeError
