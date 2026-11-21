"""Diagnostic (exploration only, not part of run_all.sh or outputs/): does coarsening
M1's location features fix the catastrophe-year portfolio-total failure seen on the
real test set, without looking at the test set itself?

Context: src/07_train_models.py's M1 underpredicts the test_full portfolio total by
57.6% (docs/decisions.md, "M1 badly underpredicts the portfolio total on test_full"),
apparently because latitude/longitude let LightGBM carve spatial leaves fine enough to
memorize each location's 2010-2019 average loss ratio, with no way to extrapolate when
that location is hit far harder by a later, more severe storm.

This script tests that hypothesis using ONLY training-period (2010-2019) data, so it
never touches validation or test (CLAUDE.md rule 5): hold out Hurricane Irma (2017) and
Hurricane Michael (2018) claims from train, fit three models on the remaining training
rows, and score each on the two held-out storms:
  - M1        : M1_FEATURES as-is (latitude, longitude included).
  - M1_county : M1_FEATURES with latitude/longitude replaced by countyCode (categorical,
                a coarser, discrete stand-in for location -- one of the two options
                asked for; county chosen over an arbitrary lat/long grid size).
  - B1        : the flood-zone-mean baseline, for reference.
All three reuse M1's already-tuned LightGBM hyperparameters (data/models/m1_lgbm.joblib)
and its L2 objective (docs/decisions.md "M1/M2 objective fixed to L2") -- only the
feature set changes between M1 and M1_county, so any difference isolates the effect of
location granularity.

Output: a printed table of claim-level MAE ($) and portfolio total percent error per
(storm, model). Rerun directly:
    python notebooks/diag_leave_one_storm_out.py
"""

import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.config import get_path, load_config, require  # noqa: E402
from utils.models import M1_FEATURES, dollar_metrics, fit_categories  # noqa: E402


def _load_07():
    """Reuse 07_train_models.py's training/prediction helpers (final_fit, fit_b1,
    predict_b1, predict_clipped) without duplicating them."""
    spec = importlib.util.spec_from_file_location("train_models", REPO_ROOT / "src" / "07_train_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Held-out storm windows (date of loss), the same Irma/Michael windows used in the
# human review of claim maturity (docs/decisions.md).
STORMS = {
    "Irma_2017": ("2017-09-06", "2017-09-15"),
    "Michael_2018": ("2018-10-10", "2018-10-14"),
}


def main() -> None:
    cfg = load_config()
    seed = int(require(cfg, "random_seed"))
    date_col = require(cfg, "fields.claims_date_of_loss")
    cov_col = require(cfg, "fields.claims_coverage")
    county_col = require(cfg, "fields.claims_county")
    cap = float(require(cfg, "target.cap_share_at"))
    zone_col = "flood_zone_group"

    m7 = _load_07()
    fixed_params, objective = cfg["models"]["fixed_params"], cfg["models"]["objective"]
    best_m1 = joblib.load(get_path(cfg, "models") / "m1_lgbm.joblib")["best_params"]

    cols = sorted(set(["id", "split", "target", cov_col, date_col, zone_col, county_col] + M1_FEATURES))
    claims = pd.read_parquet(get_path(cfg, "interim") / "claims_with_text.parquet", columns=cols)
    train = claims[claims["split"] == "train"].reset_index(drop=True)  # 2010-2019 only; no validation, no test

    storm_mask = {name: train[date_col].between(pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC"))
                 for name, (lo, hi) in STORMS.items()}
    in_any_storm = pd.concat(storm_mask.values(), axis=1).any(axis=1)
    train_minus = train[~in_any_storm].reset_index(drop=True)
    print(f"train: {len(train):,} rows; holding out "
         f"{', '.join(f'{k}={int(v.sum()):,}' for k, v in storm_mask.items())} "
         f"({int(in_any_storm.sum()):,} total); fitting on {len(train_minus):,} rows\n")

    county_features = [c for c in M1_FEATURES if c not in ("latitude", "longitude")] + [county_col]

    cats_m1 = fit_categories(train_minus, M1_FEATURES)
    cats_county = fit_categories(train_minus, county_features)
    cats_county[county_col] = sorted(train_minus[county_col].astype("string").dropna().unique().tolist())

    model_m1 = m7.final_fit(train_minus, M1_FEATURES, cats_m1, seed, best_m1,
                            fixed_params, objective, len(train_minus))
    model_county = m7.final_fit(train_minus, county_features, cats_county, seed, best_m1,
                                fixed_params, objective, len(train_minus))
    b1_means, b1_overall = m7.fit_b1(train_minus, zone_col)

    rows = []
    for storm_name, mask in storm_mask.items():
        held = train[mask].reset_index(drop=True)
        preds = {
            "M1": m7.predict_clipped(model_m1, held, M1_FEATURES, cats_m1, cap),
            "M1_county": m7.predict_clipped(model_county, held, county_features, cats_county, cap),
            "B1": np.clip(m7.predict_b1(held, zone_col, b1_means, b1_overall), 0.0, cap),
        }
        for model_name, pred in preds.items():
            m = dollar_metrics(held["target"].to_numpy(), pred, held[cov_col].to_numpy())
            rows.append({"storm": storm_name, "model": model_name, "n": m["n"],
                        "mae_dollar": round(m["mae_dollar"], 0), "mae_share": round(m["mae_share"], 4),
                        "actual_total_$M": round(m["actual_total_dollar"] / 1e6, 2),
                        "predicted_total_$M": round(m["predicted_total_dollar"] / 1e6, 2),
                        "pct_error_total": round(m["pct_error_total"], 1)})

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
