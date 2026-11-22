# Florida Flood Insurance Shortfall

15.071 (The AI Edge) Project 1. We predict the building payout of National Flood
Insurance Program (NFIP) claims in Florida, compare the result against a
flood-zone-only benchmark, and use the gap to estimate a county-level shortfall
between NFIP premiums and expected losses, plus a Hurricane Ian "replay" scenario on
today's policies.

Two tabular+text models are kept strictly separate: **M1** (pre-event information
only) prices the shortfall; **M2** (M1 + embedded storm narratives) estimates
post-event loss and drives the Ian replay.

## Data sources

| Source | Link | Used for |
|---|---|---|
| NFIP Redacted Claims v3 (`NfipClaims`) | `fema.gov/api/open/v3/NfipClaims` | Target + features, 2010-2024, FL |
| NFIP Redacted Policies v3 (`NfipPolicies`) | `fema.gov/api/open/v3/NfipPolicies` | Pricing, 2025 terms, FL |
| NOAA NCEI Storm Events | `ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles` | Storm narratives, matched to claims |
| NWS zone-to-county correlation file | `weather.gov/gis/ZoneCounty` | Maps forecast zones to counties |
| Census cartographic county boundaries | `www2.census.gov/geo/tiger` | Shortfall choropleth |

All sources are public OpenFEMA/NOAA/Census data, used per their open-data terms; this
project is not endorsed by FEMA, NOAA, or the Census Bureau. Saved field dictionaries
are in `docs/data_dictionaries/`.

## Pipeline (`src/`, run in order)

| Scripts | What |
|---|---|
| `01`-`03` | Download claims, policies, storm events + zone/county file |
| `04` | Clean claims, build the target, time-based train/validation/test split |
| `05`-`06` | Match storm narratives to claims (±3 days, same county); embed + PCA |
| `07` | Train B0/B1/M1/M2/M2_masked (LightGBM), tune on validation, evaluate on test once |
| `08`, `08b`, `08c` | SHAP explanation, risk-ranking check, segment breakdown (M1) |
| `09`-`10` | Claim frequency by county/zone; county-level shortfall |
| `11` | Hurricane Ian replay on current policies |

## Key design choices

- **Target** = building payout ÷ building coverage, capped at 1.0 (handles the
  $250k cap and inflation without a CPI adjustment).
- **Time-based split**: train ≤2019 (38,678 claims), validation 2020-2021 (7,657),
  test 2022-2024 (112,573) — chosen over a random split specifically so a storm never
  appears in both train and test, and so the test period (Ian, Helene, Milton, Idalia)
  reflects genuinely future, unseen storms.
- **Leakage**: an explicit allowed-feature list (`config.yaml`); a pytest test fails
  the build if any other column reaches a model. Every transformer (PCA, category
  encodings) is fit on training rows only.
- **Objective**: LightGBM trained with L2 (squared-error) loss only, not L1 — M1's
  predictions feed the shortfall's portfolio *total*, and an L1/median objective
  systematically under-predicts a skewed target's sum even when per-claim error looks
  similar (`docs/decisions.md`).
- 46,971 Helene claims (no NOAA narrative for Tampa Bay counties) are quarantined
  out of M2/M2_masked only; B0/B1/M1 are trained and scored on the full data.

## Results

`outputs/tables/metrics_overall.csv` (test set, evaluated once):

| eval set | model | MAE ($/claim) | portfolio total, % error |
|---|---|---:|---:|
| test_full | B1 (flood-zone benchmark) | $109,767 | −3.8% |
| test_full | **M1** | **$78,067** | **−57.6%** |
| test_matched | M1 | $73,457 | −54.6% |
| test_matched | M2 (M1 + text) | $74,783 | −56.8% |

Shortfall (`outputs/tables/shortfall_totals_comparison.csv`, 1,103,860 FL policies,
$1.52B premium): M1 → $421.7M expected loss (loss ratio 0.28); the flood-zone
benchmark B1 → $769.6M (loss ratio 0.51).

Ian replay (`outputs/tables/ian_replay_headline_comparison.csv`): official M2 predicts
$1.32B against our data's own historical Ian total of $4.32-4.36B (FEMA reported
$4.9B nationwide) — a ~70% miss on the single worst case in the data.

## Key finding: M1 wins on per-claim accuracy, loses on the portfolio total

M1 has the lowest claim-level error of every model, but underpredicts the test
portfolio total far more than the trivial flood-zone benchmark. Test severity (mean
share of coverage 0.37) is roughly double training severity (0.19) — Ian, Helene, and
Milton caused losses with no precedent in 2010-2019 training data, and M1
(a LightGBM tree ensemble) cannot extrapolate past what it saw in training by
construction, a documented property of tree models, not a bug (`docs/course-
checklist.md`, citing the 15.071 lecture material). SHAP confirms latitude/longitude
are M1's most-used features by split count. A follow-up risk-ranking check
(`outputs/tables/ranking_check_summary.csv`) found M1 still ranks dollar risk better
than B1 (Gini +0.35 vs. +0.27, sorted correctly by predicted dollars) — the
miscalibration is in the *level* of predicted severity, not the relative ranking —
and a segment breakdown (`outputs/tables/segment_breakdown.csv`) shows the
underprediction is broad and systemic (44-63% across every coverage band, occupancy
type, flood zone, and year), not isolated to one property type.

## Reproducing this project

```
pytest -q            # 94 passed
bash run_all.sh       # 01 through 11 (incl. 08b/08c), from raw downloads to the Ian replay
```

**Known issue, as of 2026-11-21**: `02_download_policies.py`'s bulk file download is
returning a 403 from FEMA's own CDN on a from-scratch clone — confirmed external (not
a bug here), and unrelated to any other step. If `bash run_all.sh` stops at step 02
and retrying doesn't help, **download the `data/` bundle from this repo's GitHub
Releases** (tag `data-v1`) and unzip it into the repo root before rerunning — every
other step then proceeds with no FEMA dependency. Full detail: `docs/reproducibility.md`.

Verified 2026-11-22 from two fresh clones: with the full `data-v1` bundle, every output
in `outputs/` reproduces exactly. Rebuilding from `data/raw/` alone reproduces B0/B1/M1
exactly, but M2 and the Ian replay can shift by ~2-3% because the text embeddings are
recomputed (see `docs/reproducibility.md`).

## Limitations

- **M1's catastrophe-severity gap (above) is reported, not fixed** — a deliberate
  decision (`docs/decisions.md`, RUNBOOK 5.2), not an oversight: a diagnostic showed a
  proposed feature-set fix recovers only ~4 of the 57 points, too small a payoff for
  the cost of re-verifying everything downstream before the deadline.
- A **"deployment fit"** of M1/M2 (refit on all claims 2010-2024, not just
  train+validation) is reported only as a side-by-side comparison, not a headline
  result — it has no genuine held-out test of its own. Whether a model like this
  should ever be used for real pricing is a separate, forward-looking question for
  whoever might adopt it; it isn't a decision this project needs to make.
- Two gaps identified against the 15.071 course material (`docs/course-checklist.md`)
  are deferred, not fixed, given the deadline: LightGBM's `lambda_l1`/`lambda_l2`
  regularization was never tuned; hyperparameters were tuned on a single validation
  split, not k-fold cross-validation.
- Minor data-quality notes, not fixed: 11 policies with a non-Florida county code; 7,616
  policies (0.69%) with no `censusGeoid` at all.
- Frequency and shortfall use the single available 2025 policy snapshot as a stand-in
  for every year's exposure (no historical policy-count file exists).

Full decision log: `docs/decisions.md`. Full course-material review: `docs/course-checklist.md`.
