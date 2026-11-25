# Florida Flood Risk Ranking for Reinsurers

15.071 (The AI Edge) Project 1. Written for a data scientist at a reinsurer assessing
Florida flood exposure. We predict the building payout of each National Flood
Insurance Program (NFIP) claim and test whether machine learning separates low-loss
from high-loss policies better than FEMA flood zones, the current rule of thumb.

## Approach

- **Target:** building payout ÷ building coverage (paid claims only, capped at 1.0).
- **Split by date of loss:** train 2010–19 (38,678 claims), validation 2020–21 (7,657),
  test 2022–24 (112,573; Ian, Helene, Milton), test scored once. Final models are refit
  on train + validation.
- **Models:** B0 = statewide mean; **B1 = mean by flood zone (the benchmark)**;
  **M1** = LightGBM on 20 pre-event features (allowed list in `config.yaml`; a test fails
  if any other column reaches a model); **M2** = M1 + month + NOAA storm narratives
  embedded with `all-MiniLM-L6-v2` and reduced to 24 PCA components fit on training
  rows; **M2_masked** = M2 with dollar and damage figures removed (leakage control).
  M1 is the pre-event pricing model; M2 is post-event only (loss estimates, Ian replay).
- **Expected loss** on 1.1M policies in force in 2025 = county × flood-zone claim
  frequency × M1 severity × coverage.
- **Data (public):** OpenFEMA NFIP Claims v3 and Policies v3, NOAA Storm Events, NWS
  zone–county file, Census county boundaries. Field dictionaries: `docs/data_dictionaries/`.

## Results (test set 2022–24)

| | B1 (flood zone) | M1 |
|---|---:|---:|
| Average error per claim | $109,767 | **$78,067** |
| Rank correlation, predicted vs actual $ | 0.25 | **0.52** |
| Gini on predicted $ | 0.27 | **0.35** |
| Paid by the half of claims rated safest | $4.17B (35%) | **$3.28B (27%)** |
| Portfolio total vs actual | **−3.8%** | −57.6% |

- **M1 ranks risk well but sets the level too low.** Test claims paid 37% of coverage
  vs 19% in training; a tree ensemble can't predict beyond what it has seen, and the
  miss is 44–63% in every segment (`segment_breakdown.csv`). Use M1 for relative
  pricing and selection; set the overall level from catastrophe scenarios.
- **Storm text adds no accuracy.** On 64,620 matched claims: M1 $73,457, M2 $74,783,
  M2_masked $73,228 per claim. About 90% of payout-share variance is within a single
  storm narrative (`within_storm_variance.csv`), which text cannot see.
- **Premiums vs expected loss:** $1.52B premium vs $422M expected building loss (M1, a
  lower bound) or $770M (B1). **Ian replay:** M2 predicts $1.32B vs $4.32–4.36B paid.

Tables: `outputs/tables/` (`metrics_overall`, `ranking_check_*`, `shortfall_*`,
`ian_replay_*`). Figures: `outputs/figures/`.

## Reproduce

```
pip install -r requirements.txt   # pinned; tested on Python 3.14
pytest -q
bash run_all.sh                   # src/01-11 in order, incl. 08b/08c
```

Scripts: `01–03` download, `04` clean + split, `05–06` match + embed narratives,
`07` train and evaluate, `08/08b/08c` SHAP, ranking and segment checks, `09–10`
frequency and county expected loss, `11` Ian replay. All settings are in `config.yaml`.

FEMA's bulk policy download (step 02) returned 403 as of 2026-11-21. Unzip the `data/`
bundle from this repo's GitHub release `data-v1` into the repo root first: every output
then reproduces exactly. Rebuilding from `data/raw/` alone moves M2 and the Ian replay
by ~2–3% (embedding float noise). Details: `docs/reproducibility.md`.

## Limitations

- M1's catastrophe-level gap is reported, not fixed (decision in `docs/decisions.md`).
- An all-years refit ("deployment fit") is a sensitivity check only; it has no test set.
- Claim frequency uses 2025 policy counts for every year; losses cover buildings only.
- Hyperparameters tuned on one validation split; L1/L2 regularization not searched.
- Claim locations are rounded to ~11 km by FEMA, so street-level risk is not visible.

More: `docs/decisions.md` (decision log), `docs/course-checklist.md` (course review).
