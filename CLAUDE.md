# CLAUDE.md — Florida Flood Insurance Shortfall Project

Context and rules for Claude Code. Read this before every task.

## Project summary

15.071 (The AI Edge) Project 1, due Nov 20, 2026. We predict the building payout of
National Flood Insurance Program (NFIP) claims in Florida and compare against a
flood-zone-only benchmark. Storm narratives from NOAA Storm Events are embedded with a
pretrained sentence model as an unstructured-data (deep learning) component. Predicted
losses are compared with NFIP premiums to estimate a county-level shortfall, plus an
"Ian replay" scenario.

Two model uses (keep them separate):
- Tabular model (M1): pre-event information only. Used for pricing and the shortfall.
- Tabular + text model (M2): post-event loss estimation and the Ian replay.

## Data sources (all public)

- OpenFEMA NFIP Redacted Claims **v3** (`NfipClaims`): claim-level payouts. The v2
  dataset ("FIMA NFIP Redacted Claims", `FimaNfipClaims`) is retired; do not switch back.
- OpenFEMA NFIP Redacted Policies **v3** (`NfipPolicies`): policy-level coverage and
  premium. Script 02 filters FEMA's bulk parquet locally because API paging times out.
- NOAA NCEI Storm Events "details" CSVs: event type, dates, county/zone, narratives.
- NWS zone-to-county correlation file: maps forecast zones (CZ_TYPE = Z) to counties.
- Census cartographic boundary files: Florida county shapes for maps.
- CPI (BLS CPI-U annual averages): only needed for a dollar target; the current target
  (share of coverage) does not use it.

Always verify current endpoints and field names against the official data dictionaries.
If a field name differs from what a task expects, STOP and report; do not guess.

The saved dictionaries in `docs/data_dictionaries/` are the source of truth for field
names. Several differ between claims and policies (e.g. `state` vs `propertyState`;
policies have no county field, so county = first 5 digits of `censusGeoid`); the
mapping is in `config.yaml` under `fields`.

## Current status

Update this section whenever a RUNBOOK step is finished.

- Done: RUNBOOK 0.1–4.1. Scripts 01–06 are implemented; all decisions so far are in
  `config.yaml` and logged in `docs/decisions.md` (field review: `docs/field_review_1_4.md`).
- Next: **5.1** (`src/07_train_models.py`). Before it: decide whether to keep the Helene
  quarantine (below). Text features for M2 are in `data/processed/text_features.parquet`
  (`txt_pc*` raw, `txtm_pc*` masked); the fitted PCAs are in `data/models/`.
- Quarantine: 46,971 Helene claims (Sept 24–30, 2024) have no NOAA narrative because
  NOAA's 2024 file lacks Helene events for the Tampa Bay counties. They are flagged
  `quarantined` in `claims_with_text.parquet`; steps 06+ must call
  `utils.matching.exclude_quarantined()` while `quarantine.enabled` is true. Revisit
  before 5.1.
- Open for human review: cleaning defaults in `config.yaml` (`cleaning.building_age_years`,
  `cleaning.min_policy_tenure_years`).
- Fresh clone: `data/` is not in git. Run `src/01`–`06` to rebuild it (02 downloads a
  3.7 GB bulk file once), or `bash run_all.sh` once the later scripts exist.

## Repository layout

```
config.yaml            # all decisions: years, splits, windows, model settings, seed
src/                   # numbered pipeline scripts, run in order
  01_download_claims.py
  02_download_policies.py
  03_download_storms.py
  04_clean_claims.py
  05_match_narratives.py
  06_embed_text.py
  07_train_models.py
  08_explain.py
  09_frequency.py
  10_shortfall.py
  11_ian_replay.py
  utils/               # shared helpers (io, config loading, metrics)
tests/                 # pytest tests for matching, leakage guards, splits
notebooks/             # exploration only; never the source of reported results
data/raw|interim|processed/   # gitignored
outputs/figures|tables/       # committed (small files only)
docs/data_dictionaries/       # saved dictionary metadata
docs/decisions.md             # human decisions log
run_all.sh             # reproduces everything from raw downloads
```

## Hard rules

1. Never commit data files from `data/`. Never commit files over 5 MB.
2. All tunable choices live in `config.yaml`. Do not change decision values there
   (years, splits, target definition, feature lists, match window) without asking.
3. Leakage: never use post-loss fields as features. Excluded at minimum: building damage
   amount, contents damage amount, cause of damage, any amount paid field other than the
   target, and named flood event identifiers. The allowed feature list is in
   `config.yaml`; a test must fail if any other column reaches the model.
4. Splits are by date of loss (time-based), defined in `config.yaml`. Fit every
   transformer (PCA, encoders, scalers) on training rows only.
5. The test set is used once, for final reported metrics. Tune on validation only.
6. Set random seeds from `config.yaml` everywhere (numpy, sklearn, lightgbm, torch).
7. Scripts read from and write to the paths in `config.yaml`, log row counts at every
   step, and are idempotent (safe to rerun; skip downloads that already exist).
8. Prefer parquet for intermediate data. Use pandas or polars; geopandas for maps.
9. Every figure saved to `outputs/figures/` must have a title, labeled axes with units,
   and a source note.
10. When finishing a task, report: what changed, row counts, any warnings, and anything
    a human needs to check. Do not mark work complete if tests fail.

## Commands

- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
  (Windows: activate with `.venv\Scripts\activate`; tested on Python 3.14)
- PR review: ask Claude Code to "review this branch with pr-reviewer" (`.claude/agents/pr-reviewer.md`)
- Tests: `pytest -q`
- Full pipeline: `bash run_all.sh`
