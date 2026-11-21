# CLAUDE.md — Florida Flood Insurance Shortfall Project

Context and rules for Claude Code. Read this before every task.

## Project summary

15.071 (The AI Edge) Project 1, due Nov 25, 2026 (moved from Nov 20). We predict the building payout of
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

- Done: RUNBOOK 0.1–6.3. Scripts 01–11 are all implemented and tested (`pytest -q`
  passes, 86 tests). **Not yet merged to `main`** — sitting on two branches,
  `jason/5.1-train-models` (07, RUNBOOK 5.1) and `jason/overnight` (08–11, RUNBOOK
  5.3–6.3), both pushed, neither has an open PR yet. All decisions are in
  `config.yaml` and logged in `docs/decisions.md`.
- **Next, blocking: RUNBOOK 5.2** (human sign-off on M1/M2). Open item logged in
  `docs/decisions.md`: M1 has the best per-claim MAE of any model but underpredicts
  the test portfolio total by 57.6%, because latitude/longitude let it memorize
  pre-2020 location averages that don't extrapolate to Ian/Helene/Milton severity —
  a named, textbook limitation of tree ensembles (`docs/course-checklist.md`,
  Lecture 5: "a tree can never predict a value outside the range in the training
  data"). Decide: present M1 as-is with the limitation documented, fix the feature
  set (e.g. coarsen lat/long to county — a holdout diagnostic in
  `notebooks/diag_leave_one_storm_out.py` shows this recovers only ~4 points) and
  rerun 07–11, or adopt the "deployment fit" comparison models (pending approval,
  see below). Whatever is decided, phases 6–7 downstream need rerunning if M1/M2
  themselves change.
- Also pending team approval: the "deployment fit" option (`config.yaml:
  deployment_fit.enabled`) — a second copy of M1/M2 refit on all claims 2010–2024
  instead of just train+validation, used only in 10_shortfall.py/11_ian_replay.py,
  never touching 07's reported test metrics. Logged in `docs/decisions.md`.
- `docs/course-checklist.md` has a full review of the pipeline against the 15.071
  lecture decks' recommendations (splits, baselines, metrics, CV, regularization,
  interpretability, business framing), prioritized. Two cheap, high-value fixes
  flagged there and not yet done: an in-sample (train) row in
  `outputs/tables/metrics_overall.csv` alongside the existing test rows, and
  `lambda_l1`/`lambda_l2` in 07's LightGBM search grid (config.yaml `models.
  search_grid`) — neither is in yet.
- Due Nov 25, 2026. Phase 7 (reproducibility pass, README, slides, video) has not
  started.
- Fresh clone: `data/` is not in git. Run `src/01`–`11` to rebuild it (02 downloads a
  3.7 GB bulk file once), or `bash run_all.sh`. **As of 2026-11-21, step 02's bulk
  download is blocked by a 403 from FEMA's own CDN (confirmed external, not a code
  bug) — see `docs/reproducibility.md`.** The already-built `data/` folder is
  submitted alongside the repo as `ai-edge-project1-data.zip` so this doesn't block
  grading; unzip it into the repo root before running anything.

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
