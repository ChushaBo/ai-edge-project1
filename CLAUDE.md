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

- Done, merged to `main`: RUNBOOK 0.1–6.3 (scripts 01–11), plus three extras beyond
  the RUNBOOK: `08b_ranking_check.py` (Spearman + Gini risk-ranking check on frozen
  M1/B1 — see "key finding" below; also writes `within_storm_variance.csv`: ~90% of
  payout-share variance is within one storm narrative, which is why M2's text can't
  add much), `08c_segment_breakdown.py` (M1/B1/M1_deploy
  actual-vs-predicted by coverage band, occupancy type, flood zone, year), and
  in-sample (train/trainval) rows added to `metrics_overall.csv` by scoring the
  frozen models (no refit). `pytest -q` passes (95 passed, 1 skipped). `README.md` (RUNBOOK 7.2)
  rewritten 2026-11-24 to one page, framed for a reinsurer's data scientist to match
  the deck (ranking first, catastrophe-level gap as the limitation).
- **RUNBOOK 5.2 is DONE** (human sign-off on M1/M2, decided 2026-11-21): present M1 as
  final, as trained, with its known limitation documented rather than fixed before
  submission — see `docs/decisions.md` for the full reasoning (it has the best
  per-claim MAE of any model but underpredicts the test portfolio total by 57.6%, a
  named, textbook limitation of tree ensembles, not a bug; confirmed via SHAP,
  `outputs/tables/ranking_check_summary.csv`, and `segment_breakdown.csv`). **No
  further training work is planned.** The "deployment fit" models remain a labeled
  secondary comparison only, not adopted for the headline numbers (next bullet).
- Finished, not adopted: the "deployment fit" option (`config.yaml:
  deployment_fit.enabled`) — a second copy of M1/M2 refit on all claims 2010–2024,
  used only in 10/11, never touching 07's reported test metrics. Reported as a
  side-by-side comparison only; whether to ever use it for real pricing is a
  separate, forward-looking question for a future adopter, not an open task here.
- Two gaps from `docs/course-checklist.md`'s review are **deferred as documented
  limitations, not fixed** (decision logged in `docs/decisions.md`, given the
  deadline): `lambda_l1`/`lambda_l2` were never added to 07's LightGBM search grid;
  hyperparameters were tuned on a single validation split, not k-fold CV.
- Due **Nov 25, 2026** (moved from Nov 20). Remaining Phase 7 work: slides, video,
  final number-check against `outputs/tables/`, submission. Reproducibility (RUNBOOK
  7.1) **verified 2026-11-22 from the `data-v1` release**: the full bundle reproduces
  every output exactly; rebuilding from `data/raw/` only reproduces B0/B1/M1 exactly
  but shifts M2/Ian replay by ~2-3% (embedding float noise). Details in
  `docs/reproducibility.md`. `requirements.txt` is now pinned to the verified versions.
- Fresh clone: `data/` is not in git. **As of 2026-11-21, `02_download_policies.py`'s
  bulk file is blocked by a 403 from FEMA's own CDN** — confirmed external (not a
  code bug) on two independent attempts; see `docs/reproducibility.md`. The built
  `data/` folder (~553MB) is submitted alongside the repo as a GitHub Release
  (tag `data-v1`) so grading doesn't depend on FEMA's endpoint recovering — unzip it
  into the repo root, then `pytest -q` and `bash run_all.sh` both work with no FEMA
  dependency. **`.gitignore`'s `*.zip` rule had a bug (inline comment on the pattern
  line, which git doesn't support) that left the zip untracked-but-not-actually-
  ignored — fixed and verified with `git check-ignore`; re-verify before any future
  `git add .` if `.gitignore` changes again.**

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
