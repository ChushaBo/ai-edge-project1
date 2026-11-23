# Reproducibility notes

## Known issue: FEMA's bulk policies download is currently blocked (2026-11-21)

`src/02_download_policies.py` filters Florida's 2025 policy terms out of a FEMA bulk
file (`NfipPoliciesV3.parquet`, ~3.7GB, all states/years) because the OpenFEMA API
times out paging a dataset this large. As of 2026-11-21, that bulk file URL --
confirmed current and correct against FEMA's own data page
(`fema.gov/openfema-data-page/nfip-redacted-policies-v3`) -- returns a hard
**403 Forbidden** from FEMA's Akamai CDN/WAF, on every attempt, with the normal
request our script always makes (no header spoofing attempted or appropriate). This
is an external outage or block on FEMA's side, not a bug in this repo: step 01
(`01_download_claims.py`, a different OpenFEMA endpoint) downloads cleanly from a
genuinely fresh clone with no local data.

**What this means in practice:**
- Every result in this repo (`outputs/`, `data/models/`) was already computed
  successfully before this block appeared and is unaffected.
- A **fresh clone** that tries `bash run_all.sh` from nothing (no `data/` folder --
  which is never committed to git; see `.gitignore`) will download claims
  successfully at step 01, then stop at step 02 with the 403 above.
- This was caught specifically by RUNBOOK 7.1's reproducibility pass: a clone into an
  isolated temp directory with a fresh venv, run end to end. It is exactly the kind
  of thing that check exists to catch, and it has nothing to do with the modeling
  work in phases 5-6.

**What we did about it:** rather than work around FEMA's access control (not
appropriate), we are submitting the already-downloaded `data/` folder alongside the
repo (see below) so the pipeline can be verified and rerun without depending on this
endpoint being back up. If FEMA's block clears before grading, `run_all.sh` from a
truly empty `data/` folder should be retried as the real end-to-end check; until
then, treat the bundled `data/` folder as the authoritative starting point.

## Data bundle

Because `data/` is gitignored (CLAUDE.md rule 1 -- raw/interim data never goes in
git, and some of it is large), it is submitted separately from the GitHub repo as
`ai-edge-project1-data.zip`. Unzip it into the repo root (so `data/raw/`,
`data/interim/`, `data/processed/`, and `data/models/` sit next to `src/`) before
running anything. With this in place:
- `pytest -q` passes immediately (several tests read from `data/interim/` and
  `data/models/`, and are otherwise skipped).
- `bash run_all.sh` will skip every already-completed download (01, 02, 03, and the
  Census boundary file in 10) since each one checks whether its output already
  exists, and will re-run cleaning, matching, embedding, training, and the
  shortfall/Ian-replay/ranking-check scripts fresh, deterministically (fixed
  `random_seed`), using the bundled inputs -- with no network dependency on FEMA.

## What's in the bundle (sizes as of 2026-11-21)

| Path | Size | Contents |
|---|---|---|
| `data/raw/` | 239M | Florida-filtered claims and policies, NOAA storm CSVs, NWS zone-county file, Census county boundaries |
| `data/interim/` | 289M | Cleaned claims, claims matched to storm narratives |
| `data/processed/` | 17M | Text embeddings (cache + PCA features) |
| `data/models/` | 8.3M | All trained model bundles, including the deployment-fit variants |
| **Total** | **~553M** | |

## Verification run (2026-11-22)

Two fresh clones of `main` (`caca49c`), each in a new venv from the pinned
`requirements.txt` (Python 3.14, Windows, CPU only), running `bash run_all.sh` plus
`08b`/`08c`, then `pytest -q`:

> **Pin update (2026-11-23):** this run used `transformers==5.24.0` and `pyyaml==6.0.4`.
> `requirements.txt` now pins `transformers==5.23.0` and `pyyaml==6.0.3` to match the
> environment that trained the committed models (all other pins already matched it).
> The full-bundle route doesn't depend on either version. Whether the transformers
> version contributes to the M2 drift below is untested.

| Starting data | Result |
|---|---|
| **Full bundle** unzipped (the documented route) | All outputs identical to the committed ones; 94 tests pass; ~5 min. |
| **`data/raw/` only** (everything from cleaning onward rebuilt, embeddings recomputed) | B0, B1, M1, SHAP, frequency, shortfall, ranking check and segment breakdown identical. **M2/M2_masked and the Ian replay differ slightly** (below); 94 tests pass; ~12 min. |

Why M2 moves when embeddings are recomputed: re-encoding the narratives on a different
machine produces vectors that differ by at most ~1e-5 (ordinary floating-point
differences in the sentence-transformer). Several M2 hyperparameter settings are nearly
tied on validation MAE, so that noise is enough to change which one 07 selects. Effect
on the reported numbers:

| Output | Committed | Rebuilt from raw |
|---|---:|---:|
| M2 test_matched MAE | $74,783 | $75,139 |
| M2 test_matched portfolio total, % error | −56.8% | −54.0% |
| Ian replay, M2 (current fit) | $1.32B | $1.35B |
| Ian replay, M2 deploy-fit (circular) | $3.36B | $3.45B |

None of the conclusions change (M1 still has the lowest per-claim MAE, and M2 still
misses Ian by roughly 70%). To reproduce the committed M2 numbers exactly, keep the
bundled `data/processed/embedding_cache.parquet`: step 06 then reuses the cached
vectors instead of re-encoding.

The only other difference is cosmetic: `outputs/tables/policies_columns.csv` lists
date columns as `datetime64[ms, UTC]` instead of `datetime64[s, UTC]` under newer
pandas/pyarrow. The data are the same.
