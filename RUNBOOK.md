# RUNBOOK — Building the Florida Flood Shortfall Project with Claude Code

How to use this file:
- **[HUMAN]** steps must be done by a person. Do not hand them to Claude Code.
- **[CLAUDE CODE]** steps include a prompt to paste into Claude Code, run from the repo root.
- **CHECKPOINT** lines say what to verify before moving on.
- After every Claude Code step: review the diff, run `pytest -q`, and commit.

---

## Phase 0 — Setup (Fri 11/13)

### 0.1 [HUMAN] Create the repository
- Create a GitHub repo (e.g. `fl-flood-shortfall`), add teammates as collaborators.
- Clone it locally. Copy `CLAUDE.md` and `RUNBOOK.md` into the repo root and commit.
- Agree on a workflow: one branch per person, pull requests into `main`, one reviewer.
- Install Claude Code on each machine (see https://docs.claude.com/en/docs/claude-code/overview)
  and confirm it can reach the internet, since the pipeline downloads public data.

### 0.2 [CLAUDE CODE] Scaffold the repository
```
Read CLAUDE.md. Scaffold the repository layout it describes: folders, empty numbered
scripts in src/ with docstrings stating each script's inputs and outputs, src/utils/
with a config loader and logging helper, a .gitignore that excludes data/ and .venv/,
requirements.txt (pandas, pyarrow, requests, numpy, scikit-learn, lightgbm, shap,
sentence-transformers, geopandas, matplotlib, pyyaml, pytest, tqdm), run_all.sh that
runs the scripts in order and stops on error, docs/decisions.md with a dated log
template, and config.yaml with placeholder sections for: paths, years, split dates,
target definition, allowed features, match window, storm event types, embedding model,
PCA dimensions, and random seed. Mark every decision value in config.yaml with a
"# DECISION:" comment. Do not fill decision values yet. Add one smoke test.
```
**CHECKPOINT:** `pytest -q` passes; folder layout matches CLAUDE.md; nothing in `data/` is tracked.

### 0.3 [HUMAN] Fill in the first decisions in `config.yaml`
Suggested starting values (log each in `docs/decisions.md`):
- Claims years: 2010–2024. Exclude 2025+ because many of those claims may still be open.
- Split by date of loss: train ≤ 2019, validation 2020–2021, test 2022–2024.
- Match window: ±3 days; event types: Flood, Flash Flood, Coastal Flood,
  Storm Surge/Tide, Heavy Rain, Hurricane, Tropical Storm.
- Embedding model: `all-MiniLM-L6-v2`; PCA dimensions: 24; seed: 42.
- Policies: Florida, policies in force during the most recent complete year available.

---

## Phase 1 — Data acquisition (Fri 11/13 – Sat 11/14)

### 1.1 [CLAUDE CODE] Download Florida claims
```
Implement src/01_download_claims.py. Find the current OpenFEMA API endpoint and data
dictionary for "FIMA NFIP Redacted Claims" (v2). Save the dictionary to
docs/data_dictionaries/. Download only Florida records with year of loss in the
config.yaml range, paging through the API with $filter and $skip, with retries and a
progress bar. Save to data/raw/claims_fl.parquet. Skip the download if the file
exists unless --force is passed. Log total rows and rows per year. Then list every
column with dtype and percent missing, and save that table to
outputs/tables/claims_columns.csv. Do not drop or rename columns yet.
```

### 1.2 [CLAUDE CODE] Download Florida policies
```
Implement src/02_download_policies.py following the same pattern as 01. Find the
current OpenFEMA endpoint and dictionary for "FIMA NFIP Redacted Policies" and save
the dictionary. This dataset is very large: filter server-side to Florida and the
policy period in config.yaml, and use $select to keep only columns needed for county,
flood zone, coverage, premium, policy dates, and the building characteristics that
match our claim features. Report which premium field(s) exist and what they include
(e.g. whether fees and surcharges are separate). Save to data/raw/policies_fl.parquet
and a column summary to outputs/tables/policies_columns.csv.
```

### 1.3 [CLAUDE CODE] Download storm events and the zone-county file
```
Implement src/03_download_storms.py. From NOAA NCEI's Storm Events bulk CSV directory,
download the "details" files for each year in config.yaml, filter to STATE = FLORIDA,
and keep: event and episode IDs, event type, begin and end dates, CZ_TYPE, CZ_FIPS,
CZ_NAME, damage property, episode narrative, and event narrative. Save to
data/raw/storms_fl.parquet. Also download the National Weather Service zone-to-county
correlation file and save it to data/raw/. Log rows per year, rows per event type,
and the share of rows with CZ_TYPE = Z versus C.
```

### 1.4 [HUMAN] Review the data dictionaries
- Read the saved dictionaries. For each field we plan to use, confirm the exact name,
  meaning, and units. Write the confirmed list into `config.yaml` (allowed features,
  target field, coverage field, premium field, county field, date-of-loss field).
- Check row counts against common sense (e.g. 2017 and 2022 should be large years).
- Record anything surprising in `docs/decisions.md`.

**CHECKPOINT:** all three raw files exist; column tables are committed; field names in
`config.yaml` are confirmed by a person.

---

## Phase 2 — Cleaning and target (Sun 11/15)

### 2.1 [HUMAN] Decide the target definition
Choose one and record it in `config.yaml` and `docs/decisions.md`:
- **Option 1 (recommended):** payout as a share of building coverage. Handles inflation
  and the $250,000 cap without adjustment.
- **Option 2:** log of payout in 2025 dollars (requires CPI adjustment).

Also decide: drop claims with $0 paid from the severity model (recommended) and report
how many there were.

### 2.2 [CLAUDE CODE] Clean claims and build features
```
Implement src/04_clean_claims.py. Using only fields confirmed in config.yaml:
build the target as defined in config.yaml; drop $0-paid claims and log how many;
apply CPI adjustment if the target requires it (add a small CPI CSV under data/raw
with its source noted); derive month of loss and building age at loss; build the
feature table using ONLY the allowed feature list; assign each row to train,
validation, or test by date of loss per config.yaml. Save to
data/interim/claims_clean.parquet. Write tests/test_leakage.py that fails if any
column outside the allowed list (plus the target, split, ID, county, and date
columns) is present, and tests/test_split.py that checks no date overlap between
splits. Produce outputs/tables/claims_summary_by_split.csv with counts and target
mean/median by split and flood zone.
```

### 2.3 [HUMAN] Leakage sign-off
- Read the final feature list line by line and ask: "Would we know this before the
  flood?" Remove anything that fails. Log the sign-off in `docs/decisions.md`.

**CHECKPOINT:** leakage and split tests pass; summary table looks sensible (e.g. V and
AE zones have higher average payouts than X).

---

## Phase 3 — Matching claims to storm narratives (Sun 11/15 – Mon 11/16)

### 3.1 [CLAUDE CODE] Build the matcher
```
Implement src/05_match_narratives.py. Map every storm event to county FIPS codes:
CZ_TYPE = C events use CZ_FIPS directly; CZ_TYPE = Z events use the NWS zone-to-county
file (a zone can map to multiple counties; keep all). Filter to the event types in
config.yaml. For each claim, find events in the same county whose begin date is within
the config.yaml window of the claim's date of loss. Concatenate unique episode and
event narratives per claim (deduplicate repeated episode text). Add columns:
n_matched_events, matched_event_types, has_narrative. Save to
data/interim/claims_with_text.parquet. Log the overall match rate and the match rate by
year and by split. Write tests/test_matching.py with small synthetic examples
covering: county match, zone match, multi-county zone, outside window, wrong type.
Also export 20 random matched claims (claim date, county, narrative excerpt) to
outputs/tables/match_spot_check.csv.
```

### 3.2 [HUMAN] Review match quality
- If the overall match rate is below about 70%, check the zone mapping first, then
  consider widening the window. Change values in `config.yaml` yourself and rerun.
- Read the 20 spot-check rows: does each narrative plausibly describe the flood that
  caused that claim? Log the result.

**CHECKPOINT:** matching tests pass; match rate recorded; spot check reviewed.

---

## Phase 4 — Text features (Mon 11/16)

### 4.1 [CLAUDE CODE] Embed narratives
```
Implement src/06_embed_text.py. Embed each claim's combined narrative with the
sentence-transformers model in config.yaml, batching and caching embeddings per unique
narrative text so duplicates are embedded once. Create a second "masked" version of
the text where dollar amounts and numbers attached to damage words are replaced with a
placeholder, and embed that too. Fit PCA on TRAIN rows only to the dimension in
config.yaml, apply to all rows, and save both variants to
data/processed/text_features.parquet. Claims without a narrative get zeros plus the
has_narrative flag. Log explained variance and runtime. Add a test that PCA was fit
only on training rows.
```

**CHECKPOINT:** runtime is reasonable on a laptop or Colab; the masked variant exists.

---

## Phase 5 — Modeling and evaluation (Mon 11/16 – Tue 11/17)

### 5.1 [CLAUDE CODE] Train and evaluate models
```
Implement src/07_train_models.py. Models:
B0 = training mean; B1 = training mean by flood zone (fallback to B0 for unseen zones);
M1 = LightGBM on tabular features; M2 = M1 features + text PCA features;
M2_masked = M1 features + masked text PCA features.
Tune M1 and M2 hyperparameters with a small search on the VALIDATION set only. Then
retrain on train+validation and evaluate ONCE on test. Metrics: MAE and RMSE in the
target's units and converted to dollars; for each storm event in test (group by
matched episode, or by date-of-loss clusters when unmatched), total predicted vs.
total actual payout and percent error. Save outputs/tables/metrics_overall.csv,
outputs/tables/metrics_by_event.csv, the trained models, and a bar chart comparing
MAE across models. Print a short plain-English summary of results.
```

### 5.2 [HUMAN] Interpret results
- Does M1 beat B1? Does M2 beat M1? Does M2_masked keep most of M2's gain? If masking
  erases the gain, the text is mostly leaking damage figures; say so honestly.
- Check whether one storm (likely Ian) dominates the test results.
- Do not go back and tune on test results. If you must change something, log why.

### 5.3 [CLAUDE CODE] Explain the model
```
Implement src/08_explain.py. Compute SHAP values for M1 on a sample of test rows.
Save a summary plot, dependence plots for the top 5 features (including elevation
difference if present), and a table translating the top effects into approximate
dollars for a typical $250,000-coverage home. Write a short markdown note
(outputs/tables/shap_findings.md) listing the top findings in plain language.
```

**CHECKPOINT:** metrics and SHAP outputs committed; a person has written 3–5 key
findings into `docs/decisions.md`.

---

## Phase 6 — Shortfall and Ian replay (Tue 11/17 – Wed 11/18)

### 6.1 [CLAUDE CODE] Claim frequency
```
Implement src/09_frequency.py. Estimate annual claim frequency per policy by county
and flood zone over the longest consistent window available, using policies-in-force
counts as the denominator and paid claims as the numerator. Apply simple smoothing
(shrink small cells toward the county or statewide rate) and document the method in
the script docstring. Save outputs/tables/frequency_by_county_zone.csv.
```

### 6.2 [CLAUDE CODE] Shortfall by county
```
Implement src/10_shortfall.py. For the current Florida policy file, predict severity
with M1 (tabular only), multiply by the frequency for each policy's county and zone to
get expected annual loss, and sum by county. Compare to total premium by county using
the premium field confirmed in config.yaml. Save
outputs/tables/shortfall_by_county.csv (policies, premium, expected loss, gap, loss
ratio) and a Florida county choropleth of the loss ratio using Census county
boundaries. Include a statewide total row.
```

### 6.3 [CLAUDE CODE] Ian replay
```
Implement src/11_ian_replay.py. Using Hurricane Ian's matched narratives (Sept–Oct
2022), assign each county Ian affected its Ian narrative embedding, apply M2 to
current policies in those counties, and assume every policy in those counties with a
flood zone in the high-risk set files a claim at the historical Ian claim rate for its
county and zone (computed from 2022 claims and policy counts). Report total predicted
payout vs. annual premium collected in those counties. Save a table and one clear
headline chart. State every assumption in the script docstring and in
outputs/tables/ian_replay_assumptions.md.
```

### 6.4 [HUMAN] Sanity-check the headline numbers
- Compare model totals for Ian with publicly reported NFIP payouts for Ian. If they are
  off by a large factor, find out why before using them.
- Write the caveats: premiums also cover administrative costs, NFIP can borrow from
  the U.S. Treasury, frequency estimates are rough, and the replay is a scenario, not
  a forecast.

**CHECKPOINT:** results frozen by end of Wednesday. Tag the commit `results-v1`.

---

## Phase 7 — Deliverables (Wed 11/18 – Fri 11/20)

### 7.1 [CLAUDE CODE] Reproducibility pass
```
From a fresh clone in a temporary directory, create a new virtual environment, install
requirements.txt, and run run_all.sh end to end. Fix any missing dependencies,
hard-coded paths, or ordering issues. Pin package versions in requirements.txt to the
versions that worked. Report total runtime and any step that needs network access.
```

### 7.2 [CLAUDE CODE] Draft the README
```
Draft README.md (one page) for a data scientist at FEMA: problem, data sources with
links and citation per OpenFEMA terms, pipeline overview mapped to src/ scripts, key
design choices (time split, leakage exclusions, narrative matching, PCA fit on train),
results table from outputs/tables/metrics_overall.csv, how to reproduce, and
limitations. Use numbers only from files in outputs/. Do not invent results.
```

### 7.3 [HUMAN] Edit the README and check the repo
- Verify every number in the README against `outputs/tables/`.
- Confirm no raw data, credentials, or personally identifying information is in the
  repo (the data is redacted at source, but check anyway before publishing).

### 7.4 [HUMAN] Slides and video
- Slides (5–10) for a FEMA or Florida policy official: problem, data, approach,
  results vs. flood-zone benchmark, SHAP findings, shortfall map, Ian replay headline,
  recommendations and limitations.
- Two-minute video: open with the Ian replay number, show the map, one sentence on what
  the model does better than flood zones, close with the recommendation.
- Claude Code can regenerate figures in a presentation style if needed; the story,
  recommendations, and narration are yours.

### 7.5 [HUMAN] Final check (Fri 11/20)
- Fresh clone runs end to end; tests pass; README numbers match outputs.
- Submit the repo link, slides, and video.

---

## Tips for working with Claude Code on this project

- For larger steps (1.1, 3.1, 5.1, 6.3), ask Claude Code to propose a plan first and
  approve it before it writes code.
- Keep each Claude Code session to one step, then commit. Small diffs are easier to
  review.
- If Claude Code reports a field name mismatch or an unexpected row count, treat that
  as a human decision point, not something to work around.
- Decisions belong in `config.yaml` and `docs/decisions.md`, made by people. Claude
  Code implements them.
