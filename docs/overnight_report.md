# Overnight report — 2026-11-15/16

Branch: `jason/overnight` (off `jason/5.1-train-models`). No PR opened; `main` untouched.
Commits: one per step below, each with `pytest -q` passing (86 tests, up from 75).

Everything from RUNBOOK 5.3 through 6.3 is implemented. **Nothing was skipped**, no
step failed outright; where a choice came up I took the safest option that didn't
change a `# DECISION` value in `config.yaml`, noted it below, and kept going. I
stopped before 6.4 **[HUMAN]** and did not touch `main`, `CLAUDE.md`'s "Current
status," or anything in phase 7.

**Everything here inherits M1/M2's open item** (logged last night, reconfirmed by
SHAP tonight): the official models underpredict catastrophe-year severity. Read every
dollar figure below as a plausible lower bound, not a final number, until that's
resolved in 5.2.

---

## Setup: the "deployment fit" option (pending team approval)

Added `config.yaml: deployment_fit.enabled` (implementation setting, not a
`# DECISION`) and a matching entry in `docs/decisions.md`. When true,
`10_shortfall.py` and `11_ian_replay.py` each fit a **second** copy of M1/M2 — same
already-tuned hyperparameters, refit on **all** claims 2010–2024 (train+validation+test)
instead of just train+validation — saved separately as `data/models/m1_deploy_lgbm.joblib`
/ `m2_deploy_lgbm.joblib`. It never touches `07_train_models.py` or any reported test
metric. The motivation: last night's holdout diagnostic showed coarsening
latitude/longitude to county only recovers ~4 points of the portfolio-total
underprediction — the deeper issue is that 2010–2021 training never saw
catastrophe-level severity at all, so a model that has already seen it (deployment fit)
has no train/test gap left to fall into *for new policies*. Whether to actually adopt
these models for real pricing is flagged as a team decision, not something I resolved —
see the full reasoning in `docs/decisions.md`.

---

## Step 1 — RUNBOOK 5.3: `src/08_explain.py` (SHAP on M1)

**Done.** TreeExplainer on the saved M1 model, 3,000-row random test sample (seed 42).

- `outputs/figures/shap_summary.png`, 5 dependence plots (top 5 by mean |SHAP|, plus
  `elevationDifference` which was already in the top 5) + box/strip plots for
  categorical features.
- `outputs/tables/shap_top_effects_dollars.csv` (19 rows: dollar effect for a typical
  $250k-coverage home, by category level or numeric tercile).
- `outputs/tables/shap_findings.md`.
- All outputs marked **PROVISIONAL (pending RUNBOOK 5.2)**.

**Does lat/long dominate SHAP?** Two metrics, and they disagree, so I reported both:
- **By SHAP impact magnitude** (this 3,000-row sample): latitude + longitude = **22.8%**
  of total mean |SHAP|. `totalBuildingInsuranceCoverage` ranks #1 individually (0.0401);
  latitude is #2 (0.0342); longitude is only #10 (0.0076). Not an outright majority.
- **By tree split count** (how finely the model partitions space): latitude + longitude
  = **31.4%** of every split M1 makes — more than any other feature by a wide margin
  (longitude 1,459 splits, #1; latitude 1,386, #2; next is `totalBuildingInsuranceCoverage`
  at 738).
- These measure different things. SHAP magnitude is averaged over mostly-typical
  claims, where lat/long's effect is real but not dominant. Split count reflects
  structure — how many separate, fine-grained spatial leaves the tree builds — which is
  the actual mechanism in the open-item hypothesis (a leaf memorizes one location's
  pre-2020 average and can't extrapolate to a catastrophe there). **The split-count
  evidence is the more direct support for the hypothesis**; I did not overclaim "yes,
  dominates" from SHAP magnitude alone, since `totalBuildingInsuranceCoverage` actually
  ranks higher there.

No failures. No decision values touched.

---

## Step 2 — RUNBOOK 6.1: `src/09_frequency.py` (claim frequency)

**Done.** Annual claim frequency per policy = claims over 2010–2024 (15 years) /
(policies in force × 15), with two-level Bühlmann credibility shrinkage (cell → county
→ statewide). Credibility constants (`k_county=100`, `k_cell=30`, claim counts) are a
new `config.yaml: frequency.*` block — implementation settings, not decisions.

- Numerator: `claims_clean.parquet`, 158,908 rows.
- Denominator: `policies_fl.parquet`, 1,146,045 raw → **1,103,860** after excluding
  cancelled terms (per the existing `cleaning.exclude_cancelled_policies` decision).
- Statewide rate: **0.00960** claims/policy-year.
- `outputs/tables/frequency_by_county_zone.csv`: 229 rows. Smoothed rate range: min
  0.00065, median 0.00960 (= statewide, as expected for low-credibility cells), max
  0.17894 (Monroe County, zone V — the Keys, unsurprising).

**Data-quality notes (safe default taken, not fixed, flagged for the team):**
1. 10 non-Florida county codes (e.g. `06095`, a California FIPS) appear among
   11 policies — almost certainly bad `censusGeoid` values at the source, 0.001% of
   policies. Left in; negligible effect (0 claims attached to any of them).
2. Separately, **7,616 of 1,103,860 non-cancelled policies (0.69%) have no
   `censusGeoid` at all.** These policies get frequency = 0 (safe default: no county
   means no cell to look up), and they still count in statewide totals downstream
   (step 3) but drop out of per-county breakdowns. Worth a real fix if these policies
   matter to the team (e.g. a different join key), but 0.69% is small enough that I
   judged "note and continue" the safer overnight choice over guessing at a repair.

No failures. No decision values touched.

---

## Step 3 — RUNBOOK 6.2: `src/10_shortfall.py` (shortfall, 3 severity versions)

**Done.** Current (2025) policy file scored for expected annual loss (severity ×
coverage × frequency), summed by county, compared against premium (all three premium
measures from the existing decision: `totalInsurancePremiumOfThePolicy`, `policyCost`,
`fullRiskPremium`). Census county boundaries (`cb_2023_us_county_500k`, downloaded once
to `data/raw/census/`) for the choropleth.

**The three statewide totals, side by side** (`outputs/tables/shortfall_totals_comparison.csv`,
all PROVISIONAL):

| severity_model | n_policies | premium_total | expected_loss_total | gap | loss_ratio |
|---|---:|---:|---:|---:|---:|
| **m1** (official, current) | 1,103,860 | $1,515,706,398 | **$421,655,664** | −$1,094,050,734 | **0.278** |
| **m1_deploy** (refit on 2010–2024, pending approval) | 1,103,860 | $1,515,706,398 | **$774,391,032** | −$741,315,366 | **0.511** |
| **b1** (flood-zone baseline) | 1,103,860 | $1,515,706,398 | $769,627,007 | −$746,079,391 | 0.508 |

(Premium total differs slightly from the $1.58B cited in the earlier premium decision
because that figure predates this run's consistent cancelled-policy exclusion;
$1.516B × ~1.037 ≈ $1.572B, consistent with the ~3.7% cancelled share already on
record — not a new discrepancy.)

**Key finding:** `m1_deploy` very nearly **doubles** the official M1's expected-loss
estimate and lands almost exactly on the independent B1 baseline. That's the clearest
evidence yet that the catastrophe-blindness problem is substantially a train/test
split artifact for *this* forward-looking use case, not an unfixable flaw in M1's
features — a model that has actually seen 2022–2024 severity prices the shortfall much
more like a naive zone average than like the official, test-blind M1. This doesn't
resolve the open item (M1 still needs a real fix before it's trusted for anything
backtested), but it's a materially different number for the team to weigh for phase 6.

Per-county table (`outputs/tables/shortfall_by_county.csv`, 234 rows, 3 severity models
× ~78 counties incl. ALL) and the official choropleth (`outputs/figures/shortfall_loss_ratio_map.png`,
M1 current) both saved. Network was available; the Census download and the map
succeeded on the first try — nothing to report as failed there.

No decision values touched.

---

## Step 4 — RUNBOOK 6.3: `src/11_ian_replay.py` (Ian replay, 2 severity versions)

**Done.** 32 Florida counties identified as Ian-affected (36,646 2022 claims whose
matched narrative mentions "Ian"); each gets the mean text embedding of its own
Ian-matched claims, assigned to every one of its current (2025) policies along with
`month_of_loss = 9`. High-risk zones (V, A, AH_AO — stated assumption, not a config
decision) are assumed to file a claim at the historical Ian rate (claims in the
2022-09-23..2022-10-07 window ÷ 2025 policies in that county+zone cell — same
policy-count-proxy limitation as step 2). Full assumption list in
`outputs/tables/ian_replay_assumptions.md`.

**Headline comparison** (`outputs/tables/ian_replay_headline_comparison.csv`,
`outputs/figures/ian_replay_headline.png`):

| source | total | note |
|---|---:|---|
| Our data, FL, Ian-narrative-matched claims | **$4.361B** | 36,646 claims |
| Our data, FL, all claims in the Ian date window | $4.322B | 35,967 claims (matched or not) |
| **FEMA reported (nationwide, cited)** | **$4.900B** | 48,776 claims |
| Replay: M2 (official, current) | **$1.323B** | today's policies, scenario |
| Replay: M2 deploy — **CIRCULAR** | $3.358B | today's policies; trained on Ian's own 2022 claims |

Source for the FEMA figure: FEMA's reported final NFIP tally for Hurricane Ian
(48,776 claims, $4.9B, nationwide and overwhelmingly Florida), as reported in FEMA
press materials and summarized by Artemis.bm, retrieved 2026-11-16 — full citation
note in `ian_replay_assumptions.md`. Our own data's $4.32–4.36B (Florida only,
building-only) is consistent with that nationwide, all-perils figure.

**Official M2 underpredicts the replay by about 70%** relative to our own actual-Ian
figure — a *larger* relative miss than M1's 57.6% on the broader test set, which makes
sense: this scenario stress-tests exactly the worst case the open item is about.
`m2_deploy` (flagged CIRCULAR everywhere it appears, including the chart x-axis label
and every table note — Ian's own 2022 claims, with real not county-averaged
embeddings, are inside its training data) still falls **short** of the actual total
despite the circularity, which is itself worth the team's attention: even a model that
has directly seen Ian cannot fully reconstruct Ian's severity once its features are
flattened to a per-county average embedding and a generic month flag for the replay
scenario. That's evidence the gap isn't purely "hasn't seen a catastrophe" — some of it
is inherent to how coarse this replay's inputs are (one averaged embedding per county,
not per-claim detail).

Premium collected in the 32 Ian-affected counties (any zone): **$1.404B** — so even the
official M2 replay ($1.323B) is roughly the same order as one year of premium there,
while the actual historical Ian loss (~$4.3B) was about 3× annual premium.

No failures. No decision values touched.

---

## What's waiting for the team

1. **RUNBOOK 5.2 itself** — all PROVISIONAL outputs above are waiting on this. The
   SHAP split-count evidence (Step 1) adds support to last night's hypothesis but
   doesn't resolve it.
2. **The "deployment fit" option** (`config.yaml: deployment_fit.enabled`,
   `docs/decisions.md`) — whether to adopt `m1_deploy` / `m2_deploy` for real
   phase-6 numbers, keep them as a side-by-side comparison only, or wait for a
   feature-level fix to M1/M2 first. Not something I could resolve unilaterally: a
   model fit on all available history has no genuine held-out test of its own
   forward-looking accuracy.
3. **Two small data-quality gaps**, noted but not fixed (Step 2): 11 policies with a
   non-Florida `censusGeoid`, and 7,616 policies (0.69%) with no `censusGeoid` at all.
4. **High-risk flood zone set for the Ian replay** (V, A, AH_AO) and the Ian claim
   window (Sept 23–Oct 7, 2022) are stated assumptions in `11_ian_replay.py` and its
   assumptions doc, not config decisions — worth a quick sign-off if the team wants
   them promoted to `config.yaml`.
5. `CLAUDE.md`'s "Current status" section was **intentionally left unchanged** —
   scripts 07–11 are now all implemented, but given the open item, I judged that
   updating "what's done" was the team's call to make, not mine, overnight.

## Reproducing tonight's run

```
pytest -q                     # 86 passed
python src/08_explain.py      # ~15s
python src/09_frequency.py    # ~1s
python src/10_shortfall.py    # ~50s (downloads Census boundaries once, then cached)
python src/11_ian_replay.py   # ~45s
```
All four are idempotent and safe to rerun. `data/models/m1_deploy_lgbm.joblib` and
`m2_deploy_lgbm.joblib` are produced by steps 3 and 4 respectively (gitignored, not in
this diff).
