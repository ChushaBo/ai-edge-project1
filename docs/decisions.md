# Decisions log

Human decisions for the Florida flood shortfall project. Every value marked
`# DECISION:` in `config.yaml` gets an entry here when it is set or changed.
Newest entries at the top. Do not delete old entries; add a new one that supersedes them.

## Template

Copy this block for each decision.

```
### YYYY-MM-DD — <short title>
- Decided by: <name(s)>
- config.yaml keys: <e.g. splits.train_end, splits.validation_end>
- Decision: <the value(s) chosen>
- Options considered: <alternatives>
- Rationale: <why; cite data, row counts, or dictionary entries>
- Follow-ups / checks: <anything to revisit>
```

## Log

<!-- Add entries below, newest first. -->

### 2026-11-15 — Claims year range
- Decided by: Bohdan Chushak
- config.yaml keys: years.claims_start, years.claims_end
- Decision: Include Florida NFIP claims with year of loss 2010–2024; exclude 2025 onward.
- Options considered: Start in 2000 for more data (adds Wilma 2005 and others); full history back to 1978; end at 2023 if 2024 claims look incomplete.
- Rationale: 15 years includes several major Florida flood events in training (Hermine 2016, Irma 2017, Michael 2018) while avoiding older claims that reflect outdated housing stock, building codes, and flood maps. NOAA narratives are more consistent in recent years. 2025+ claims are more likely to still be open, so recorded payouts would understate final costs. Row counts (step 1.1, NfipClaims v3, downloaded 2026-11-15): 215,099 Florida claims, 2010–2024. By year: 2010: 910; 2011: 2,157; 2012: 5,369; 2013: 2,154; 2014: 3,960; 2015: 1,993; 2016: 8,838; 2017: 31,430; 2018: 5,100; 2019: 1,193; 2020: 9,992; 2021: 933; 2022: 50,324; 2023: 10,300; 2024: 80,446.
- Follow-ups / checks: Helene and Milton (late 2024) claims may still be developing. Check for a claim status field in the data dictionary; if none, compare 2024 payout totals to publicly reported figures. If 2024 looks incomplete, end the test period at 2023 or label 2024 results provisional.

### 2026-11-15 — Train/validation/test split
- Decided by: Bohdan Chushak
- config.yaml keys: splits.train_end, splits.validation_end (test runs through years.claims_end)
- Decision: Split by date of loss. Train: 2010–2019. Validation: 2020–2021. Test: 2022–2024.
- Options considered: Random split (rejected: same storm would appear in train and test, inflating results); leave-one-storm-out cross-validation; different cutoff years.
- Rationale: A time-based split mirrors real use, where the model predicts future storms from past ones. The test period contains Ian, Nicole, Idalia, Debby, Helene, and Milton, a hard and realistic test that aligns with the Ian replay. Rows per split: [fill after step 2.2].
- Follow-ups / checks: 2020–2021 were relatively quiet years, so validation may be small and unlike test storms; confirm validation row count is adequate. Ian's surge exceeded most training events, so the test partly measures extrapolation. If time allows, run leave-one-storm-out validation as a robustness check; otherwise note it as a limitation.

### 2026-11-15 — Claim-to-storm match window
- Decided by: Bohdan Chushak
- config.yaml keys: match_window.days_before, match_window.days_after
- Decision: Match storm events whose begin date is within ±3 days of the claim's date of loss, in the same county.
- Options considered: ±1 day (fewer false matches, more missed matches); ±5 or ±7 days (more matches, more noise).
- Rationale: NFIP date of loss and NOAA event begin dates don't always align; hurricanes and their flooding span several days, and rain-driven flooding can lag. Three days captures most legitimate matches without attaching unrelated summer flash-flood reports, which are frequent in Florida.
- Follow-ups / checks: Review overall and per-year match rate and the 20-claim spot check in step 3.2. Widen if the match rate is below ~70% after fixing zone mapping; narrow if spot checks show unrelated storms. Match rate: [fill after step 3.1].

### 2026-11-15 — Storm event types
- Decided by: Bohdan Chushak
- config.yaml keys: storm_event_types
- Decision: Flood, Flash Flood, Coastal Flood, Storm Surge/Tide, Heavy Rain, Hurricane, Tropical Storm.
- Options considered: Add Tropical Depression; include Thunderstorm Wind, Tornado, Hail (rejected).
- Rationale: The first four are direct causes of flood claims; Coastal Flood also covers tidal "sunny-day" flooding. Hurricane and Tropical Storm narratives often describe overall flooding even when separate flood events exist. Heavy Rain is included because rain flooding is not always logged as a Flood event. Wind, tornado, and hail damage is not covered by flood insurance, so that text would be irrelevant.
- Follow-ups / checks: Consider adding Tropical Depression, which can cause significant rain flooding. Check rows per event type in Florida after step 1.3: [fill after step 1.3].

### 2026-11-15 — Text embedding model
- Decided by: Bohdan Chushak
- config.yaml keys: embedding.model
- Decision: sentence-transformers all-MiniLM-L6-v2 (384 dimensions).
- Options considered: all-mpnet-base-v2 (better quality, several times slower); fine-tuning a model (not feasible in one week).
- Rationale: Small, fast on a laptop CPU, widely used, solid general-purpose embeddings. The goal is to test whether storm descriptions add predictive value, not to build a state-of-the-art text model.
- Follow-ups / checks: The model reads only the first 256 tokens (~200 words) of each input and silently truncates the rest; combined hurricane narratives can be much longer. Update step 4.1 to embed each narrative separately and average the vectors per claim. Try all-mpnet-base-v2 only if time remains. Embedding runtime: [fill after step 4.1].

### 2026-11-15 — PCA dimensions for text features
- Decided by: Bohdan Chushak
- config.yaml keys: pca.n_components
- Decision: Reduce embeddings to 24 dimensions with PCA fit on training rows only.
- Options considered: 8, 16, 32 dimensions; using all 384 dimensions (rejected: would swamp ~12–15 tabular features and invite overfitting).
- Rationale: Many claims share the same storm narrative, so the number of distinct narratives in training is far smaller than the number of claims, likely in the hundreds. Text features can only learn from those, so a small dimension count is appropriate. 24 is roughly twice the tabular feature count. Distinct training narratives: [fill after step 3.1]. Explained variance: [fill after step 4.1].
- Follow-ups / checks: Compare 8, 16, and 32 on the validation set and keep the best performer.

### 2026-11-15 — Random seed
- Decided by: Bohdan Chushak
- config.yaml keys: random_seed
- Decision: 42, applied to numpy, scikit-learn, LightGBM, and torch.
- Options considered: Any fixed integer; averaging over multiple seeds.
- Rationale: The value is arbitrary; fixing it makes results reproducible from a fresh clone, which the syllabus requires for the technical repository.
- Follow-ups / checks: If time allows, rerun final models with 3–5 seeds and report whether metrics are stable.

### 2026-11-15 — Policy snapshot for shortfall analysis
- Decided by: Bohdan Chushak
- config.yaml keys: years.policies_year
- Decision: Florida NFIP policies in force during the most recent complete year available.
- Options considered: Policies in force on a snapshot date (e.g. 2025-09-30); full-year 2024; multi-year average.
- Rationale: The shortfall should reflect today's policies and pricing. FEMA phased in Risk Rating 2.0 starting in late 2021, so recent premiums reflect the current pricing system. Recent data also captures Florida's growing flood coverage requirements for Citizens policyholders. Policy count: [fill after step 1.2].
- Follow-ups / checks: The NFIP lapsed in October 2025, which may have stalled new policies and renewals and distorted full-year 2025 counts. Check monthly policy counts for a dip around October 2025; if present, switch to a pre-lapse snapshot date such as 2025-09-30.
