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

### 2026-11-15 — Match quality review (RUNBOOK 3.2)
- Decided by: Bohdan Chushak
- config.yaml keys: match_window.days_before, match_window.days_after (unchanged, ±3 days)
- Decision: Match quality accepted. All 20 spot-check claims marked plausible ("yes" in outputs/tables/match_spot_check.csv).
- Options considered: Narrowing or widening the window; not needed.
- Rationale: Each narrative describes the storm that hit the claim's county on its loss date (Rina remnants 2011, Matthew 2016, Irma 2017, Michael 2018, Sally and Eta 2020, Ian and Nicole 2022, Idalia and an April 2023 rain flood, Helene and Milton 2024). Match rate 96.7% excluding quarantined claims, above the ~70% threshold.
- Follow-ups / checks: The Helene quarantine (entry below) is still open.

### 2026-11-15 — Quarantine Helene claims with no NOAA narrative (temporary)
- Decided by: Bohdan Chushak
- config.yaml keys: quarantine.enabled, quarantine.rules
- Decision: Flag, but keep, the 46,971 claims with loss dates 2024-09-24 to 2024-09-30 that have no matched narrative (quarantined = true in claims_with_text.parquet). Steps 06+ skip them while quarantine.enabled is true, for both M1 and M2. To be revisited.
- Options considered: Accept the gap and also score M2 on matched claims only; attach Helene narratives from other Florida counties on the same dates (flagged); drop the claims from M2 only.
- Rationale: NOAA's 2024 details file (c20260728) has no Helene events for the Tampa Bay counties (Pinellas, Hillsborough, Manatee, Pasco, Sarasota), though it has their Milton events (100% matched). Widening the match window would attach Milton narratives (13 days later) to Helene claims. Quarantined claims are mostly Pinellas 20,700, Hillsborough 6,729, Manatee 4,647, Pasco 3,952, Sarasota 3,872.
- Follow-ups / checks: Revisit before step 5.1. Removing 46,971 of 112,573 test claims changes the test set (now 65,602) and drops much of Helene's loss, so results must say so. Check whether NOAA publishes a newer 2024 file with these events.

### 2026-11-15 — Narrative matching results (RUNBOOK 3.1)
- Decided by: Bohdan Chushak
- config.yaml keys: match_window.days_before, match_window.days_after (unchanged, ±3 days)
- Decision: Keep the ±3-day window. Zone-coded events map to counties through the NWS zone files, checked against the county named in CZ_NAME; renumbered zones fall back to the name.
- Options considered: Widening the window (rejected: the low 2024 rate is missing NOAA data, not a window problem).
- Rationale: Of 1,450 zone-coded flood events, 1,265 map by zone (zone and name agree in every case where both exist), 183 by name (zone number not in current files) and 2 by zone only. Match rate 68.1% overall (108,244 of 158,908); 96.7% excluding the quarantined Helene claims. Train 93.9%, validation 95.5%, test 57.4% (98.5% excluding quarantine). By year, 75–99% except 2010 (41%, 261 claims) and 2024 (28.8%, the Helene gap).
- Follow-ups / checks: Human spot check of 20 claims (RUNBOOK 3.2): outputs/tables/match_spot_check.csv, column "plausible".

### 2026-11-15 — Leakage sign-off on the feature table (RUNBOOK 2.3)
- Decided by: Bohdan Chushak
- config.yaml keys: allowed_features, derived_features, month_of_loss_models
- Decision: Accept all 21 feature columns in data/interim/claims_clean.parquet: 16 raw allowed fields plus 5 derived (building_age_at_loss, policy_tenure_years, flood_zone_group, occupancy_group, month_of_loss). M1 uses 20 (no month_of_loss); M2 may use all 21.
- Options considered: Dropping fields that can change because of a flood: elevatedBuildingIndicator, elevationDifference, policy_tenure_years (strongest candidates), policyCount, buildingDeductibleCode.
- Rationale: Each is a property or policy attribute known before a flood, reviewed line by line against "would we know this before the flood?". Known caveat: claim records carry later-updated policy attributes (Risk Rating 2.0 codes on 2016–2019 losses; 2,418 claims with a policy start date after the loss), so some values may describe the property after its flood. Accepted and to be stated as a limitation.
- Follow-ups / checks: If M1's SHAP results (step 5.3) lean heavily on elevatedBuildingIndicator, elevationDifference or policy_tenure_years, revisit this sign-off. tests/test_leakage.py enforces the allowed list.

### 2026-11-15 — Field review sign-off (RUNBOOK 1.4)
- Decided by: Bohdan Chushak
- config.yaml keys: fields.*, allowed_features, month_of_loss_models, flood_zone_groups, occupancy_groups
- Decision: Field names confirmed against the saved v3 dictionaries (worksheet: docs/field_review_1_4.md). Claims: payout netBuildingPaymentAmount, coverage totalBuildingInsuranceCoverage, date dateOfLoss, county countyCode, zone ratedFloodZone. Policies: coverage totalBuildingInsuranceCoverage, county = first 5 digits of censusGeoid, zone ratedFloodZone. Features: "Core 20" (zone group, building and contents coverage, occupancy group, floors, foundation, elevated, post-FIRM, construction date, original policy date, primary residence, rental, condo coverage type, policy count, CRS class, building deductible, basement type, elevation difference, latitude, longitude). Month of loss in M2 only. Flood zones grouped V / A / AH_AO / X / D; occupancy grouped single unit / 2–4 units / 5+ or condo / non-residential.
- Options considered: Core 18 (without basement type and elevation difference, 54–71% missing); minimal 8; amountPaidOnBuildingClaim as payout (missing on 14.9% of rows); raw occupancy codes; month of loss in both models or neither.
- Rationale: Every feature is known before the flood and present on the policies file, so M1 can price 2025 policies. Excluded as judgment calls: buildingReplacementCost (missing on exactly the rows where buildingDamageAmount is missing, so likely filled in by the adjuster), rateMethod and floodZoneCurrent (reflect current, not at-loss, rating and maps), asOfDate (metadata). Occupancy codes need grouping because training years mostly use legacy codes 1–6 and test years Risk Rating 2.0 codes 11–19.
- Follow-ups / checks: Claim records carry later policy attributes (2-digit occupancy codes appear on 2016–2019 losses, before Risk Rating 2.0 began in Oct 2021; "RE" rate method on pre-2021 losses). Features may describe the policy after the loss, not at it. Note as a limitation. Leakage sign-off of the final feature table is still due in RUNBOOK 2.3.

### 2026-11-15 — Target definition and claim cleaning (RUNBOOK 2.1)
- Decided by: Bohdan Chushak
- config.yaml keys: target.definition, target.drop_zero_paid, target.drop_zero_coverage, target.cap_share_at, cleaning.drop_exact_duplicate_claims
- Decision: Target = building payout / building coverage. Drop $0-paid claims (26.1%), drop claims with $0 building coverage (5,614), cap payouts above coverage at 1.0 (468 claims). Keep claim rows that share property and loss details (separate buildings); drop exact copies (789 rows identical in every column except id and asOfDate; keep one of each).
- Options considered: Log payout in 2025 dollars (needs CPI); keep $0 claims; drop over-coverage claims instead of capping; keep or collapse matching rows.
- Rationale: Share of coverage needs no inflation adjustment and has the coverage cap built in. Claim frequency is estimated separately (step 6.1), so the payout model predicts size given a paid claim. Of 14,121 rows that share loss date, location, building and coverage details, 49% are condo associations (code 15) and 74% of groups have different payouts per row, so they are separate buildings in the same complex, not one claim split across rows.
- Follow-ups / checks: Step 2.2 logs each drop count.

### 2026-11-15 — Premium field and cancelled policies for the shortfall
- Decided by: Bohdan Chushak
- config.yaml keys: fields.policies_premium, fields.policies_premium_comparisons, cleaning.exclude_cancelled_policies
- Decision: Headline shortfall uses totalInsurancePremiumOfThePolicy ($1.58B for Florida 2025 terms). The shortfall table also reports policyCost ($2.03B) and fullRiskPremium ($2.72B). Exclude cancelled terms (42,185, 3.7%) from both premium and exposure.
- Options considered: policyCost or fullRiskPremium as the headline; pro-rating or keeping cancelled terms.
- Rationale: policyCost = premium + reserve fund assessment + federal policy fee + HFIAA surcharge (verified in 100% of rows); the federal fee mostly funds administration. fullRiskPremium is FEMA's price for full risk, not money collected, so it is a benchmark. Excluding cancelled terms keeps premium and exposure consistent.
- Follow-ups / checks: Shortfall write-up should show all three premium measures.

### 2026-11-15 — Policy year: 2025, terms starting that year (update)
- Decided by: Bohdan Chushak
- config.yaml keys: years.policies_year
- Decision: 2025. A policy counts if its policyEffectiveDate falls in 2025 (one row per annual term), so each policy is counted once with one annual premium.
- Options considered: 2024 (last full pre-lapse year); in-force snapshot on 2025-09-30; policies in force at any time in 2025 (most policies would appear twice, the expiring term plus its renewal, needing premium pro-rating).
- Rationale: Lapse check from the OpenFEMA NfipPolicies v3 API (Florida policy terms starting per month, queried 2026-11-15). Terms started Oct–Dec 2025 were 6%, 10% and 15% below the same months of 2024, but the full year was only 1.3% lower (1,146,045 in 2025 vs 1,161,505 in 2024). The lapse dip is small over the year, so the most recent complete year stands. Monthly 2026 counts run about 5–6% below 2025, a general downward trend.
- Follow-ups / checks: Mention the Q4 2025 dip as a caveat on the shortfall results. Supersedes the follow-up in the "Policy snapshot" entry below.

### 2026-11-15 — Storm event types: NOAA spelling of Hurricane (correction)
- Decided by: Bohdan Chushak
- config.yaml keys: storm_event_types
- Decision: Replace "Hurricane" with "Hurricane (Typhoon)". No change in intent.
- Options considered: Leave as is (would silently exclude every hurricane event).
- Rationale: Step 1.3 found that NOAA's Florida details files spell the type "Hurricane (Typhoon)" (122 rows, 2010–2024); "Hurricane" matched 0 rows. The other six types matched exactly: Flood 932, Flash Flood 969, Coastal Flood 308, Storm Surge/Tide 180, Heavy Rain 1,230, Tropical Storm 840.
- Follow-ups / checks: Tropical Depression (123 rows) is still excluded; see the "Storm event types" entry below.

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
- Follow-ups / checks: Review overall and per-year match rate and the 20-claim spot check in step 3.2. Widen if the match rate is below ~70% after fixing zone mapping; narrow if spot checks show unrelated storms. Match rate (step 3.1): 68.1% overall, 96.7% excluding quarantined Helene claims; see the narrative matching entry.

### 2026-11-15 — Storm event types
- Decided by: Bohdan Chushak
- config.yaml keys: storm_event_types
- Decision: Flood, Flash Flood, Coastal Flood, Storm Surge/Tide, Heavy Rain, Hurricane, Tropical Storm.
- Options considered: Add Tropical Depression; include Thunderstorm Wind, Tornado, Hail (rejected).
- Rationale: The first four are direct causes of flood claims; Coastal Flood also covers tidal "sunny-day" flooding. Hurricane and Tropical Storm narratives often describe overall flooding even when separate flood events exist. Heavy Rain is included because rain flooding is not always logged as a Flood event. Wind, tornado, and hail damage is not covered by flood insurance, so that text would be irrelevant.
- Follow-ups / checks: Consider adding Tropical Depression, which can cause significant rain flooding. Rows per event type in Florida (step 1.3): Flood 932; Flash Flood 969; Coastal Flood 308; Storm Surge/Tide 180; Heavy Rain 1,230; Hurricane (Typhoon) 122; Tropical Storm 840; Tropical Depression 123 (not included).

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
- Rationale: Many claims share the same storm narrative, so the number of distinct narratives in training is far smaller than the number of claims, likely in the hundreds. Text features can only learn from those, so a small dimension count is appropriate. 24 is roughly twice the tabular feature count. Distinct training narratives (step 3.1): 525. Explained variance: [fill after step 4.1].
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
- Rationale: The shortfall should reflect today's policies and pricing. FEMA phased in Risk Rating 2.0 starting in late 2021, so recent premiums reflect the current pricing system. Recent data also captures Florida's growing flood coverage requirements for Citizens policyholders. Policy count (step 1.2, terms effective in 2025): 1,146,045 policy terms covering 1,807,343 insured units (condo association policies cover several units).
- Follow-ups / checks: The NFIP lapsed in October 2025, which may have stalled new policies and renewals and distorted full-year 2025 counts. Check monthly policy counts for a dip around October 2025; if present, switch to a pre-lapse snapshot date such as 2025-09-30.
