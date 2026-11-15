# Field review worksheet (RUNBOOK 1.4)

Prepared 2026-11-15 from the saved dictionaries (`docs/data_dictionaries/nfip_*_v3_fields.csv`)
and the downloaded data (claims: 215,099 FL rows, 2010–2024; policies: 1,146,045 FL terms
effective 2025). Recommendations are suggestions; a person decides. Record
each decision in `docs/decisions.md` and `config.yaml`.

"% miss" = percent missing in our Florida data.

## Part 1. Field names for `config.yaml` → `fields:`

| Key | Recommended field | % miss | Notes |
|---|---|---|---|
| `claims_paid` | `netBuildingPaymentAmount` | 0.0 | Equals `amountPaidOnBuildingClaim` in 99.8% of rows where both exist, but is never missing (the paid field is missing on 14.9% of rows, all with net = 0). Net subtracts recoveries. 26.1% of claims have net = 0; 36 are negative; 468 exceed building coverage. |
| `claims_coverage` | `totalBuildingInsuranceCoverage` | 0.0 | 5,614 claims have coverage = 0, so a share-of-coverage target is undefined for them. |
| `claims_date_of_loss` | `dateOfLoss` | 0.0 | Always agrees with `yearOfLoss`. Loaded as UTC midnight. |
| `claims_county` | `countyCode` | 0.04 | 5-digit FIPS (`12xxx`), all 67 counties. |
| `claims_flood_zone` | `ratedFloodZone` | 0.06 | Zone used to rate the policy. Needs grouping (Part 3). |
| `policies_premium` | **decide** | 0.0 | `totalInsurancePremiumOfThePolicy` ($1.58B), `policyCost` ($2.03B = premium + reserve fund + federal fee + HFIAA surcharge, verified in 100% of rows), or `fullRiskPremium` ($2.72B, FEMA's full-risk price; 0.8% missing). |
| `policies_coverage` | `totalBuildingInsuranceCoverage` | 0.0 | |
| `policies_county` | `censusGeoid` (first 5 digits) | 0.8 | Policies have no county field. 12-digit block group; 9,204 missing, 12 with out-of-state codes. |
| `policies_flood_zone` | `ratedFloodZone` | 0.0 | Same grouping as claims. |

## Part 2. Claims fields: candidate features vs. excluded

A feature for M1 must be (a) known before the flood and (b) present on the policies file,
because step 6.2 applies M1 to 2025 policies. Policy names differ for four fields (noted).

### Candidate features (pre-event, on policies)

| Claims field | % miss claims | % miss policies | Notes |
|---|---|---|---|
| `ratedFloodZone` | 0.1 | 0.0 | Core; also the B1 benchmark. Group zones (Part 3). |
| `totalBuildingInsuranceCoverage` | 0.0 | 0.0 | Also the target denominator if target = share of coverage. |
| `totalContentsInsuranceCoverage` | 0.0 | 0.0 | |
| `occupancyType` | 0.0 | 0.0 | Codes changed: claims mix 1-digit legacy codes (1, 3, 6, ...) and 2-digit Risk Rating 2.0 codes (11–19); policies are all 2-digit. Needs a mapping. |
| `numberOfFloorsInTheInsuredBuilding` | 0.1 | 0.0 | Policies: `numberOfFloorsInInsuredBuilding`. |
| `foundationType` | 0.0 | 0.0 | |
| `elevatedBuildingIndicator` | 0.0 | 0.0 | |
| `postFIRMConstructionIndicator` | 0.0 | 0.0 | `preFirmIndicator` is its near-inverse; use one. |
| `originalConstructionDate` | 0.6 | 0.8 | Derive building age (at loss for claims, at 2025 for policies). |
| `originalNBDate` | 0.0 | 0.1 | Derive policy tenure. |
| `primaryResidenceIndicator` | 0.0 | 0.0 | |
| `rentalPropertyIndicator` | 0.0 | 0.0 | |
| `condominiumCoverageTypeCode` | 0.1 | 0.0 | |
| `policyCount` | 0.0 | 0.0 | Insured units; >1 for condo association policies. |
| `crsClassCode` | 17.4 | 2.3 | Community Rating System discount class. |
| `buildingDeductibleCode` | 2.5 | 2.4 | |
| `basementEnclosureCrawlspaceType` | 54.2 | 64.5 | Heavy missingness. |
| `elevationDifference` | 57.8 | 70.5 | Elevation vs. base flood elevation; likely strong but mostly missing. Keep a "missing" flag if used. |
| `latitude`, `longitude` | 0.6 | 0.8 | Rounded to 0.1° (about 11 km). |
| `smallBusinessIndicatorBuilding`, `nonProfitIndicator`, `stateOwnedIndicator`, `agricultureStructureIndicator`, `houseWorship` | 0.0 | 0.0 | Rare flags; low value. Policies: `houseOfWorshipIndicator`. |

Also pre-event and on policies, but probably not worth using: `baseFloodElevation`,
`lowestFloorElevation`, `lowestAdjacentGrade` (57–71% missing; `elevationDifference`
summarises them), `elevationCertificateIndicator` (84/91% missing), `obstructionType`
(80/94%), `contentsDeductibleCode`, `locationOfContents` (contents, not building),
`buildingDescriptionCode`, `disasterAssistanceCoverageRequired`, NFIP community numbers
(high-cardinality geography).

### Needs a human judgment

| Field | Issue |
|---|---|
| `buildingReplacementCost` | Missing on exactly the same claims rows as `buildingDamageAmount`, which suggests an adjuster fills it in after the loss. Probably exclude. |
| `rateMethod` | 170k claims show "RE" (Risk Rating 2.0), including pre-2021 losses, so claim records may carry the policy's current attributes, not those at the time of loss. Exclude; it also mostly encodes the rating era. |
| `floodZoneCurrent` | Today's map zone, not the zone at the time of loss (23% missing on claims, 44% on policies). Exclude or use only for a sensitivity check. |
| Month of loss | RUNBOOK 2.2 says to derive it, but it isn't known before a flood and doesn't exist for policies. Fine for M2; for M1 it would need averaging over months in step 6.2. Decide whether M1 uses it. |
| `numberOfUnits`, `iccCoverage` | Pre-event, but not in the policies dataset, so M1 couldn't use them in step 6.2. |
| `asOfDate` | Record update date (metadata). Exclude. |

### Excluded (post-loss or identifiers; CLAUDE.md rule 3)

- Payments and recoveries: `amountPaidOn*` (except the target), `net*PaymentAmount`
  (except the target), `netIccPaymentAmount`, `total*Recovery`, `totalSalvageRecovery`,
  `mostRecentPaymentDate`, `mostRecentRecoveryDate`
- Damage and adjuster values: `buildingDamageAmount`, `contentsDamageAmount`,
  `buildingPropertyValue`, `contentsPropertyValue`, `contentsReplacementCost`,
  `replacementCostBasis`
- Flood conditions and cause: `causeOfDamage`, `waterDepth`, `exteriorWaterDepth`,
  `interiorWaterDepth`, `floodWaterDuration`, `floodCharacteristicsIndicator` (100% missing)
- Event identifiers: `eventDesignationNumber`, `floodEvent`, `ficoNumber`
- Claim handling: `nonPaymentReasonBuilding`, `nonPaymentReasonContents`, `openDate`
- Keys, dates and location used outside the model: `id`, `dateOfLoss`, `yearOfLoss`,
  `state`, `countyCode`, `censusGeoid`, `reportedZipCode`, `reportedCity` (unavailable)

## Part 3. Other decisions this review surfaces

1. **Flood zone grouping.** Claims include legacy numbered zones (A01–A30 ≈ AE, V1–V30 ≈ VE)
   and B/C (≈ X under older maps). Suggested groups: V (V, VE, V1–30), A (A, AE, A1–30,
   A99), AH/AO (including AHB), X (X, B, C), D.
2. **Occupancy mapping** from legacy 1-digit to Risk Rating 2.0 codes (e.g. 1 → 11 single
   family). Check the legal values in the dictionary description.
3. **Premium field** for the shortfall (Part 1).
4. **Cancelled policy terms**: 42,185 (3.7%) have a cancellation date. Drop, keep, or
   pro-rate premium.
5. **Possible repeated claims**: the dataset describes itself as claims "transactions";
   14,121 rows share loss date, location, building and coverage details. Decide whether
   these are one claim split across rows or separate units (e.g. condos) before 2.2.
6. **Claims with $0 coverage (5,614) or payout above coverage (468)**: exclude or cap.
