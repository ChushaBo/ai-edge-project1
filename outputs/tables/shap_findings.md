# SHAP findings for M1 (RUNBOOK 5.3)

**PROVISIONAL -- pending RUNBOOK 5.2.** M1 has an open item in `docs/decisions.md` (it underpredicts the test portfolio total by 57.6%). These findings are evidence toward that open item, not a final explanation.

## Does latitude/longitude dominate SHAP?

**Partially, and the two metrics disagree.** By SHAP impact magnitude on this 3,000-row sample, latitude + longitude are **22.8%** of total mean |SHAP| (latitude individually ranks #2 of 20, longitude #10) -- real, but `totalBuildingInsuranceCoverage` ranks #1 individually, so lat/long are not an outright majority of SHAP impact. By tree split count, though, they clearly do dominate: **31.4%** of every split M1 makes is on latitude or longitude, more than any other feature by a wide margin (see the model-training log). The two metrics measure different things: SHAP impact is averaged over this representative sample of mostly-typical claims, while split count reflects how finely the tree partitions space -- which is exactly the mechanism in the open-item hypothesis (fine spatial leaves that memorize a location's pre-2020 average and cannot extrapolate to a catastrophe there). The hypothesis is about structure, not average impact on typical claims, so the split-count evidence is the more direct support for it.

## Top features by mean |SHAP| (share-of-coverage units)

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | totalBuildingInsuranceCoverage | 0.04012 |
| 2 | latitude | 0.03423 |
| 3 | buildingDeductibleCode | 0.01850 |
| 4 | elevationDifference | 0.00993 |
| 5 | building_age_at_loss | 0.00969 |
| 6 | elevatedBuildingIndicator | 0.00960 |
| 7 | flood_zone_group | 0.00957 |
| 8 | postFIRMConstructionIndicator | 0.00830 |
| 9 | foundationType | 0.00764 |
| 10 | longitude | 0.00763 |

## Dependence (top 5, plus elevationDifference)

See `outputs/figures/shap_dependence_<feature>.png` and `outputs/tables/shap_top_effects_dollars.csv` for the dollar translation (SHAP value x $250,000) at low/mid/high levels of each feature.

