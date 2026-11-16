# Hurricane Ian replay -- assumptions (RUNBOOK 6.3)

PROVISIONAL, pending RUNBOOK 5.2 -- see docs/decisions.md open item. M2 shares M1's
tabular features, so the (a) current-M2 replay total likely understates the true
figure for the same reason M1 underpredicts the test portfolio total.

## Ian-affected counties
32 Florida counties where a 2022 claim's matched storm narrative
contained "Ian" (36,646 claims): 12009, 12011, 12015, 12019, 12021, 12027, 12031, 12035, 12049, 12051, 12055, 12057, 12061, 12069, 12071, 12081, 12085, 12086, 12087, 12089, 12095, 12097, 12099, 12101, 12103, 12105, 12107, 12109, 12111, 12115, 12117, 12127.
Only policies in these counties are replayed.

## High-risk flood zones
V, A, AH_AO (the Special Flood Hazard Area groups in config.yaml
flood_zone_groups). X and D zones are assumed not to file a claim in this scenario.

## Historical Ian claim rate
claims with date of loss in 2022-09-23..2022-10-07, by county x
flood_zone_group, divided by 2025 policies in force in that cell. 2025 policy counts
stand in for 2022 exposure (same limitation as 09_frequency.py: no historical
policy-count file exists to do better).

## Scenario applied to current policies
Every 2025 policy in an Ian-affected county is assumed to file a claim with
probability = its cell's historical Ian rate, if its flood zone is high-risk (0 claim
probability otherwise). Its text input for M2 is its county's Ian embedding (the mean
embedding of that county's Ian-matched claims) and month_of_loss = 9
(September, Ian's landfall month) -- i.e. "if a storm like Ian hit this county's
current policies again."

## "Annual premium collected in those counties"
Sum of totalInsurancePremiumOfThePolicy for every 2025 policy in an Ian-affected
county, ANY flood zone (not just the high-risk-zone policies assumed to claim):
$1,403,737,748 (outputs/tables/ian_replay_by_county.csv's "ALL" row,
premium_total column) -- the base the predicted payout totals below are compared
against.

## Two severity versions (run and saved side by side)
- replay_m2: the official M2 (07_train_models.py, train+validation only).
- replay_m2_deploy_CIRCULAR: M2's tuned hyperparameters refit on ALL claims
  2010-2024 (train+validation+test, Helene-quarantined claims excluded per M2's
  established scope). This is CIRCULAR for the Ian replay specifically: Ian's own
  2022 claims, with their real (not county-averaged) embeddings, are inside this
  model's training data, so it has already seen the actual outcome it is being asked
  to "predict" for a similar future storm. Reported for comparison only.

## Comparison to publicly reported figures
- Our own data, Florida, claims whose narrative matched Ian: $4,360,963,518
  (36,646 claims).
- Our own data, Florida, all claims with loss date in the Ian window (matched or
  not): $4,321,708,493 (35,967 claims).
- FEMA's reported final tally (nationwide, overwhelmingly Florida): $4,900,000,000
  (48,776 claims). Source: FEMA press materials reporting NFIP's
  final Hurricane Ian payout (as summarized by Artemis.bm and FEMA.gov reporting,
  retrieved 2026-11-16); building-only vs. building+contents+ICC is not always
  specified in press figures, so treat this as approximate.
- The replay is a SCENARIO on today's (larger, more numerous) policy book, not a
  backtest of 2022 -- a replay total above the historical actual is not necessarily
  wrong, since Florida's NFIP book has grown since 2022. It is not a forecast.
