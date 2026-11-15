"""Train and evaluate B0, B1, M1, M2, M2_masked.

Inputs:
  - data/interim/claims_with_text.parquet
  - data/processed/text_features.parquet
  - config.yaml: allowed_features, target.*, splits.*, random_seed, paths.*
Outputs:
  - data/models/ (trained models)
  - outputs/tables/metrics_overall.csv (MAE, RMSE in target units and dollars)
  - outputs/tables/metrics_by_event.csv (predicted vs. actual total payout per event)
  - outputs/figures/ MAE comparison bar chart
Tuning uses validation only; the test set is evaluated once.
"""

from utils.config import load_config
from utils.log import get_logger

logger = get_logger("07_train_models")


def main() -> None:
    cfg = load_config()  # noqa: F841 (used once implemented)
    raise NotImplementedError("src/07_train_models.py is a stub; implement per RUNBOOK 5.1.")


if __name__ == "__main__":
    main()
