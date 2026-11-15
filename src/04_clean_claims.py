"""Clean claims, build the target and feature table, and assign time-based splits.

Inputs:
  - data/raw/claims_fl.parquet
  - data/raw/ CPI-U annual averages CSV (only if target.definition needs CPI)
  - config.yaml: fields.*, target.*, allowed_features, splits.*, paths.*
Outputs:
  - data/interim/claims_clean.parquet (ID, county, date of loss, split, target, and
    ONLY the allowed features plus derived month of loss and building age at loss)
  - outputs/tables/claims_summary_by_split.csv (counts, target mean/median by split
    and flood zone)
"""

from utils.config import load_config
from utils.log import get_logger

logger = get_logger("04_clean_claims")


def main() -> None:
    cfg = load_config()  # noqa: F841 (used once implemented)
    raise NotImplementedError("src/04_clean_claims.py is a stub; implement per RUNBOOK 2.2.")


if __name__ == "__main__":
    main()
