"""Estimate annual claim frequency per policy by county and flood zone.

Inputs:
  - data/interim/claims_clean.parquet (paid claims, numerator)
  - data/raw/policies_fl.parquet (policies-in-force counts, denominator)
  - config.yaml: fields.*, years.*, paths.*
Outputs:
  - outputs/tables/frequency_by_county_zone.csv (smoothed rates; method documented
    here once implemented)
"""

from utils.config import load_config
from utils.log import get_logger

logger = get_logger("09_frequency")


def main() -> None:
    cfg = load_config()  # noqa: F841 (used once implemented)
    raise NotImplementedError("src/09_frequency.py is a stub; implement per RUNBOOK 6.1.")


if __name__ == "__main__":
    main()
