"""Compare expected annual loss with premium by county (M1, tabular only).

Inputs:
  - data/raw/policies_fl.parquet
  - data/models/ (trained M1)
  - outputs/tables/frequency_by_county_zone.csv
  - Census cartographic boundary file for Florida counties (network on first run)
  - config.yaml: fields.policies_*, paths.*
Outputs:
  - outputs/tables/shortfall_by_county.csv (policies, premium, expected loss, gap,
    loss ratio; statewide total row)
  - outputs/figures/ Florida county choropleth of loss ratio
"""

from utils.config import load_config
from utils.log import get_logger

logger = get_logger("10_shortfall")


def main() -> None:
    cfg = load_config()  # noqa: F841 (used once implemented)
    raise NotImplementedError("src/10_shortfall.py is a stub; implement per RUNBOOK 6.2.")


if __name__ == "__main__":
    main()
