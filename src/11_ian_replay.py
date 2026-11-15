"""Replay Hurricane Ian (2022) on current policies using M2.

Inputs:
  - data/interim/claims_with_text.parquet (Ian-matched narratives, Sept-Oct 2022)
  - data/processed/text_features.parquet
  - data/raw/policies_fl.parquet
  - data/models/ (trained M2)
  - config.yaml: fields.*, paths.*
Outputs:
  - outputs/tables/ Ian replay table (predicted payout vs. annual premium by county)
  - outputs/figures/ headline chart
  - outputs/tables/ian_replay_assumptions.md
All assumptions are to be stated here once implemented.
"""

from utils.config import load_config
from utils.log import get_logger

logger = get_logger("11_ian_replay")


def main() -> None:
    cfg = load_config()  # noqa: F841 (used once implemented)
    raise NotImplementedError("src/11_ian_replay.py is a stub; implement per RUNBOOK 6.3.")


if __name__ == "__main__":
    main()
