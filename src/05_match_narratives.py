"""Match each claim to NOAA storm events and attach combined narratives.

Inputs:
  - data/interim/claims_clean.parquet
  - data/raw/storms_fl.parquet
  - data/raw/ NWS zone-to-county correlation file
  - config.yaml: match_window.*, storm_event_types, paths.*
Outputs:
  - data/interim/claims_with_text.parquet (adds combined narrative,
    n_matched_events, matched_event_types, has_narrative)
  - outputs/tables/match_spot_check.csv (20 random matched claims)
"""

from utils.config import load_config
from utils.log import get_logger

logger = get_logger("05_match_narratives")


def main() -> None:
    cfg = load_config()  # noqa: F841 (used once implemented)
    raise NotImplementedError("src/05_match_narratives.py is a stub; implement per RUNBOOK 3.1.")


if __name__ == "__main__":
    main()
