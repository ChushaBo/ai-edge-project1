"""Download NOAA Storm Events "details" files and the NWS zone-to-county file.

Inputs:
  - NOAA NCEI Storm Events bulk CSV directory (network), one details file per year in
    years.claims_start..years.claims_end.
  - NWS zone-to-county correlation file (network).
  - config.yaml: paths.raw, years.*
Outputs:
  - data/raw/storms_fl.parquet (STATE = FLORIDA; event/episode IDs, EVENT_TYPE, begin
    and end dates, CZ_TYPE, CZ_FIPS, CZ_NAME, DAMAGE_PROPERTY, episode and event
    narratives)
  - data/raw/ NWS zone-to-county correlation file
"""

from utils.config import load_config
from utils.log import get_logger

logger = get_logger("03_download_storms")


def main() -> None:
    cfg = load_config()  # noqa: F841 (used once implemented)
    raise NotImplementedError("src/03_download_storms.py is a stub; implement per RUNBOOK 1.3.")


if __name__ == "__main__":
    main()
