"""Download Florida NFIP policies from OpenFEMA (FIMA NFIP Redacted Policies).

Inputs:
  - OpenFEMA API (network): FIMA NFIP Redacted Policies, filtered server-side to
    Florida and years.policies_year, with $select limited to needed columns.
  - config.yaml: paths.raw, paths.tables, paths.data_dictionaries, years.policies_year
Outputs:
  - data/raw/policies_fl.parquet
  - docs/data_dictionaries/ (policies data dictionary)
  - outputs/tables/policies_columns.csv (column, dtype, percent missing)
Flags:
  - --force: re-download even if policies_fl.parquet exists.
"""

from utils.config import load_config
from utils.log import get_logger

logger = get_logger("02_download_policies")


def main() -> None:
    cfg = load_config()  # noqa: F841 (used once implemented)
    raise NotImplementedError("src/02_download_policies.py is a stub; implement per RUNBOOK 1.2.")


if __name__ == "__main__":
    main()
