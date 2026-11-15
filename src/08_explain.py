"""Explain M1 with SHAP.

Inputs:
  - data/models/ (trained M1)
  - data/interim/claims_with_text.parquet (sample of test rows)
  - config.yaml: random_seed, paths.*
Outputs:
  - outputs/figures/ SHAP summary plot and dependence plots for the top 5 features
  - outputs/tables/ SHAP effects in dollars for a typical $250,000-coverage home
  - outputs/tables/shap_findings.md
"""

from utils.config import load_config
from utils.log import get_logger

logger = get_logger("08_explain")


def main() -> None:
    cfg = load_config()  # noqa: F841 (used once implemented)
    raise NotImplementedError("src/08_explain.py is a stub; implement per RUNBOOK 5.3.")


if __name__ == "__main__":
    main()
