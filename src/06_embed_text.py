"""Embed claim narratives (raw and damage-figure-masked) and reduce with PCA.

Inputs:
  - data/interim/claims_with_text.parquet
  - Pretrained sentence-transformers model (network on first run)
  - config.yaml: embedding.model, pca.n_components, random_seed, paths.*
Outputs:
  - data/processed/text_features.parquet (PCA components for raw and masked text,
    PCA fit on train rows only; zeros plus has_narrative for claims without text)
"""

from utils.config import load_config
from utils.log import get_logger

logger = get_logger("06_embed_text")


def main() -> None:
    cfg = load_config()  # noqa: F841 (used once implemented)
    raise NotImplementedError("src/06_embed_text.py is a stub; implement per RUNBOOK 4.1.")


if __name__ == "__main__":
    main()
