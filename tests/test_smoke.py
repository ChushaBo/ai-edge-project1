"""Smoke test: the scaffold is wired together.

Checks that config.yaml parses with every required section, the shared helpers
import and work, and every numbered pipeline script is valid Python whose
docstring states its inputs and outputs.
"""

import ast
import logging
from pathlib import Path

from utils.config import REPO_ROOT, load_config, require
from utils.log import get_logger, log_rows

REQUIRED_SECTIONS = {
    "paths", "years", "splits", "target", "fields", "allowed_features",
    "match_window", "storm_event_types", "embedding", "pca", "random_seed",
}
EXPECTED_SCRIPTS = [
    "01_download_claims.py", "02_download_policies.py", "03_download_storms.py",
    "04_clean_claims.py", "05_match_narratives.py", "06_embed_text.py",
    "07_train_models.py", "08_explain.py", "09_frequency.py", "10_shortfall.py",
    "11_ian_replay.py",
]


def test_scaffold_smoke(caplog):
    cfg = load_config()
    assert REQUIRED_SECTIONS <= cfg.keys()
    assert require(cfg, "paths.raw") == "data/raw"

    logger = get_logger("smoke")
    logger.propagate = True  # let caplog see it
    with caplog.at_level(logging.INFO, logger="smoke"):
        assert log_rows(logger, [1, 2, 3], "smoke step") == 3
    assert "smoke step: 3 rows" in caplog.text

    src = REPO_ROOT / "src"
    for name in EXPECTED_SCRIPTS:
        tree = ast.parse((src / name).read_text(encoding="utf-8"), filename=name)
        doc = ast.get_docstring(tree) or ""
        assert "Inputs:" in doc and "Outputs:" in doc, f"{name} docstring incomplete"

    for script in Path(src).glob("[0-9][0-9]_*.py"):
        assert script.name in EXPECTED_SCRIPTS, f"unexpected script {script.name}"
