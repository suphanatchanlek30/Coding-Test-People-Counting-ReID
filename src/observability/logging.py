from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    log_dir: str | Path = "outputs/logs",
    level: int = logging.INFO,
) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "run.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )

    return logging.getLogger("people_analytics")