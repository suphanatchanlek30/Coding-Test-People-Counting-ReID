from __future__ import annotations

import argparse

from src.app.runner import PeopleAnalyticsApp
from src.config.settings import load_settings
from src.observability.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime entrance people analytics with tracking and ReID.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    settings = load_settings(args.config)
    PeopleAnalyticsApp(settings=settings, logger=logger).run()


if __name__ == "__main__":
    main()