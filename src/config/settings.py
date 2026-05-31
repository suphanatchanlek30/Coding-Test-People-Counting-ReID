from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils import load_config


@dataclass(frozen=True)
class AppSettings:
    config_path: Path
    data: dict[str, Any]

    @property
    def output_dir(self) -> Path:
        return Path("outputs")


def load_settings(config_path: str | Path) -> AppSettings:
    path = Path(config_path)
    config = load_config(path)
    _validate_required_sections(config)
    return AppSettings(config_path=path, data=config)


def _validate_required_sections(config: dict[str, Any]) -> None:
    required_sections = [
        "video",
        "detection",
        "tracking",
        "reid",
        "counting",
        "attribute",
        "export",
    ]

    missing = [section for section in required_sections if section not in config]
    if missing:
        raise KeyError(f"Missing required config section(s): {', '.join(missing)}")