"""Configuration helpers for the vehicle closest-frame service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Store runtime settings for the FastAPI application."""

    storage_root: Path = Path("data/jobs")
    yolo_config_root: Path = Path("data/runtime")
    model_name: str = "yolo11n.pt"
    tracker_config: str = "bytetrack.yaml"
    allowed_classes: tuple[int, ...] = field(default_factory=lambda: (2,))
    save_annotated_video: bool = True


def get_settings() -> Settings:
    """Return the default application settings."""

    return Settings()
