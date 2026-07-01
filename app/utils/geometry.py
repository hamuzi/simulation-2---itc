"""Geometry helper functions for bounding-box calculations."""

from __future__ import annotations

from app.domain import BoundingBox


def build_bounding_box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    """Create a bounding box with normalized float values."""

    return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))


def calculate_bbox_area(bbox: BoundingBox) -> float:
    """Return the area of a bounding box."""

    return bbox.area()


def calculate_bbox_center(bbox: BoundingBox) -> tuple[float, float]:
    """Return the center point of a bounding box."""

    return bbox.center()
