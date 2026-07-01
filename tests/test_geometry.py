"""Unit tests for geometry helper functions."""

from app.utils.geometry import build_bounding_box, calculate_bbox_area, calculate_bbox_center


def test_calculate_bbox_area_returns_expected_value() -> None:
    """Verify that the bounding-box area is computed correctly."""

    bbox = build_bounding_box(10, 20, 30, 50)
    assert calculate_bbox_area(bbox) == 600


def test_calculate_bbox_center_returns_expected_point() -> None:
    """Verify that the bounding-box center is computed correctly."""

    bbox = build_bounding_box(10, 20, 30, 50)
    assert calculate_bbox_center(bbox) == (20.0, 35.0)
