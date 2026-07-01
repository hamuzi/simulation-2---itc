"""Unit tests for closest-frame selection logic."""

from app.domain import FrameEvidence, VehicleTrack
from app.utils.geometry import build_bounding_box, calculate_bbox_area, calculate_bbox_center


def _build_evidence(frame_index: int, bbox_values: tuple[float, float, float, float], confidence: float) -> FrameEvidence:
    """Create one frame evidence object for test assertions."""

    bbox = build_bounding_box(*bbox_values)
    return FrameEvidence(
        frame_index=frame_index,
        timestamp_sec=frame_index / 10.0,
        bbox=bbox,
        center=calculate_bbox_center(bbox),
        bbox_area=calculate_bbox_area(bbox),
        confidence=confidence,
    )


def test_vehicle_track_prefers_larger_area() -> None:
    """Verify that the closest frame is the one with the largest area."""

    vehicle = VehicleTrack(track_id=7)
    smaller = _build_evidence(0, (0, 0, 10, 10), 0.70)
    larger = _build_evidence(1, (0, 0, 20, 20), 0.60)

    vehicle.add_evidence(smaller)
    vehicle.add_evidence(larger)

    assert vehicle.closest_frame == larger


def test_vehicle_track_uses_y2_as_tie_breaker() -> None:
    """Verify that the lower-on-screen box wins when areas are equal."""

    vehicle = VehicleTrack(track_id=9)
    first = _build_evidence(0, (0, 0, 10, 20), 0.80)
    second = _build_evidence(1, (0, 5, 10, 25), 0.70)

    vehicle.add_evidence(first)
    vehicle.add_evidence(second)

    assert vehicle.closest_frame == second


def test_vehicle_track_uses_confidence_then_frame_index_in_ties() -> None:
    """Verify that confidence and later frame index break complete ties."""

    vehicle = VehicleTrack(track_id=11)
    first = _build_evidence(0, (0, 0, 10, 10), 0.85)
    second = _build_evidence(1, (0, 0, 10, 10), 0.90)
    third = _build_evidence(2, (0, 0, 10, 10), 0.90)

    vehicle.add_evidence(first)
    vehicle.add_evidence(second)
    vehicle.add_evidence(third)

    assert vehicle.closest_frame == third
