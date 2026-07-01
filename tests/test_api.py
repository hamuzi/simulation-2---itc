"""API tests for the vehicle closest-frame service."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.controllers.job_controller import JobController
from app.domain import AnalysisResult, BoundingBox, FrameEvidence, VehicleTrack
from app.logic.job_logic import JobLogic
from app.services.analysis import VideoAnalyzer
from main import create_app


class FakeAnalyzer(VideoAnalyzer):
    """Provide deterministic analysis results without loading YOLO or OpenCV."""

    def __init__(self, settings: Settings) -> None:
        """Store settings for compatibility with the real analyzer interface."""

        super().__init__(settings)

    def analyze_video(self, source_video: Path, job_dir: Path) -> AnalysisResult:
        """Return a fixed analysis payload and write simple text artifacts."""

        artifacts_dir = job_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        vehicle = VehicleTrack(track_id=5)
        first = self._build_evidence(0, 0.0, BoundingBox(0, 0, 20, 20), 0.92)
        second = self._build_evidence(1, 0.1, BoundingBox(5, 5, 35, 35), 0.94)
        vehicle.add_evidence(first)
        vehicle.add_evidence(second)

        closest_image = artifacts_dir / "vehicle_5_closest.jpg"
        closest_image.write_text("fake image", encoding="utf-8")
        frames_csv = artifacts_dir / "vehicle_5_frames.csv"
        frames_csv.write_text("frame_index,bbox_area\n0,400\n1,900\n", encoding="utf-8")
        summary_csv = artifacts_dir / "vehicles_summary.csv"
        summary_csv.write_text("track_id,max_bbox_area\n5,900\n", encoding="utf-8")

        return AnalysisResult(
            fps=10.0,
            total_frames=2,
            vehicles={5: vehicle},
            artifacts={
                "vehicles_summary_csv": "artifacts/vehicles_summary.csv",
                "vehicle_5_frames_csv": "artifacts/vehicle_5_frames.csv",
                "vehicle_5_closest_image": "artifacts/vehicle_5_closest.jpg",
            },
        )

    def _build_evidence(self, frame_index: int, timestamp_sec: float, bbox: BoundingBox, confidence: float) -> FrameEvidence:
        """Create frame evidence objects for the fake analyzer response."""

        return FrameEvidence(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            bbox=bbox,
            center=bbox.center(),
            bbox_area=bbox.area(),
            confidence=confidence,
        )


def test_job_lifecycle_endpoints(tmp_path: Path) -> None:
    """Verify the main API flow from upload to closest-frame retrieval."""

    settings = Settings(storage_root=tmp_path / "jobs", save_annotated_video=False)
    job_logic = JobLogic(settings=settings, analyzer=FakeAnalyzer(settings))
    job_controller = JobController(job_logic=job_logic)
    client = TestClient(create_app(settings=settings, job_controller=job_controller))

    response = client.post(
        "/jobs",
        files={"video": ("road.mp4", b"fake-video-content", "video/mp4")},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    status_response = client.get(f"/jobs/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"
    assert status_response.json()["vehicle_count"] == 1

    vehicles_response = client.get(f"/jobs/{job_id}/vehicles")
    assert vehicles_response.status_code == 200
    assert vehicles_response.json()[0]["track_id"] == 5

    closest_response = client.get(f"/jobs/{job_id}/vehicles/5/closest-frame")
    assert closest_response.status_code == 200
    assert closest_response.json()["closest_frame_index"] == 1
    assert closest_response.json()["bbox_area"] == 900

    frames_response = client.get(f"/jobs/{job_id}/vehicles/5/frames")
    assert frames_response.status_code == 200
    frames_payload = frames_response.json()
    assert len(frames_payload) == 2
    assert frames_payload[1]["is_closest_frame"] is True

    artifact_response = client.get(f"/jobs/{job_id}/artifacts/vehicle_5_closest.jpg")
    assert artifact_response.status_code == 200


def test_missing_job_returns_404(tmp_path: Path) -> None:
    """Verify that unknown job IDs return a not-found response."""

    settings = Settings(storage_root=tmp_path / "jobs", save_annotated_video=False)
    job_logic = JobLogic(settings=settings, analyzer=FakeAnalyzer(settings))
    job_controller = JobController(job_logic=job_logic)
    client = TestClient(create_app(settings=settings, job_controller=job_controller))

    response = client.get("/jobs/unknown")
    assert response.status_code == 404


def test_missing_track_returns_404(tmp_path: Path) -> None:
    """Verify that unknown track IDs return a not-found response."""

    settings = Settings(storage_root=tmp_path / "jobs", save_annotated_video=False)
    job_logic = JobLogic(settings=settings, analyzer=FakeAnalyzer(settings))
    job_controller = JobController(job_logic=job_logic)
    client = TestClient(create_app(settings=settings, job_controller=job_controller))

    create_response = client.post(
        "/jobs",
        files={"video": ("road.mp4", b"fake-video-content", "video/mp4")},
    )
    job_id = create_response.json()["job_id"]

    response = client.get(f"/jobs/{job_id}/vehicles/999/closest-frame")
    assert response.status_code == 404
