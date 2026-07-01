"""Business logic for asynchronous vehicle video analysis jobs."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from starlette import status

from app.config import Settings
from app.domain import AnalysisResult, JobRecord, VehicleTrack
from app.errors import LogicError
from app.services.analysis import VideoAnalyzer


class JobLogic:
    """Own the job lifecycle, persistence, and vehicle result lookups."""

    def __init__(self, settings: Settings, analyzer: VideoAnalyzer | None = None) -> None:
        """Store shared dependencies and ensure the job storage root exists."""

        self._settings = settings
        self._analyzer = analyzer or VideoAnalyzer(settings)
        self._storage_root = settings.storage_root
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create_job(self, source_video_bytes: bytes, source_video_name: str) -> JobRecord:
        """Create a pending job and persist the uploaded video to the filesystem."""

        job_id = uuid.uuid4().hex
        timestamp = self._utc_now()
        job_dir = self._storage_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        source_video_path = job_dir / "source_video.mp4"
        source_video_path.write_bytes(source_video_bytes)

        record = JobRecord(
            job_id=job_id,
            status="pending",
            created_at=timestamp,
            updated_at=timestamp,
            source_video_name=source_video_name,
            job_dir=str(job_dir),
        )
        self._set_job(record)
        self._persist_job(record)
        return record

    def process_job(self, job_id: str) -> None:
        """Run analysis for a pending job and persist its final state."""

        record = self.get_job(job_id)
        record.status = "running"
        record.updated_at = self._utc_now()
        self._set_job(record)
        self._persist_job(record)

        try:
            source_video_path = Path(record.job_dir) / "source_video.mp4"
            record.analysis = self._analyzer.analyze_video(source_video_path, Path(record.job_dir))
            record.status = "completed"
            record.error_message = None
        except Exception as exc:
            record.status = "failed"
            record.error_message = str(exc)
        finally:
            record.updated_at = self._utc_now()
            self._set_job(record)
            self._persist_job(record)

    def get_job(self, job_id: str) -> JobRecord:
        """Return a job record from memory or persisted storage."""

        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]

        summary_path = self._storage_root / job_id / "job_summary.json"
        if not summary_path.exists():
            raise LogicError("Job not found.", status.HTTP_404_NOT_FOUND)

        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        record = JobRecord.from_dict(payload)
        self._set_job(record)
        return record

    def get_completed_analysis(self, job_id: str) -> AnalysisResult:
        """Return analysis data only for completed jobs."""

        record = self.get_job(job_id)
        if record.status != "completed" or record.analysis is None:
            raise LogicError("Job analysis is not completed yet.", status.HTTP_409_CONFLICT)
        return record.analysis

    def get_vehicle_track(self, job_id: str, track_id: int) -> VehicleTrack:
        """Return one vehicle track from a completed job."""

        analysis = self.get_completed_analysis(job_id)
        vehicle = analysis.vehicles.get(track_id)
        if vehicle is None:
            raise LogicError(
                f"Track {track_id} was not found in job {job_id}.",
                status.HTTP_404_NOT_FOUND,
            )
        return vehicle

    def get_required_closest_frame(self, job_id: str, track_id: int):
        """Return the stored closest frame for a track or raise a logic error."""

        vehicle = self.get_vehicle_track(job_id, track_id)
        if vehicle.closest_frame is None:
            raise LogicError(
                f"Track {track_id} in job {job_id} does not have a closest frame.",
                status.HTTP_404_NOT_FOUND,
            )
        return vehicle.closest_frame

    def get_artifact_path(self, job_id: str, artifact_name: str) -> Path:
        """Return the filesystem path for a generated artifact."""

        record = self.get_job(job_id)
        artifact_path = Path(record.job_dir) / "artifacts" / artifact_name
        if not artifact_path.exists() or not artifact_path.is_file():
            raise LogicError("Artifact not found.", status.HTTP_404_NOT_FOUND)
        return artifact_path

    def build_artifact_urls(self, job_id: str) -> dict[str, str]:
        """Return API-facing artifact URLs for a job."""

        record = self.get_job(job_id)
        if record.analysis is None:
            return {}
        return {
            key: f"/jobs/{job_id}/{value}"
            for key, value in record.analysis.artifacts.items()
        }

    def _persist_job(self, record: JobRecord) -> None:
        """Write the latest job state to the job summary JSON file."""

        summary_path = Path(record.job_dir) / "job_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(record.to_dict(), handle, indent=2)

    def _set_job(self, record: JobRecord) -> None:
        """Store a job record in the in-memory registry."""

        with self._lock:
            self._jobs[record.job_id] = record

    def _utc_now(self) -> str:
        """Return the current UTC timestamp in ISO 8601 format."""

        return datetime.now(UTC).isoformat()
