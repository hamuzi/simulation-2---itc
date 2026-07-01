"""Domain models for job state and vehicle tracking analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    """Represent a bounding box in image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        """Return the area of the bounding box."""

        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def center(self) -> tuple[float, float]:
        """Return the center point of the bounding box."""

        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_dict(self) -> dict[str, float]:
        """Serialize the bounding box into a JSON-friendly dictionary."""

        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoundingBox":
        """Create a bounding box from serialized data."""

        return cls(
            x1=float(payload["x1"]),
            y1=float(payload["y1"]),
            x2=float(payload["x2"]),
            y2=float(payload["y2"]),
        )


@dataclass
class FrameEvidence:
    """Store all evidence values for one tracked car in one frame."""

    frame_index: int
    timestamp_sec: float
    bbox: BoundingBox
    center: tuple[float, float]
    bbox_area: float
    confidence: float

    def to_dict(self, is_closest_frame: bool = False) -> dict[str, Any]:
        """Serialize the frame evidence into a JSON-friendly dictionary."""

        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "bbox": self.bbox.to_dict(),
            "center": {"x": self.center[0], "y": self.center[1]},
            "bbox_area": self.bbox_area,
            "confidence": self.confidence,
            "is_closest_frame": is_closest_frame,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrameEvidence":
        """Create frame evidence from serialized data."""

        center_payload = payload["center"]
        return cls(
            frame_index=int(payload["frame_index"]),
            timestamp_sec=float(payload["timestamp_sec"]),
            bbox=BoundingBox.from_dict(payload["bbox"]),
            center=(float(center_payload["x"]), float(center_payload["y"])),
            bbox_area=float(payload["bbox_area"]),
            confidence=float(payload["confidence"]),
        )


def is_better_candidate(candidate: FrameEvidence, current: FrameEvidence) -> bool:
    """Return whether the candidate frame should replace the current closest frame."""

    if candidate.bbox_area != current.bbox_area:
        return candidate.bbox_area > current.bbox_area
    if candidate.bbox.y2 != current.bbox.y2:
        return candidate.bbox.y2 > current.bbox.y2
    if candidate.confidence != current.confidence:
        return candidate.confidence > current.confidence
    return candidate.frame_index > current.frame_index


@dataclass
class VehicleTrack:
    """Store all frames and the best closest-frame candidate for one tracked car."""

    track_id: int
    frames: list[FrameEvidence] = field(default_factory=list)
    closest_frame: FrameEvidence | None = None

    def add_evidence(self, evidence: FrameEvidence) -> bool:
        """Add a new frame evidence item and update the closest frame if needed."""

        self.frames.append(evidence)
        if self.closest_frame is None or is_better_candidate(evidence, self.closest_frame):
            self.closest_frame = evidence
            return True
        return False

    def frames_seen(self) -> int:
        """Return the number of frames in which the car was tracked."""

        return len(self.frames)

    def max_bbox_area(self) -> float:
        """Return the largest bounding-box area recorded for the track."""

        return self.closest_frame.bbox_area if self.closest_frame else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full vehicle track into a JSON-friendly dictionary."""

        closest_frame = self.closest_frame
        return {
            "track_id": self.track_id,
            "frames_seen": self.frames_seen(),
            "closest_frame": closest_frame.to_dict(is_closest_frame=True) if closest_frame else None,
            "frames": [
                frame.to_dict(is_closest_frame=closest_frame is not None and frame.frame_index == closest_frame.frame_index)
                for frame in self.frames
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VehicleTrack":
        """Create a vehicle track from serialized data."""

        track = cls(track_id=int(payload["track_id"]))
        for frame_payload in payload.get("frames", []):
            track.add_evidence(FrameEvidence.from_dict(frame_payload))
        return track


@dataclass
class AnalysisResult:
    """Store the analysis output for one processed video job."""

    fps: float
    total_frames: int
    vehicles: dict[int, VehicleTrack]
    artifacts: dict[str, str]

    def vehicle_count(self) -> int:
        """Return the number of unique tracked cars."""

        return len(self.vehicles)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the analysis result into a JSON-friendly dictionary."""

        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "vehicles": [vehicle.to_dict() for vehicle in self.vehicles.values()],
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnalysisResult":
        """Create an analysis result from serialized data."""

        vehicles = {
            int(vehicle_payload["track_id"]): VehicleTrack.from_dict(vehicle_payload)
            for vehicle_payload in payload.get("vehicles", [])
        }
        return cls(
            fps=float(payload.get("fps", 0.0)),
            total_frames=int(payload.get("total_frames", 0)),
            vehicles=vehicles,
            artifacts={str(key): str(value) for key, value in payload.get("artifacts", {}).items()},
        )


@dataclass
class JobRecord:
    """Store the lifecycle state for one asynchronous analysis job."""

    job_id: str
    status: str
    created_at: str
    updated_at: str
    source_video_name: str
    job_dir: str
    error_message: str | None = None
    analysis: AnalysisResult | None = None

    def vehicle_count(self) -> int:
        """Return the number of vehicles detected for the job."""

        return self.analysis.vehicle_count() if self.analysis else 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the job record into a JSON-friendly dictionary."""

        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_video_name": self.source_video_name,
            "job_dir": self.job_dir,
            "error_message": self.error_message,
            "vehicle_count": self.vehicle_count(),
            "artifacts": self.analysis.artifacts if self.analysis else {},
            "analysis": self.analysis.to_dict() if self.analysis else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobRecord":
        """Create a job record from serialized data."""

        analysis_payload = payload.get("analysis")
        analysis = AnalysisResult.from_dict(analysis_payload) if analysis_payload else None
        return cls(
            job_id=str(payload["job_id"]),
            status=str(payload["status"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            source_video_name=str(payload["source_video_name"]),
            job_dir=str(payload["job_dir"]),
            error_message=payload.get("error_message"),
            analysis=analysis,
        )
