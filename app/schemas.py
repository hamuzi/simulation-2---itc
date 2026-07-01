"""Pydantic schemas exposed by the FastAPI application."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PointResponse(BaseModel):
    """Represent a point in image coordinates."""

    x: float
    y: float


class BoundingBoxResponse(BaseModel):
    """Represent a bounding box in image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


class JobCreateResponse(BaseModel):
    """Return the initial job creation details."""

    job_id: str
    status: str
    created_at: str


class JobStatusResponse(BaseModel):
    """Return the current state of a video analysis job."""

    job_id: str
    status: str
    created_at: str
    error_message: str | None = None
    vehicle_count: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)


class VehicleSummaryResponse(BaseModel):
    """Return the summary data for one tracked car."""

    track_id: int
    frames_seen: int
    closest_frame_index: int
    closest_timestamp_sec: float
    closest_bbox: BoundingBoxResponse
    closest_center: PointResponse
    max_bbox_area: float


class ClosestFrameResponse(BaseModel):
    """Return the closest-frame details for one tracked car."""

    track_id: int
    closest_frame_index: int
    closest_timestamp_sec: float
    bbox: BoundingBoxResponse
    center: PointResponse
    bbox_area: float
    confidence: float
    frame_image_url: str
    proof_summary: dict[str, Any]


class FrameEvidenceResponse(BaseModel):
    """Return one per-frame evidence row for one tracked car."""

    frame_index: int
    timestamp_sec: float
    bbox: BoundingBoxResponse
    center: PointResponse
    bbox_area: float
    confidence: float
    is_closest_frame: bool
