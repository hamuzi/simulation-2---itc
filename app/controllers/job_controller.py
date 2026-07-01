"""Controller layer for routes, logic calls, and HTTP error translation."""

from __future__ import annotations

from fastapi import BackgroundTasks, HTTPException, UploadFile

from app.domain import BoundingBox, FrameEvidence
from app.errors import LogicError
from app.logic.job_logic import JobLogic
from app.schemas import (
    BoundingBoxResponse,
    ClosestFrameResponse,
    FrameEvidenceResponse,
    JobCreateResponse,
    JobStatusResponse,
    PointResponse,
    VehicleSummaryResponse,
)


class JobController:
    """Translate HTTP requests into logic calls and response schemas."""

    def __init__(self, job_logic: JobLogic) -> None:
        """Store the business-logic dependency used by the controller."""

        self._job_logic = job_logic

    async def create_job(self, upload: UploadFile, background_tasks: BackgroundTasks) -> JobCreateResponse:
        """Create a job from an uploaded video file and enqueue background analysis."""

        try:
            source_video_bytes = await upload.read()
            source_video_name = upload.filename or "uploaded_video.mp4"
            record = self._job_logic.create_job(source_video_bytes, source_video_name)
            background_tasks.add_task(self._job_logic.process_job, record.job_id)
            return JobCreateResponse(
                job_id=record.job_id,
                status=record.status,
                created_at=record.created_at,
            )
        except LogicError as exc:
            raise self._to_http_exception(exc) from exc
        finally:
            await upload.close()

    def get_job_status(self, job_id: str) -> JobStatusResponse:
        """Return the current API status view for one job."""

        try:
            record = self._job_logic.get_job(job_id)
            return JobStatusResponse(
                job_id=record.job_id,
                status=record.status,
                created_at=record.created_at,
                error_message=record.error_message,
                vehicle_count=record.vehicle_count(),
                artifacts=self._job_logic.build_artifact_urls(job_id),
            )
        except LogicError as exc:
            raise self._to_http_exception(exc) from exc

    def list_vehicles(self, job_id: str) -> list[VehicleSummaryResponse]:
        """Return summary rows for all tracked cars in a completed job."""

        try:
            analysis = self._job_logic.get_completed_analysis(job_id)
            return [
                self._build_vehicle_summary_response(job_id, track_id)
                for track_id in sorted(analysis.vehicles)
            ]
        except LogicError as exc:
            raise self._to_http_exception(exc) from exc

    def get_closest_frame(self, job_id: str, track_id: int) -> ClosestFrameResponse:
        """Return the closest-frame response for one tracked car."""

        try:
            vehicle = self._job_logic.get_vehicle_track(job_id, track_id)
            closest = self._job_logic.get_required_closest_frame(job_id, track_id)
            return ClosestFrameResponse(
                track_id=track_id,
                closest_frame_index=closest.frame_index,
                closest_timestamp_sec=closest.timestamp_sec,
                bbox=self._build_bbox_response(closest.bbox),
                center=self._build_point_response(closest.center),
                bbox_area=closest.bbox_area,
                confidence=closest.confidence,
                frame_image_url=f"/jobs/{job_id}/artifacts/vehicle_{track_id}_closest.jpg",
                proof_summary={
                    "heuristic": "closest frame is the frame with the maximal bounding-box area",
                    "frames_evaluated": vehicle.frames_seen(),
                    "selected_bbox_area": closest.bbox_area,
                    "evidence_endpoint": f"/jobs/{job_id}/vehicles/{track_id}/frames",
                },
            )
        except LogicError as exc:
            raise self._to_http_exception(exc) from exc

    def get_vehicle_frames(self, job_id: str, track_id: int) -> list[FrameEvidenceResponse]:
        """Return all frame evidence rows for one tracked car."""

        try:
            vehicle = self._job_logic.get_vehicle_track(job_id, track_id)
            closest = self._job_logic.get_required_closest_frame(job_id, track_id)
            return [
                self._build_frame_evidence_response(frame, frame.frame_index == closest.frame_index)
                for frame in vehicle.frames
            ]
        except LogicError as exc:
            raise self._to_http_exception(exc) from exc

    def get_artifact_path(self, job_id: str, artifact_name: str):
        """Return the filesystem path for one generated job artifact."""

        try:
            return self._job_logic.get_artifact_path(job_id, artifact_name)
        except LogicError as exc:
            raise self._to_http_exception(exc) from exc

    def _build_vehicle_summary_response(self, job_id: str, track_id: int) -> VehicleSummaryResponse:
        """Build the API summary response for one tracked car."""

        vehicle = self._job_logic.get_vehicle_track(job_id, track_id)
        closest = self._job_logic.get_required_closest_frame(job_id, track_id)
        return VehicleSummaryResponse(
            track_id=track_id,
            frames_seen=vehicle.frames_seen(),
            closest_frame_index=closest.frame_index,
            closest_timestamp_sec=closest.timestamp_sec,
            closest_bbox=self._build_bbox_response(closest.bbox),
            closest_center=self._build_point_response(closest.center),
            max_bbox_area=closest.bbox_area,
        )

    def _build_frame_evidence_response(
        self,
        frame: FrameEvidence,
        is_closest_frame: bool,
    ) -> FrameEvidenceResponse:
        """Build one API evidence row from domain frame data."""

        return FrameEvidenceResponse(
            frame_index=frame.frame_index,
            timestamp_sec=frame.timestamp_sec,
            bbox=self._build_bbox_response(frame.bbox),
            center=self._build_point_response(frame.center),
            bbox_area=frame.bbox_area,
            confidence=frame.confidence,
            is_closest_frame=is_closest_frame,
        )

    def _build_bbox_response(self, bbox: BoundingBox) -> BoundingBoxResponse:
        """Convert a domain bounding box into an API schema."""

        return BoundingBoxResponse(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2)

    def _build_point_response(self, center: tuple[float, float]) -> PointResponse:
        """Convert a point tuple into an API schema."""

        return PointResponse(x=center[0], y=center[1])

    def _to_http_exception(self, error: LogicError) -> HTTPException:
        """Translate a logic-layer exception into an HTTP exception."""

        return HTTPException(status_code=error.status_code, detail=error.message)
