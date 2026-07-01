"""Route definitions for vehicle analysis job endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse

from app.controllers.job_controller import JobController
from app.schemas import (
    ClosestFrameResponse,
    FrameEvidenceResponse,
    JobCreateResponse,
    JobStatusResponse,
    VehicleSummaryResponse,
)


def create_job_router(controller: JobController) -> APIRouter:
    """Create the jobs router bound to the provided controller."""

    router = APIRouter()

    @router.post("/jobs", response_model=JobCreateResponse, status_code=202)
    async def create_job(background_tasks: BackgroundTasks, video: UploadFile = File(...)) -> JobCreateResponse:
        """Create a new asynchronous job for uploaded video analysis."""

        return await controller.create_job(video, background_tasks)

    @router.get("/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job_status(job_id: str) -> JobStatusResponse:
        """Return the current processing status for one job."""

        return controller.get_job_status(job_id)

    @router.get("/jobs/{job_id}/vehicles", response_model=list[VehicleSummaryResponse])
    async def list_vehicles(job_id: str) -> list[VehicleSummaryResponse]:
        """Return summary rows for all tracked vehicles in a completed job."""

        return controller.list_vehicles(job_id)

    @router.get("/jobs/{job_id}/vehicles/{track_id}/closest-frame", response_model=ClosestFrameResponse)
    async def get_closest_frame(job_id: str, track_id: int) -> ClosestFrameResponse:
        """Return the closest frame for one tracked vehicle."""

        return controller.get_closest_frame(job_id, track_id)

    @router.get("/jobs/{job_id}/vehicles/{track_id}/frames", response_model=list[FrameEvidenceResponse])
    async def get_vehicle_frames(job_id: str, track_id: int) -> list[FrameEvidenceResponse]:
        """Return the full evidence timeline for one tracked vehicle."""

        return controller.get_vehicle_frames(job_id, track_id)

    @router.get("/jobs/{job_id}/artifacts/{artifact_name}")
    async def get_artifact(job_id: str, artifact_name: str) -> FileResponse:
        """Serve one generated artifact file from a completed analysis job."""

        artifact_path = controller.get_artifact_path(job_id, artifact_name)
        return FileResponse(path=artifact_path)

    return router
