"""ASGI entrypoint for the vehicle closest-frame FastAPI service."""

from __future__ import annotations

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.controllers.job_controller import JobController
from app.logic.job_logic import JobLogic
from app.routes.job_routes import create_job_router
from app.services.analysis import VideoAnalyzer


def create_app(
    settings: Settings | None = None,
    job_controller: JobController | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""

    resolved_settings = settings or get_settings()
    resolved_controller = job_controller or JobController(
        JobLogic(
            settings=resolved_settings,
            analyzer=VideoAnalyzer(resolved_settings),
        )
    )

    app = FastAPI(title="Vehicle Closest-Frame Service", version="1.0.0")
    app.include_router(create_job_router(resolved_controller))
    return app


app = create_app()
