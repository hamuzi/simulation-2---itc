"""Compatibility wrapper that re-exports the FastAPI app factory."""

from __future__ import annotations

from fastapi import FastAPI

from app.config import Settings
from app.controllers.job_controller import JobController
from main import create_app as _create_app


def create_app(
    settings: Settings | None = None,
    job_controller: JobController | None = None,
) -> FastAPI:
    """Return the main application factory for backward compatibility."""

    return _create_app(settings=settings, job_controller=job_controller)
