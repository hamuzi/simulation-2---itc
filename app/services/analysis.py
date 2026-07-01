"""Video analysis service that runs YOLO tracking and produces artifacts."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from app.config import Settings
from app.domain import AnalysisResult, FrameEvidence, VehicleTrack
from app.utils.geometry import build_bounding_box, calculate_bbox_area, calculate_bbox_center


class VideoAnalyzer:
    """Run YOLO tracking on an uploaded video and persist output artifacts."""

    def __init__(self, settings: Settings) -> None:
        """Store runtime settings for future video analysis jobs."""

        self._settings = settings

    def analyze_video(self, source_video: Path, job_dir: Path) -> AnalysisResult:
        """Run YOLO tracking on the video and persist all required output artifacts."""

        cv2 = self._import_cv2()
        yolo_cls = self._import_yolo()

        artifacts_dir = job_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(source_video))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video file: {source_video}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        writer = self._build_writer(cv2, artifacts_dir, fps, frame_width, frame_height)
        model = yolo_cls(self._settings.model_name)

        vehicles: dict[int, VehicleTrack] = {}
        best_frames: dict[int, tuple[Any, FrameEvidence]] = {}
        total_frames = 0

        try:
            while True:
                has_frame, frame = capture.read()
                if not has_frame:
                    break

                result = self._track_frame(model, frame)
                total_frames += 1
                frame_index = total_frames - 1
                timestamp_sec = frame_index / fps if fps > 0 else 0.0

                if writer is not None:
                    writer.write(result.plot())

                self._update_tracks(
                    result=result,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    frame=frame,
                    vehicles=vehicles,
                    best_frames=best_frames,
                )
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        artifacts = self._write_artifacts(
            cv2=cv2,
            artifacts_dir=artifacts_dir,
            vehicles=vehicles,
            best_frames=best_frames,
        )

        return AnalysisResult(
            fps=fps,
            total_frames=total_frames,
            vehicles=vehicles,
            artifacts=artifacts,
        )

    def _import_cv2(self) -> Any:
        """Import OpenCV lazily so tests can patch the analyzer without the dependency."""

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("opencv-python is required to analyze videos.") from exc
        return cv2

    def _import_yolo(self) -> Any:
        """Import Ultralytics lazily so the app can still boot without the dependency."""

        self._configure_yolo_runtime()
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ultralytics is required to analyze videos.") from exc
        return YOLO

    def _configure_yolo_runtime(self) -> None:
        """Point Ultralytics runtime files to a writable directory inside the workspace."""

        runtime_root = self._settings.yolo_config_root
        runtime_root.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(runtime_root.resolve())

    def _build_writer(self, cv2: Any, artifacts_dir: Path, fps: float, width: int, height: int) -> Any | None:
        """Create an annotated video writer when enabled and video metadata is valid."""

        if not self._settings.save_annotated_video or fps <= 0 or width <= 0 or height <= 0:
            return None
        output_path = artifacts_dir / "annotated_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    def _track_frame(self, model: Any, frame: Any) -> Any:
        """Run YOLO tracking for a single frame and return the first result object."""

        results = model.track(
            source=frame,
            persist=True,
            tracker=self._settings.tracker_config,
            classes=list(self._settings.allowed_classes),
            verbose=False,
        )
        return results[0]

    def _update_tracks(
        self,
        result: Any,
        frame_index: int,
        timestamp_sec: float,
        frame: Any,
        vehicles: dict[int, VehicleTrack],
        best_frames: dict[int, tuple[Any, FrameEvidence]],
    ) -> None:
        """Update all per-track evidence values for the current frame."""

        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "id", None) is None:
            return

        ids = boxes.id.int().cpu().tolist()
        xyxy_list = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [0.0] * len(ids)

        for track_id, xyxy, confidence in zip(ids, xyxy_list, confidences):
            bbox = build_bounding_box(*xyxy)
            evidence = FrameEvidence(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                bbox=bbox,
                center=calculate_bbox_center(bbox),
                bbox_area=calculate_bbox_area(bbox),
                confidence=float(confidence),
            )

            vehicle_track = vehicles.setdefault(int(track_id), VehicleTrack(track_id=int(track_id)))
            was_updated = vehicle_track.add_evidence(evidence)
            if was_updated:
                best_frames[int(track_id)] = (frame.copy(), evidence)

    def _write_artifacts(
        self,
        cv2: Any,
        artifacts_dir: Path,
        vehicles: dict[int, VehicleTrack],
        best_frames: dict[int, tuple[Any, FrameEvidence]],
    ) -> dict[str, str]:
        """Write image and CSV artifacts for the completed analysis job."""

        artifacts: dict[str, str] = {}
        summary_csv = artifacts_dir / "vehicles_summary.csv"

        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "track_id",
                    "frames_seen",
                    "closest_frame_index",
                    "closest_timestamp_sec",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "center_x",
                    "center_y",
                    "max_bbox_area",
                    "confidence",
                ],
            )
            writer.writeheader()
            for track_id in sorted(vehicles):
                vehicle = vehicles[track_id]
                self._write_vehicle_frames_csv(artifacts_dir, vehicle)
                closest_frame_name = self._write_closest_frame_image(cv2, artifacts_dir, track_id, best_frames[track_id])
                artifacts[f"vehicle_{track_id}_frames_csv"] = f"artifacts/{self._build_frames_csv_name(track_id)}"
                artifacts[f"vehicle_{track_id}_closest_image"] = f"artifacts/{closest_frame_name}"

                closest = vehicle.closest_frame
                if closest is None:
                    continue

                writer.writerow(
                    {
                        "track_id": track_id,
                        "frames_seen": vehicle.frames_seen(),
                        "closest_frame_index": closest.frame_index,
                        "closest_timestamp_sec": closest.timestamp_sec,
                        "x1": closest.bbox.x1,
                        "y1": closest.bbox.y1,
                        "x2": closest.bbox.x2,
                        "y2": closest.bbox.y2,
                        "center_x": closest.center[0],
                        "center_y": closest.center[1],
                        "max_bbox_area": closest.bbox_area,
                        "confidence": closest.confidence,
                    }
                )

        artifacts["vehicles_summary_csv"] = "artifacts/vehicles_summary.csv"
        annotated_video = artifacts_dir / "annotated_video.mp4"
        if annotated_video.exists():
            artifacts["annotated_video"] = "artifacts/annotated_video.mp4"
        return artifacts

    def _write_vehicle_frames_csv(self, artifacts_dir: Path, vehicle: VehicleTrack) -> None:
        """Write the full per-frame evidence CSV for one tracked car."""

        csv_path = artifacts_dir / self._build_frames_csv_name(vehicle.track_id)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "frame_index",
                    "timestamp_sec",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "center_x",
                    "center_y",
                    "bbox_area",
                    "confidence",
                    "is_closest_frame",
                ],
            )
            writer.writeheader()
            closest_frame_index = vehicle.closest_frame.frame_index if vehicle.closest_frame else -1
            for frame in vehicle.frames:
                writer.writerow(
                    {
                        "frame_index": frame.frame_index,
                        "timestamp_sec": frame.timestamp_sec,
                        "x1": frame.bbox.x1,
                        "y1": frame.bbox.y1,
                        "x2": frame.bbox.x2,
                        "y2": frame.bbox.y2,
                        "center_x": frame.center[0],
                        "center_y": frame.center[1],
                        "bbox_area": frame.bbox_area,
                        "confidence": frame.confidence,
                        "is_closest_frame": frame.frame_index == closest_frame_index,
                    }
                )

    def _write_closest_frame_image(
        self,
        cv2: Any,
        artifacts_dir: Path,
        track_id: int,
        best_frame_entry: tuple[Any, FrameEvidence],
    ) -> str:
        """Write the annotated closest-frame image for one tracked car."""

        frame, evidence = best_frame_entry
        label = f"track={track_id} area={evidence.bbox_area:.1f} time={evidence.timestamp_sec:.2f}s"
        frame_copy = frame.copy()
        cv2.rectangle(
            frame_copy,
            (int(evidence.bbox.x1), int(evidence.bbox.y1)),
            (int(evidence.bbox.x2), int(evidence.bbox.y2)),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame_copy,
            label,
            (int(evidence.bbox.x1), max(20, int(evidence.bbox.y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        file_name = f"vehicle_{track_id}_closest.jpg"
        cv2.imwrite(str(artifacts_dir / file_name), frame_copy)
        return file_name

    def _build_frames_csv_name(self, track_id: int) -> str:
        """Return the standard per-track CSV artifact file name."""

        return f"vehicle_{track_id}_frames.csv"
