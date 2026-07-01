# Vehicle Closest-Frame Service

FastAPI service that accepts a road video, tracks cars with YOLO, and returns the closest frame for each tracked car based on the maximal bounding-box area heuristic.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## API flow

1. `POST /jobs` with a multipart `video` file.
2. Poll `GET /jobs/{job_id}` until the status becomes `completed`.
3. Call `GET /jobs/{job_id}/vehicles` to list available `track_id` values.
4. Call `GET /jobs/{job_id}/vehicles/{track_id}/closest-frame` for the chosen car.
5. Call `GET /jobs/{job_id}/vehicles/{track_id}/frames` to inspect all evidence rows and prove the selected frame is correct.
