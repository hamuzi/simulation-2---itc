# Vehicle Proximity Analysis – System Explanation

## The model and the data flow:

The system receives a video of road traffic as input and processes it frame by frame. In each frame, vehicles are detected using a YOLO model, and tracking is then applied to preserve a consistent identifier for each vehicle throughout the video.

For every detected vehicle, the system stores the following information: frame number, timestamp, vehicle location in the frame, and bounding box area. The main assumption is that the closer a vehicle is to the camera, the larger it appears in the frame. Therefore, for each vehicle, the system keeps the frame in which its bounding box area is the largest.

The data flow is:

Video -> Frame Extraction -> YOLO Detection -> Tracking -> Bounding Box Area Calculation -> Closest Frame Selection -> CSV and Visual Output Generation

I chose this approach because it is simple, fast, and well suited to a limited development window of 90 minutes. In addition, it makes it possible to present both numerical and visual results in a clear way.

## System Accuracy

This version model YOLO11n i chose its the fastest but with not 100% accuracy.

The metric is:

Number of correctly identified vehicles / Total number of examined vehicles

For example, if the system correctly selects the closest frame for 18 out of 20 vehicles, the system accuracy is 90%.

The accuracy depends on video quality, camera angle, vehicle occlusions, and the tracker's ability to maintain the same ID throughout the video.

## Critical Analysis

The system performs especially well when the camera is static, the vehicles clearly approach the camera, and there are few occlusions. In such cases, the bounding box area truly increases as the vehicle gets closer, so selecting the closest frame is relatively accurate.

The system may fail in cases where a vehicle is partially occluded, when traffic is very dense, when the tracker switches IDs in the middle of the video, or when vehicles differ significantly in physical size. For example, a distant truck may appear larger than a nearby private car, so using only bounding box area is not always a perfect distance indicator.

In addition, the system does not estimate real physical depth. Instead, it uses a visual approximation based on the apparent size of the vehicle in the frame.

## Task Prioritization

If I had one more hour to work on the project, I would invest it in improving camera-distance estimation using a depth model such as Depth Anything or MiDaS. That would make it possible to estimate the vehicle’s distance from the camera more accurately instead of relying only on bounding box size.

In addition, I would add tracking quality validation, improve the visualization, and provide clearer highlighting of the selected frame for each vehicle. For example, I would save a separate image for each vehicle at its closest moment together with a structured results table containing the ID, timestamp, frame number, bounding box, and maximum area.
