# Warehouse Ergonomics & Flow Analytics

An independent computer vision project that analyzes warehouse worker ergonomics and routing flow using real-time pose estimation and object tracking.

## Project goals
- Monitor worker movement with pose estimation.
- Compute ergonomic risk indicators in real time.
- Track worker routes and detect spatial bottlenecks.
- Export analytics outputs suitable for operations dashboards.

## Tech stack
- **YOLOv8** for person detection/tracking IDs.
- **RTMO** (or compatible top-down keypoint model) for per-person pose estimation.
- **OpenCV** for video processing and visualization.
- **NumPy/Pandas** for metrics and exports.

## Repository structure
- `src/pipeline.py` – end-to-end video analytics pipeline.
- `src/ergonomics.py` – ergonomic scoring from pose keypoints.
- `src/flow.py` – heatmap and bottleneck analytics.
- `configs/default.yaml` – runtime configuration.
- `requirements.txt` – Python dependencies.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py --video data/sample.mp4 --config configs/default.yaml --output runs/demo
```

## Outputs
The pipeline writes:
- `metrics.csv` with per-frame ergonomic + movement metrics.
- `route_points.csv` with tracked worker trajectory points.
- `bottlenecks.json` summarizing congestion hotspots.
- `annotated.mp4` with overlays.

## Resume-ready summary
**Warehouse Ergonomics & Flow Analytics | Independent Project | Feb 2026**  
Designed an automated CV pipeline using RTMO and YOLOv8 for spatial logistics optimization.
- Deployed RTMO pose estimation to monitor warehouse worker movements, extracting real-time ergonomic risk scores.
- Tracked personnel routing to identify spatial bottlenecks, translating physical movements into data to drive supply chain efficiency.
