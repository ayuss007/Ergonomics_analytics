from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from ergonomics import ErgonomicsAnalyzer, risk_bucket
from flow import FlowAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def fake_rtmo_keypoints(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Placeholder RTMO output. Replace with actual RTMO inference integration."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h = y2 - y1
    w = x2 - x1
    points = np.zeros((17, 2), dtype=np.float32)
    points[0] = [cx, y1 + 0.1 * h]
    points[5] = [cx - 0.15 * w, y1 + 0.25 * h]
    points[6] = [cx + 0.15 * w, y1 + 0.25 * h]
    points[11] = [cx - 0.12 * w, y1 + 0.55 * h]
    points[12] = [cx + 0.12 * w, y1 + 0.55 * h]
    points[13] = [cx - 0.12 * w, y1 + 0.78 * h]
    points[14] = [cx + 0.12 * w, y1 + 0.78 * h]
    return points


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(cfg["yolo_model"])
    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Unable to read video: {args.video}")

    height, width = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = cv2.VideoWriter(
        str(output_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        cap.get(cv2.CAP_PROP_FPS) or 25,
        (width, height),
    )

    ergo = ErgonomicsAnalyzer()
    flow = FlowAnalyzer((height, width), grid_size=cfg["flow"]["grid_size"])
    metrics = []

    frame_idx = 0
    for result in model.track(source=args.video, stream=True, persist=True, conf=cfg["confidence_threshold"]):
        img = result.orig_img.copy()
        boxes = result.boxes
        if boxes is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy))

        for box, track_id in zip(xyxy, ids):
            x1, y1, x2, y2 = box.tolist()
            keypoints = fake_rtmo_keypoints(x1, y1, x2, y2)
            score = ergo.score(keypoints)
            label = risk_bucket(score.overall_risk, cfg["risk_thresholds"])

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            flow.ingest(frame_idx, int(track_id), cx, cy)

            metrics.append(
                {
                    "frame": frame_idx,
                    "track_id": int(track_id),
                    "risk_score": score.overall_risk,
                    "risk_level": label,
                    "back_flexion": score.back_flexion_risk,
                    "squat": score.squat_risk,
                    "asymmetry": score.asymmetry_risk,
                    "cx": cx,
                    "cy": cy,
                }
            )

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"ID {track_id} | {label} {score.overall_risk:.2f}", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        writer.write(img)
        frame_idx += 1

    writer.release()
    cap.release()

    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame([p.__dict__ for p in flow.points]).to_csv(output_dir / "route_points.csv", index=False)
    flow.save_bottlenecks(output_dir / "bottlenecks.json", percentile=cfg["flow"]["bottleneck_percentile"])


if __name__ == "__main__":
    main()
