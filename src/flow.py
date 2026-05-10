from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class FlowPoint:
    frame: int
    track_id: int
    x: float
    y: float


class FlowAnalyzer:
    def __init__(self, frame_shape: Tuple[int, int], grid_size: int = 32) -> None:
        h, w = frame_shape
        self.grid_size = grid_size
        self.grid_h = max(1, h // grid_size)
        self.grid_w = max(1, w // grid_size)
        self.heatmap = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.points: List[FlowPoint] = []

    def ingest(self, frame_idx: int, track_id: int, x: float, y: float) -> None:
        gx = int(np.clip(x // self.grid_size, 0, self.grid_w - 1))
        gy = int(np.clip(y // self.grid_size, 0, self.grid_h - 1))
        self.heatmap[gy, gx] += 1.0
        self.points.append(FlowPoint(frame_idx, track_id, x, y))

    def bottlenecks(self, percentile: float) -> List[Dict[str, float]]:
        threshold = np.percentile(self.heatmap, percentile)
        ys, xs = np.where(self.heatmap >= threshold)
        return [
            {"grid_x": int(x), "grid_y": int(y), "count": float(self.heatmap[y, x])}
            for y, x in zip(ys, xs)
            if self.heatmap[y, x] > 0
        ]

    def save_bottlenecks(self, output_path: Path, percentile: float = 92) -> None:
        payload = {"percentile": percentile, "hotspots": self.bottlenecks(percentile)}
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
