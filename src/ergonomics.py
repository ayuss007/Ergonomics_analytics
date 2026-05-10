from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ErgonomicScore:
    overall_risk: float
    back_flexion_risk: float
    squat_risk: float
    asymmetry_risk: float


class ErgonomicsAnalyzer:
    """Compute simplified ergonomic risk from COCO-style keypoints."""

    def score(self, keypoints: np.ndarray) -> ErgonomicScore:
        if keypoints.shape[0] < 17:
            return ErgonomicScore(0.0, 0.0, 0.0, 0.0)

        nose = keypoints[0]
        l_shoulder, r_shoulder = keypoints[5], keypoints[6]
        l_hip, r_hip = keypoints[11], keypoints[12]
        l_knee, r_knee = keypoints[13], keypoints[14]

        shoulder_mid = (l_shoulder + r_shoulder) / 2.0
        hip_mid = (l_hip + r_hip) / 2.0
        knee_mid = (l_knee + r_knee) / 2.0

        trunk_vec = shoulder_mid - hip_mid
        leg_vec = knee_mid - hip_mid

        trunk_angle = self._vertical_angle(trunk_vec)
        knee_angle = self._angle_between(-leg_vec, trunk_vec)
        shoulder_tilt = abs(l_shoulder[1] - r_shoulder[1])

        back_flexion_risk = np.clip(trunk_angle / 65.0, 0.0, 1.0)
        squat_risk = np.clip((130.0 - knee_angle) / 60.0, 0.0, 1.0)
        asymmetry_risk = np.clip(shoulder_tilt / 60.0, 0.0, 1.0)

        overall = float(np.mean([back_flexion_risk, squat_risk, asymmetry_risk]))
        return ErgonomicScore(overall, float(back_flexion_risk), float(squat_risk), float(asymmetry_risk))

    @staticmethod
    def _vertical_angle(vec: np.ndarray) -> float:
        vertical = np.array([0.0, -1.0])
        return ErgonomicsAnalyzer._angle_between(vec, vertical)

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))


def risk_bucket(score: float, thresholds: Dict[str, float]) -> str:
    if score >= thresholds["high"]:
        return "high"
    if score >= thresholds["medium"]:
        return "medium"
    return "low"
