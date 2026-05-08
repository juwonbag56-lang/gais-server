"""
Heuristic + linear attractiveness scorer (gais-v14-prototype).

Inputs:
- 478-point face mesh in original-image pixel coords
- aligned 112x112 BGR face crop
- detection score

The scorer extracts a small set of geometric & photometric features that are
known correlates of perceived facial attractiveness in literature:

  * facial symmetry (left/right landmark mirror error)
  * facial proportions (golden-ratio adherence: width/height, third-thirds)
  * eye openness and balance
  * inter-ocular distance / face-width ratio
  * skin tone uniformity (std-dev of L channel inside cheek mask)
  * sharpness / focus (Laplacian variance of grayscale crop)

Each feature is mapped to a [0,1] score via a clipped linear map. The final
attractiveness_score is a weighted average; the weights live in
`PROTOTYPE_WEIGHTS` and are intended to be replaced by a learned MLP head
trained on H100. Confidence is the geometric mean of (detection score,
landmark health, image sharpness).

Returns dict ready for /analyze-face response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .detector import FaceCrop

MODEL_VERSION = "gais-v14-prototype"

PROTOTYPE_WEIGHTS: Dict[str, float] = {
    "symmetry": 0.28,
    "proportion": 0.22,
    "eye_balance": 0.14,
    "skin_uniformity": 0.18,
    "sharpness": 0.10,
    "smile_curve": 0.08,
}

# MediaPipe FaceMesh canonical pairs (right-index, left-index) for symmetry
# (small subset; covers eyes, brows, mouth corners, cheeks, jaw)
SYMMETRY_PAIRS: List[Tuple[int, int]] = [
    (33, 263),    # eye outer
    (133, 362),   # eye inner
    (159, 386),   # eye top
    (145, 374),   # eye bottom
    (70, 300),    # brow inner
    (105, 334),   # brow outer
    (61, 291),    # mouth corners
    (78, 308),    # mouth inner corners
    (50, 280),    # cheek
    (234, 454),   # face side (jaw)
    (172, 397),   # jawline
    (152, 152),   # chin (self)
    (10, 10),     # forehead (self)
]


@dataclass
class FeatureBundle:
    symmetry: float
    proportion: float
    eye_balance: float
    skin_uniformity: float
    sharpness: float
    smile_curve: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "symmetry": self.symmetry,
            "proportion": self.proportion,
            "eye_balance": self.eye_balance,
            "skin_uniformity": self.skin_uniformity,
            "sharpness": self.sharpness,
            "smile_curve": self.smile_curve,
        }


def _clip01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return float(max(0.0, min(1.0, x)))


def _symmetry(lm: np.ndarray) -> float:
    """1 - mean normalized mirror distance over canonical pairs."""
    if lm.shape != (478, 3):
        return 0.5
    # canonicalise: vertical mid-line through nose tip (1) and chin (152)
    nose = lm[1, :2]
    chin = lm[152, :2]
    axis = chin - nose
    norm = float(np.linalg.norm(axis))
    if norm < 1e-3:
        return 0.5
    axis_unit = axis / norm
    perp = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float32)

    def reflect(p: np.ndarray) -> np.ndarray:
        # reflect p across the midline through nose along axis
        d = p - nose
        proj_axis = float(np.dot(d, axis_unit)) * axis_unit
        proj_perp = float(np.dot(d, perp)) * perp
        return nose + proj_axis - proj_perp  # mirror perp component

    err = 0.0
    cnt = 0
    inter_eye = float(np.linalg.norm(lm[33, :2] - lm[263, :2]))
    if inter_eye < 1.0:
        return 0.5
    for r, l in SYMMETRY_PAIRS:
        if r == l:
            # self-symmetry: distance from midline
            d_to_axis = abs(float(np.dot(lm[r, :2] - nose, perp)))
            err += d_to_axis / inter_eye
            cnt += 1
            continue
        mirrored = reflect(lm[l, :2])
        d = float(np.linalg.norm(lm[r, :2] - mirrored)) / inter_eye
        err += d
        cnt += 1
    if cnt == 0:
        return 0.5
    mean_err = err / cnt
    # mean_err 0.0 -> perfect, 0.25 -> bad. Map to score:
    return _clip01(1.0 - mean_err / 0.25)


def _proportion(lm: np.ndarray) -> float:
    """Adherence to classic thirds: forehead / midface / chin should be ~equal."""
    if lm.shape != (478, 3):
        return 0.5
    forehead = lm[10, 1]
    brow = lm[9, 1]      # mid-brow
    nose_base = lm[2, 1]  # nose subnasale
    chin = lm[152, 1]
    h = chin - forehead
    if h <= 1e-3:
        return 0.5
    a = (brow - forehead) / h
    b = (nose_base - brow) / h
    c = (chin - nose_base) / h
    # ideal: 0.33 each
    err = (abs(a - 0.33) + abs(b - 0.33) + abs(c - 0.33)) / 3.0
    return _clip01(1.0 - err / 0.20)


def _eye_balance(lm: np.ndarray) -> float:
    """Eye openness L vs R should be similar."""
    if lm.shape != (478, 3):
        return 0.5
    # right eye: 159 (top) - 145 (bottom); left eye: 386 (top) - 374 (bottom)
    r_open = abs(lm[159, 1] - lm[145, 1])
    l_open = abs(lm[386, 1] - lm[374, 1])
    if max(r_open, l_open) < 1e-3:
        return 0.5
    ratio = min(r_open, l_open) / max(r_open, l_open)
    return _clip01(ratio)


def _skin_uniformity(bgr_aligned: np.ndarray) -> float:
    """Lower std-dev of L channel inside center mask -> more uniform skin tone."""
    if bgr_aligned.size == 0:
        return 0.5
    # cheek box: central 70% horizontally, 50%-80% vertically
    h, w = bgr_aligned.shape[:2]
    x1, x2 = int(w * 0.15), int(w * 0.85)
    y1, y2 = int(h * 0.50), int(h * 0.80)
    patch = bgr_aligned[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.5
    # convert BGR -> approximate luminance L
    lum = (0.114 * patch[..., 0] + 0.587 * patch[..., 1] + 0.299 * patch[..., 2]).astype(np.float32)
    s = float(np.std(lum))
    # std around 5..25 typical; map: <=8 best (1.0), >=30 worst (0)
    return _clip01(1.0 - max(0.0, s - 8.0) / 22.0)


def _sharpness(bgr_aligned: np.ndarray) -> float:
    """Laplacian variance proxy via a 3x3 kernel without scipy/cv2."""
    if bgr_aligned.size == 0:
        return 0.5
    g = (0.114 * bgr_aligned[..., 0] + 0.587 * bgr_aligned[..., 1] + 0.299 * bgr_aligned[..., 2]).astype(np.float32)
    # discrete laplacian
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    var = float(np.var(lap))
    # blur < 50, sharp > 400
    return _clip01((var - 50.0) / 350.0)


def _smile_curve(lm: np.ndarray) -> float:
    """Mouth corners higher than mid-lower lip => smile.

    Returns a tame [0,1] value; not too dominant in final score.
    """
    if lm.shape != (478, 3):
        return 0.5
    left_corner = lm[61, 1]
    right_corner = lm[291, 1]
    mid = lm[14, 1]  # lower lip center
    if mid <= 1e-3:
        return 0.5
    # in image y, smaller y = higher; smile means corners ABOVE mid (smaller y)
    diff = mid - (left_corner + right_corner) / 2.0
    inter_eye = float(np.linalg.norm(lm[33, :2] - lm[263, :2]))
    if inter_eye < 1.0:
        return 0.5
    s = diff / inter_eye  # range roughly [-0.05, 0.10]
    return _clip01(0.5 + s * 5.0)


def extract_features(face: FaceCrop) -> FeatureBundle:
    return FeatureBundle(
        symmetry=_symmetry(face.landmarks),
        proportion=_proportion(face.landmarks),
        eye_balance=_eye_balance(face.landmarks),
        skin_uniformity=_skin_uniformity(face.aligned_bgr),
        sharpness=_sharpness(face.aligned_bgr),
        smile_curve=_smile_curve(face.landmarks),
    )


def score_face(face: FaceCrop) -> Dict[str, Any]:
    fb = extract_features(face)
    feats = fb.as_dict()
    score = sum(PROTOTYPE_WEIGHTS[k] * feats[k] for k in PROTOTYPE_WEIGHTS)
    score = _clip01(score)

    # confidence: geometric mean of (detection score, landmark sanity, sharpness)
    landmark_health = 1.0 if face.landmarks.shape == (478, 3) and float(np.std(face.landmarks)) > 1e-3 else 0.0
    confidence = (max(face.score, 1e-3) * max(landmark_health, 1e-3) * max(feats["sharpness"], 1e-3)) ** (1.0 / 3.0)
    confidence = _clip01(confidence)

    # uncertainty heuristic: 1 - confidence
    uncertainty = _clip01(1.0 - confidence)

    # explanations: top 2 positive / negative deviations from 0.5
    items = sorted(feats.items(), key=lambda kv: kv[1], reverse=True)
    top_pos = [{"feature": k, "value": round(v, 4)} for k, v in items[:2] if v > 0.55]
    top_neg = [{"feature": k, "value": round(v, 4)} for k, v in items[::-1][:2] if v < 0.45]

    return {
        "attractiveness_score": round(score, 4),
        "confidence_score": round(confidence, 4),
        "uncertainty": round(uncertainty, 4),
        "top_positive_features": top_pos,
        "top_negative_features": top_neg,
        "explanation_vector": [round(feats[k], 4) for k in PROTOTYPE_WEIGHTS],
        "feature_names": list(PROTOTYPE_WEIGHTS.keys()),
        "model_version": MODEL_VERSION,
    }
