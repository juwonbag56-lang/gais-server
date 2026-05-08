"""
Face detection + landmark extraction using MediaPipe Tasks API (CPU-only).

Why Tasks API: legacy `mp.solutions.*` was removed from newer mediapipe wheels
(>= 0.10.10), so we use the supported `mediapipe.tasks.vision` API. Models
(`.tflite` / `.task`) are downloaded once on first use into a writable cache
directory (default `/tmp/gais_models`) so Render Free has zero baked-in assets.

Memory budget on CPU:
  BlazeFace short-range: ~230 KB on disk, ~5 MB resident
  FaceLandmarker (478 mesh, no blendshapes): ~3.7 MB on disk, ~25 MB resident

Total: well under Render Free's 512 MB cap, even alongside FastAPI + asyncpg.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("gais.ai.detector")

# Public model URLs (Google official MediaPipe model garden).
_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

_MODEL_CACHE_DIR = os.environ.get("GAIS_MODEL_CACHE_DIR", "/tmp/gais_models")
_DETECTOR_PATH = os.path.join(_MODEL_CACHE_DIR, "blaze_face_short_range.tflite")
_LANDMARKER_PATH = os.path.join(_MODEL_CACHE_DIR, "face_landmarker.task")

_LOCK = threading.Lock()
_FACE_DETECTOR = None       # mediapipe.tasks.vision.FaceDetector
_FACE_LANDMARKER = None     # mediapipe.tasks.vision.FaceLandmarker


@dataclass
class FaceCrop:
    """One detected face with aligned crop and 478-point 3D landmarks."""

    bbox_xyxy: Tuple[int, int, int, int]
    score: float                          # detection confidence in [0, 1]
    landmarks: np.ndarray                 # (478, 3) float32, image-coords (px); z is relative depth
    aligned_bgr: np.ndarray               # 112x112 BGR uint8 aligned crop


def _download(url: str, dst: str, timeout: int = 30) -> None:
    """Download a file to `dst` atomically (idempotent)."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    logger.info("Fetching model: %s -> %s", url, dst)
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:  # noqa: S310
        while True:
            chunk = r.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst)
    logger.info("Model ready: %s (%d bytes)", dst, os.path.getsize(dst))


def _ensure_models() -> None:
    """Lazy-init MediaPipe Tasks face detector & landmarker (singleton)."""
    global _FACE_DETECTOR, _FACE_LANDMARKER
    if _FACE_DETECTOR is not None and _FACE_LANDMARKER is not None:
        return
    with _LOCK:
        if _FACE_DETECTOR is not None and _FACE_LANDMARKER is not None:
            return

        _download(_DETECTOR_MODEL_URL, _DETECTOR_PATH)
        _download(_LANDMARKER_MODEL_URL, _LANDMARKER_PATH)

        # Lazy import — keeps cold start memory low until /analyze-face is called.
        from mediapipe.tasks import python as mp_py
        from mediapipe.tasks.python import vision as mp_vision

        det_opts = mp_vision.FaceDetectorOptions(
            base_options=mp_py.BaseOptions(model_asset_path=_DETECTOR_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        lm_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_py.BaseOptions(model_asset_path=_LANDMARKER_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        _FACE_DETECTOR = mp_vision.FaceDetector.create_from_options(det_opts)
        _FACE_LANDMARKER = mp_vision.FaceLandmarker.create_from_options(lm_opts)
        logger.info("MediaPipe Tasks (FaceDetector + FaceLandmarker) initialised")


def decode_image_bytes(buf: bytes) -> np.ndarray:
    """Decode raw bytes -> BGR uint8 numpy array via Pillow.

    EXIF orientation is honoured. Output shape: (H, W, 3), dtype uint8, BGR order.
    """
    from PIL import Image, ImageOps  # lazy

    try:
        with Image.open(io.BytesIO(buf)) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            arr = np.array(im, dtype=np.uint8)  # HxWx3 RGB
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Cannot decode image: {exc}") from exc
    return arr[:, :, ::-1].copy()  # BGR


def _crop_with_padding(
    bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: float = 0.30
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = bgr.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    side = int(max(bw, bh) * (1 + pad))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    nx1 = max(0, cx - side // 2)
    ny1 = max(0, cy - side // 2)
    nx2 = min(w, nx1 + side)
    ny2 = min(h, ny1 + side)
    return bgr[ny1:ny2, nx1:nx2].copy(), (nx1, ny1, nx2, ny2)


def _resize_bgr(img: np.ndarray, size: int = 112) -> np.ndarray:
    from PIL import Image as _Image  # lazy

    rgb = img[:, :, ::-1]
    pil = _Image.fromarray(rgb)
    pil = pil.resize((size, size), _Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)[:, :, ::-1].copy()


def _to_mp_image(rgb: np.ndarray):
    """Build a mediapipe.Image from an HxWx3 RGB uint8 numpy array."""
    import mediapipe as mp  # lazy

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def detect_face(bgr: np.ndarray) -> Optional[FaceCrop]:
    """Detect the largest face and return aligned crop + 478-point landmarks."""
    _ensure_models()
    h, w = bgr.shape[:2]
    rgb = bgr[:, :, ::-1].copy()
    mp_image = _to_mp_image(rgb)

    det_result = _FACE_DETECTOR.detect(mp_image)  # type: ignore[union-attr]
    if not det_result.detections:
        return None

    # Pick largest face by bbox area
    best = None
    best_area = 0
    for d in det_result.detections:
        bb = d.bounding_box  # x, y, width, height in px
        area = max(0, bb.width) * max(0, bb.height)
        if area > best_area:
            best_area = area
            best = d
    if best is None:
        return None

    bb = best.bounding_box
    x1 = max(0, int(bb.origin_x))
    y1 = max(0, int(bb.origin_y))
    x2 = min(w, int(bb.origin_x + bb.width))
    y2 = min(h, int(bb.origin_y + bb.height))
    if x2 <= x1 or y2 <= y1:
        return None

    score = 0.0
    if best.categories:
        score = float(best.categories[0].score)

    crop_bgr, (cx1, cy1, cx2, cy2) = _crop_with_padding(bgr, x1, y1, x2, y2, pad=0.30)
    aligned = _resize_bgr(crop_bgr, 112)

    # Run landmarker on the original full image; coords are normalized.
    landmarks_xyz = np.zeros((478, 3), dtype=np.float32)
    lm_result = _FACE_LANDMARKER.detect(mp_image)  # type: ignore[union-attr]
    if lm_result.face_landmarks:
        first = lm_result.face_landmarks[0]
        for i, p in enumerate(first[:478]):
            landmarks_xyz[i] = (p.x * w, p.y * h, p.z * w)

    return FaceCrop(
        bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
        score=score,
        landmarks=landmarks_xyz,
        aligned_bgr=aligned,
    )
