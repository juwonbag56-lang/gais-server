"""
End-to-end image -> analysis bundle. Single entry point used by the FastAPI router.

Outputs a dict that maps cleanly into both the API response and the DB rows
(face_analysis_results / user_face_uploads).
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, Tuple

from .detector import decode_image_bytes, detect_face
from .embedder import get_embedder
from .scorer import score_face

logger = logging.getLogger("gais.ai.pipeline")


class FaceNotFoundError(Exception):
    """Raised when no face can be detected in the input image."""


class InvalidImageError(Exception):
    """Raised when the input bytes cannot be decoded into an image."""


# Render-Free safe upload limit
MAX_UPLOAD_BYTES = 4 * 1024 * 1024  # 4 MiB
ALLOWED_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


def analyze_image_bytes(buf: bytes, mime_type: str | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the full pipeline.

    Returns (analysis_dict, upload_meta_dict).
    `analysis_dict` has the API-shape fields (no DB-only fields).
    `upload_meta_dict` carries info needed to write user_face_uploads.
    """
    if buf is None or len(buf) == 0:
        raise InvalidImageError("Empty image payload")
    if len(buf) > MAX_UPLOAD_BYTES:
        raise InvalidImageError(
            f"Image too large ({len(buf)} bytes), max {MAX_UPLOAD_BYTES} bytes"
        )
    if mime_type and mime_type.lower() not in ALLOWED_MIMES:
        raise InvalidImageError(f"Unsupported MIME type: {mime_type}")

    t0 = time.perf_counter()

    sha256 = hashlib.sha256(buf).hexdigest()

    try:
        bgr = decode_image_bytes(buf)
    except ValueError as exc:
        raise InvalidImageError(str(exc)) from exc

    h, w = bgr.shape[:2]

    face = detect_face(bgr)
    if face is None:
        raise FaceNotFoundError("No face detected")

    embedder = get_embedder()
    embedding = embedder.embed(face)
    embedding_dim = int(embedding.shape[0]) if embedding is not None else 0

    score_pack = score_face(face)

    elapsed_ms = int(round((time.perf_counter() - t0) * 1000))

    analysis = {
        "attractiveness_score": score_pack["attractiveness_score"],
        "confidence_score": score_pack["confidence_score"],
        "uncertainty": score_pack["uncertainty"],
        "elo_rating": 1500,  # cold-start default; updated by elo service later
        "processing_time_ms": elapsed_ms,
        "top_positive_features": score_pack["top_positive_features"],
        "top_negative_features": score_pack["top_negative_features"],
        "explanation_vector": score_pack["explanation_vector"],
        "feature_names": score_pack["feature_names"],
        "model_version": score_pack["model_version"],
        "embedder_version": embedder.version,
        "embedding_dim": embedding_dim,
    }

    upload_meta = {
        "sha256": sha256,
        "mime_type": mime_type or "application/octet-stream",
        "file_size_bytes": len(buf),
        "width": int(w),
        "height": int(h),
        "bbox_xyxy": list(map(int, face.bbox_xyxy)),
        "detection_score": float(face.score),
    }

    return analysis, upload_meta
