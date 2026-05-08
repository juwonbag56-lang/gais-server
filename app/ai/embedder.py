"""
Lightweight face embedding.

Default backend uses MediaPipe FaceMesh's 478 3D landmarks. We canonicalise
them into a translation/scale invariant 1434-d vector. This avoids any extra
model download (Render-Free safe) and keeps memory < 50 MB.

The interface (`Embedder.embed(face) -> np.ndarray`) is stable, so once H100
training produces a stronger ArcFace/MobileFaceNet ONNX or PyTorch model, we
can swap the backend by setting `GAIS_EMBEDDER_BACKEND=onnx` and pointing
`GAIS_EMBEDDER_MODEL_PATH` at the new file.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from .detector import FaceCrop

logger = logging.getLogger("gais.ai.embedder")

EMBEDDING_DIM_LANDMARK = 478 * 3  # 1434
EMBEDDING_VERSION_LANDMARK = "landmark-mp-v1"


def _normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """Translation + scale + roll-rotation invariant landmark vector.

    1) translate so the centroid is at origin
    2) rotate so the inter-ocular line is horizontal (uses landmarks 33, 263)
    3) scale so the inter-pupil distance == 1.0
    """
    if lm.shape != (478, 3):
        return np.zeros(EMBEDDING_DIM_LANDMARK, dtype=np.float32)

    # MediaPipe FaceMesh: 33 = right eye outer, 263 = left eye outer
    p_right = lm[33, :2]
    p_left = lm[263, :2]
    eye_vec = p_left - p_right
    inter_eye = float(np.linalg.norm(eye_vec))
    if inter_eye < 1e-3:
        return np.zeros(EMBEDDING_DIM_LANDMARK, dtype=np.float32)

    # rotation
    angle = np.arctan2(eye_vec[1], eye_vec[0])
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)

    centroid = lm[:, :2].mean(axis=0)
    xy = lm[:, :2] - centroid
    xy = xy @ rot.T  # rotate
    xy /= inter_eye

    z = lm[:, 2:3] / inter_eye

    out = np.concatenate([xy, z], axis=1).astype(np.float32)
    return out.reshape(-1)


class Embedder:
    """Stable interface around the chosen backend."""

    def __init__(self) -> None:
        self.backend = os.environ.get("GAIS_EMBEDDER_BACKEND", "landmark").lower()
        self._onnx_session = None  # populated lazily if backend=='onnx'

    @property
    def version(self) -> str:
        if self.backend == "onnx":
            return os.environ.get("GAIS_EMBEDDER_MODEL_VERSION", "onnx-custom")
        return EMBEDDING_VERSION_LANDMARK

    @property
    def dim(self) -> int:
        if self.backend == "onnx":
            # the real dim is determined at first inference; default to 512
            return int(os.environ.get("GAIS_EMBEDDER_DIM", "512"))
        return EMBEDDING_DIM_LANDMARK

    def embed(self, face: FaceCrop) -> np.ndarray:
        if self.backend == "onnx":
            return self._embed_onnx(face)
        return _normalize_landmarks(face.landmarks)

    # ----- optional ONNX backend (kept for the future H100 model swap) -----
    def _embed_onnx(self, face: FaceCrop) -> np.ndarray:
        if self._onnx_session is None:
            try:
                import onnxruntime as ort  # lazy

                model_path = os.environ.get("GAIS_EMBEDDER_MODEL_PATH", "")
                if not model_path or not os.path.exists(model_path):
                    raise FileNotFoundError(f"Embedder ONNX model not found: {model_path!r}")
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                self._onnx_session = ort.InferenceSession(
                    model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
                )
                logger.info("ONNX embedder loaded: %s", model_path)
            except Exception as exc:  # noqa: BLE001
                logger.exception("ONNX embedder unavailable, falling back to landmark backend: %s", exc)
                self.backend = "landmark"
                return _normalize_landmarks(face.landmarks)

        # Standard ArcFace-style preprocessing: 112x112 BGR -> RGB, /127.5 - 1
        img = face.aligned_bgr[:, :, ::-1].astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW

        sess = self._onnx_session
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: img})[0][0]
        out = out.astype(np.float32)
        # L2 normalise
        n = float(np.linalg.norm(out))
        if n > 1e-6:
            out = out / n
        return out


# module-level singleton
_GLOBAL: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = Embedder()
    return _GLOBAL
