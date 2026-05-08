"""
POST /analyze-face

Multipart form: image=<file>, optional user_id=<handle> (defaults to demo handle).

Pipeline:
  1) Read bytes from request, validate size & MIME
  2) Compute SHA-256 of original bytes
  3) Run app.ai.pipeline.analyze_image_bytes()
  4) Insert row into user_face_uploads (s3_key = 'memory://<sha>', is_deleted=true)
  5) Insert row into face_analysis_results
  6) Insert app_events row ('analyze_face')
  7) Insert ai_analysis_runs row (status='succeeded')
  8) Return JSON with shape:
        {
          "success": true,
          "analysis": {
            "attractiveness_score": 0.74,
            "confidence_score": 0.81,
            "elo_rating": 1500,
            "processing_time_ms": 93,
            "top_positive_features": [...],
            "top_negative_features": [...],
            "model_version": "gais-v14-prototype"
          }
        }

Error shape (HTTP 4xx):
        { "success": false, "error": { "code": "...", "message": "..." } }

Original image bytes are never persisted to disk; they live only in the request
buffer and are released when the function returns.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..ai.pipeline import (
    ALLOWED_MIMES,
    MAX_UPLOAD_BYTES,
    FaceNotFoundError,
    InvalidImageError,
    analyze_image_bytes,
)
from ..config import get_settings
from ..db import get_db
from ..users import get_or_create_user_id

logger = logging.getLogger("gais.routers.analyze_face")

router = APIRouter()


def _err(code: str, message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}},
    )


@router.post("/analyze-face")
async def analyze_face(
    image: UploadFile = File(...),
    user_id: Optional[str] = Form(default=None),
    session: AsyncSession = Depends(get_db),
) -> Any:
    settings = get_settings()
    handle = (user_id or "").strip() or settings.demo_user_handle

    # 1. Validate MIME quickly (we re-validate by magic bytes via decoder later)
    mime_type = (image.content_type or "").lower()
    if mime_type and mime_type not in ALLOWED_MIMES:
        return _err("invalid_mime", f"Unsupported content_type: {mime_type}", 415)

    # 2. Read bytes (limit enforced by pipeline; also short-circuit here)
    raw = await image.read()
    if not raw:
        return _err("empty_payload", "Empty image upload", 400)
    if len(raw) > MAX_UPLOAD_BYTES:
        return _err(
            "image_too_large",
            f"Image is {len(raw)} bytes; maximum allowed is {MAX_UPLOAD_BYTES}.",
            413,
        )

    # 3. Resolve internal user UUID
    db_user_id = await get_or_create_user_id(session, handle)

    started_ts = time.time()

    # Pre-create ai_analysis_runs row so we always have a trace row
    run_row = (
        await session.execute(
            text(
                """
                INSERT INTO ai_analysis_runs (user_id, input_payload, model_name, status)
                VALUES (:uid, CAST(:payload AS JSONB), :model, 'queued')
                RETURNING id
                """
            ),
            {
                "uid": db_user_id,
                "payload": json.dumps(
                    {
                        "size_bytes": len(raw),
                        "mime_type": mime_type or "unknown",
                        "started_ts": started_ts,
                    },
                    ensure_ascii=False,
                ),
                "model": "gais-v14-prototype",
            },
        )
    ).first()
    run_id = run_row[0] if run_row else None

    # 4. Run AI pipeline (sync; fast). Catch domain errors -> 4xx; ensure rollback.
    try:
        analysis, upload_meta = analyze_image_bytes(raw, mime_type=mime_type or None)
    except FaceNotFoundError as exc:
        # mark run failed
        if run_id is not None:
            await session.execute(
                text(
                    """
                    UPDATE ai_analysis_runs
                    SET status='failed', error_message=:msg, output_payload='{}'::jsonb
                    WHERE id=:rid
                    """
                ),
                {"rid": run_id, "msg": str(exc)},
            )
        return _err("face_not_found", str(exc) or "No face detected", 422)
    except InvalidImageError as exc:
        if run_id is not None:
            await session.execute(
                text(
                    """
                    UPDATE ai_analysis_runs
                    SET status='failed', error_message=:msg
                    WHERE id=:rid
                    """
                ),
                {"rid": run_id, "msg": str(exc)},
            )
        return _err("invalid_image", str(exc) or "Invalid image", 400)
    except Exception:
        logger.exception("analyze_face pipeline crashed")
        if run_id is not None:
            await session.execute(
                text(
                    """
                    UPDATE ai_analysis_runs
                    SET status='failed', error_message='internal_error'
                    WHERE id=:rid
                    """
                ),
                {"rid": run_id},
            )
        # Surface a stable JSON error
        return _err("internal_error", "Analysis failed", 500)
    finally:
        # Free original bytes ASAP (image content is never persisted on disk).
        del raw

    # 5. Persist user_face_uploads (no S3 in Render-Free; store memory:// pseudo-key)
    sha = upload_meta["sha256"]
    pseudo_key = f"memory://{sha}"
    upload_row = (
        await session.execute(
            text(
                """
                INSERT INTO user_face_uploads
                    (user_id, s3_key, mime_type, file_size_bytes,
                     width, height, sha256, is_deleted, deleted_at)
                VALUES
                    (:uid, :key, :mime, :size, :w, :h, :sha, TRUE, now())
                RETURNING id
                """
            ),
            {
                "uid": db_user_id,
                "key": pseudo_key,
                "mime": upload_meta["mime_type"],
                "size": upload_meta["file_size_bytes"],
                "w": upload_meta["width"],
                "h": upload_meta["height"],
                "sha": sha,
            },
        )
    ).first()
    upload_id = upload_row[0] if upload_row else None

    # 6. Persist face_analysis_results
    meta_payload = {
        "bbox_xyxy": upload_meta["bbox_xyxy"],
        "detection_score": upload_meta["detection_score"],
        "embedder_version": analysis["embedder_version"],
        "embedding_dim": analysis["embedding_dim"],
        "feature_names": analysis["feature_names"],
        "processing_time_ms": analysis["processing_time_ms"],
    }
    analysis_row = (
        await session.execute(
            text(
                """
                INSERT INTO face_analysis_results
                    (user_id, upload_id, score, confidence, uncertainty, elo_rating,
                     model_version, explanation_vector, top_positive_features,
                     top_negative_features, meta)
                VALUES
                    (:uid, :upload, :score, :conf, :unc, :elo,
                     :model, CAST(:exp AS JSONB), CAST(:pos AS JSONB),
                     CAST(:neg AS JSONB), CAST(:meta AS JSONB))
                RETURNING id
                """
            ),
            {
                "uid": db_user_id,
                "upload": upload_id,
                "score": float(analysis["attractiveness_score"]),
                "conf": float(analysis["confidence_score"]),
                "unc": float(analysis["uncertainty"]),
                "elo": int(analysis["elo_rating"]),
                "model": analysis["model_version"],
                "exp": json.dumps(analysis["explanation_vector"]),
                "pos": json.dumps(analysis["top_positive_features"]),
                "neg": json.dumps(analysis["top_negative_features"]),
                "meta": json.dumps(meta_payload, ensure_ascii=False),
            },
        )
    ).first()
    analysis_id = analysis_row[0] if analysis_row else None

    # 7. Mark ai_analysis_runs as succeeded
    if run_id is not None:
        await session.execute(
            text(
                """
                UPDATE ai_analysis_runs
                SET status='succeeded',
                    output_payload = CAST(:out AS JSONB)
                WHERE id=:rid
                """
            ),
            {
                "rid": run_id,
                "out": json.dumps(
                    {
                        "analysis_id": str(analysis_id) if analysis_id else None,
                        "upload_id": str(upload_id) if upload_id else None,
                        "score": analysis["attractiveness_score"],
                        "confidence": analysis["confidence_score"],
                        "processing_time_ms": analysis["processing_time_ms"],
                    },
                    ensure_ascii=False,
                ),
            },
        )

    # 8. Log app event
    await session.execute(
        text(
            """
            INSERT INTO app_events (user_id, event_name, event_data)
            VALUES (:uid, 'analyze_face', CAST(:data AS JSONB))
            """
        ),
        {
            "uid": db_user_id,
            "data": json.dumps(
                {
                    "analysis_id": str(analysis_id) if analysis_id else None,
                    "score": analysis["attractiveness_score"],
                    "confidence": analysis["confidence_score"],
                    "processing_time_ms": analysis["processing_time_ms"],
                    "model_version": analysis["model_version"],
                },
                ensure_ascii=False,
            ),
        },
    )

    # 9. Final response (shape strictly per spec)
    return {
        "success": True,
        "analysis": {
            "attractiveness_score": analysis["attractiveness_score"],
            "confidence_score": analysis["confidence_score"],
            "elo_rating": int(analysis["elo_rating"]),
            "processing_time_ms": int(analysis["processing_time_ms"]),
            "top_positive_features": analysis["top_positive_features"],
            "top_negative_features": analysis["top_negative_features"],
            "model_version": analysis["model_version"],
        },
    }
