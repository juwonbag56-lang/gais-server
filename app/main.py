"""
GAIS FastAPI server (PostgreSQL edition).

- SQLAlchemy Async + asyncpg
- Alembic migration on startup (RUN_MIGRATIONS_ON_STARTUP=true)
- Response shape 100% compatible with previous SQLite version:
  * /me, /missions, /missions/complete, /shop, /state/{user_id}, /events/{user_id},
    /health, /app/meta
- DB column names exp/score are mapped to response keys xp/stars (Flutter compatibility).
- Placeholder fields (stats/face_score/rank/unlocked/weak_points) are synthesized
  with stable defaults so existing Flutter APK keeps working unchanged.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .db import dispose_engine, get_db, session_scope
from .users import get_or_create_user_id

logger = logging.getLogger("gais")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Constants for response shape compatibility
# ---------------------------------------------------------------------------

DEFAULT_STATS: Dict[str, int] = {
    "skin": 50,
    "symmetry": 50,
    "vibe": 50,
    "style": 50,
    "cleanliness": 50,
}

# Catalog ordering for /missions response (deterministic).
DEFAULT_MISSION_CATALOG_ORDER: List[str] = [
    "skin_cleanse",
    "walk_15",
    "posture_reset",
    "journal",
    "fit_check",
]

# Per-mission stat_gain (display-only; not persisted in schema).
MISSION_STAT_GAIN: Dict[str, Dict[str, int]] = {
    "skin_cleanse": {"skin": 4, "cleanliness": 4},
    "walk_15": {"vibe": 4, "symmetry": 1},
    "posture_reset": {"symmetry": 3, "vibe": 2},
    "journal": {"vibe": 2, "style": 1},
    "fit_check": {"style": 4, "vibe": 2},
}


# ---------------------------------------------------------------------------
# Pydantic request models (kept from SQLite version)
# ---------------------------------------------------------------------------


class MissionCompleteRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    mission_id: str = Field(..., min_length=1)


class QuestCompleteRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    quest_id: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _level_from_xp(xp: int) -> int:
    if xp < 0:
        return 1
    return max(1, 1 + xp // 100)


def _ts(dt: Optional[datetime]) -> float:
    if dt is None:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _strip_user_prefix(mission_id_raw: str, handle: str) -> str:
    """
    Flutter는 /missions에서 받은 id(`<handle>:<key>`)를 그대로 /missions/complete에 보낸다.
    이를 missions.id(=key)로 정규화한다.
    """
    prefix = f"{handle}:"
    if mission_id_raw.startswith(prefix):
        return mission_id_raw[len(prefix):]
    return mission_id_raw


def _focus_for_user(_handle: str, completed_focuses: Dict[str, int]) -> str:
    """가장 많이 완료된 focus의 보색을 살짝 섞되, 단순화: 가장 적게 완료된 focus."""
    pool = ["skin", "mood", "style", "overall"]
    if not completed_focuses:
        return "skin"
    least = min(pool, key=lambda f: completed_focuses.get(f, 0))
    return least


# ---------------------------------------------------------------------------
# Lifespan: run migrations on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if settings.run_migrations_on_startup:
        try:
            from .migrate import run_alembic_upgrade

            await run_alembic_upgrade()
            logger.info("Alembic upgrade head: OK")
        except Exception:
            logger.exception("Alembic migration failed during startup")
            raise

        # 데모 유저 시드(없으면 생성)
        try:
            async with session_scope() as session:
                await get_or_create_user_id(session, settings.demo_user_handle)
        except Exception:
            logger.exception("Demo user seeding failed")
            raise

    yield

    await dispose_engine()


app = FastAPI(title="GAIS Server", version=get_settings().app_version, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
from .routers.analyze_face import router as analyze_face_router  # noqa: E402
app.include_router(analyze_face_router)


# ---------------------------------------------------------------------------
# Composition helpers (DB rows → API response shape)
# ---------------------------------------------------------------------------


async def _load_user_stats(session: AsyncSession, user_id: uuid.UUID) -> Dict[str, Any]:
    row = (
        await session.execute(
            text(
                """
                SELECT level, exp, score, currency, battlepass_level, mood,
                       last_quest_id, last_evaluation_source, updated_at
                FROM user_stats
                WHERE user_id = :uid
                """
            ),
            {"uid": user_id},
        )
    ).mappings().first()
    if row is None:
        return {
            "level": 1,
            "exp": 0,
            "score": 0,
            "currency": 0,
            "battlepass_level": 0,
            "mood": "neutral",
            "last_quest_id": "",
            "last_evaluation_source": "fallback",
            "updated_at": None,
        }
    return dict(row)


def _compose_state(handle: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    xp = int(stats.get("exp") or 0)
    return {
        "user_id": handle,
        "level": int(stats.get("level") or _level_from_xp(xp)),
        "xp": xp,
        "stars": int(stats.get("score") or 0),
        "currency": int(stats.get("currency") or 0),
        "battlepass_level": int(stats.get("battlepass_level") or 0),
        "stats": dict(DEFAULT_STATS),
        "mood": stats.get("mood") or "neutral",
        "face_score": 50,
        "rank": "D",
        "weak_points": [],
        "unlocked": [],
        "last_updated": _ts(stats.get("updated_at")),
        "last_quest_id": stats.get("last_quest_id") or "",
        "last_evaluation_source": stats.get("last_evaluation_source") or "fallback",
    }


def _compose_user_block(handle: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": handle,
        "name": "Server User",
        "level": int(stats.get("level") or _level_from_xp(int(stats.get("exp") or 0))),
        "xp": int(stats.get("exp") or 0),
        "stars": int(stats.get("score") or 0),
        "mood": stats.get("mood") or "neutral",
        "rank": "D",
        "stats": dict(DEFAULT_STATS),
        "unlocked": [],
        "battlepass_level": int(stats.get("battlepass_level") or 0),
        "currency": int(stats.get("currency") or 0),
    }


async def _log_event(
    session: AsyncSession,
    user_id: Optional[uuid.UUID],
    event_name: str,
    payload: Dict[str, Any],
) -> None:
    await session.execute(
        text(
            """
            INSERT INTO app_events (user_id, event_name, event_data)
            VALUES (:uid, :name, CAST(:data AS JSONB))
            """
        ),
        {"uid": user_id, "name": event_name, "data": _json_dumps(payload)},
    )


def _json_dumps(payload: Dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "ts": time.time()}


@app.get("/app/meta")
async def app_meta() -> Dict[str, Any]:
    s = get_settings()
    return {
        "app": "GAIS",
        "version": s.app_version,
        "db_path": "postgres",
        "vision_enabled": bool(s.vision_api_url and s.vision_api_key),
    }


@app.get("/me")
async def me(session: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    s = get_settings()
    handle = s.demo_user_handle
    user_id = await get_or_create_user_id(session, handle)
    stats = await _load_user_stats(session, user_id)
    return {
        "user": _compose_user_block(handle, stats),
        "state": _compose_state(handle, stats),
    }


@app.get("/state/{user_handle}")
async def state(user_handle: str, session: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    user_id = await get_or_create_user_id(session, user_handle)
    stats = await _load_user_stats(session, user_id)
    return _compose_state(user_handle, stats)


@app.get("/events/{user_handle}")
async def events(
    user_handle: str, session: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    user_id = await get_or_create_user_id(session, user_handle)
    rows = (
        await session.execute(
            text(
                """
                SELECT event_name, event_data, created_at
                FROM app_events
                WHERE user_id = :uid
                ORDER BY created_at DESC
                LIMIT 50
                """
            ),
            {"uid": user_id},
        )
    ).mappings().all()

    items: List[Dict[str, Any]] = []
    for r in rows:
        items.append(
            {
                "event_type": r["event_name"],
                "payload": dict(r["event_data"] or {}),
                "created_at": _ts(r["created_at"]),
            }
        )
    return {"user_id": user_handle, "events": items}


@app.get("/missions")
async def list_missions(session: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    s = get_settings()
    handle = s.demo_user_handle
    user_id = await get_or_create_user_id(session, handle)

    # 카탈로그 (active=TRUE)
    catalog_rows = (
        await session.execute(
            text(
                """
                SELECT id, title, focus, xp_reward, shard_reward
                FROM missions
                WHERE active = TRUE
                """
            )
        )
    ).mappings().all()
    catalog = {row["id"]: dict(row) for row in catalog_rows}

    # 정렬 우선순위: 미리 정의한 순서 먼저, 그다음 알파벳
    def _order_key(mid: str) -> tuple:
        try:
            return (0, DEFAULT_MISSION_CATALOG_ORDER.index(mid))
        except ValueError:
            return (1, mid)

    # 이미 완료한 미션
    completed_rows = (
        await session.execute(
            text(
                """
                SELECT mission_id
                FROM mission_completions
                WHERE user_id = :uid
                """
            ),
            {"uid": user_id},
        )
    ).all()
    completed_ids = {r[0] for r in completed_rows}

    # 오늘(=UTC 자정 기준) 완료 카운트
    today_count = (
        await session.execute(
            text(
                """
                SELECT COUNT(*) FROM mission_completions
                WHERE user_id = :uid
                  AND created_at >= date_trunc('day', now() AT TIME ZONE 'UTC')
                """
            ),
            {"uid": user_id},
        )
    ).scalar_one()

    focus_counts_rows = (
        await session.execute(
            text(
                """
                SELECT m.focus, COUNT(*)
                FROM mission_completions mc
                JOIN missions m ON m.id = mc.mission_id
                WHERE mc.user_id = :uid
                GROUP BY m.focus
                """
            ),
            {"uid": user_id},
        )
    ).all()
    focus_counts = {fr[0]: int(fr[1]) for fr in focus_counts_rows}
    focus = _focus_for_user(handle, focus_counts)

    open_ids = sorted([mid for mid in catalog.keys() if mid not in completed_ids], key=_order_key)
    missions: List[Dict[str, Any]] = []
    for mid in open_ids:
        meta = catalog[mid]
        missions.append(
            {
                "id": f"{handle}:{mid}",
                "quest_key": mid,
                "title": meta["title"],
                "xp": int(meta["xp_reward"]),
                "stars": int(meta["shard_reward"]),
                "stat_gain": MISSION_STAT_GAIN.get(mid, {}),
            }
        )

    await _log_event(session, user_id, "list_missions", {"focus": focus})

    return {
        "user_id": handle,
        "today_completed": int(today_count),
        "today_total": len(catalog),
        "missions": missions,
        "focus": focus,
    }


async def _complete_mission_core(
    session: AsyncSession, user_handle: str, mission_id_raw: str
) -> Dict[str, Any]:
    mission_key = _strip_user_prefix(mission_id_raw, user_handle)
    user_id = await get_or_create_user_id(session, user_handle)

    # 카탈로그 검증
    mission_row = (
        await session.execute(
            text(
                """
                SELECT id, title, focus, xp_reward, shard_reward
                FROM missions
                WHERE id = :mid AND active = TRUE
                """
            ),
            {"mid": mission_key},
        )
    ).mappings().first()
    if mission_row is None:
        raise HTTPException(status_code=404, detail=f"Unknown mission_id: {mission_key}")

    xp_reward = int(mission_row["xp_reward"])
    shard_reward = int(mission_row["shard_reward"])

    # 멱등 INSERT: 이미 완료했으면 보상은 0으로 처리(이중 보상 방지)
    inserted = (
        await session.execute(
            text(
                """
                INSERT INTO mission_completions (user_id, mission_id, xp_gained, shards_gained, raw_response)
                VALUES (:uid, :mid, :xp, :shards, '{}'::jsonb)
                ON CONFLICT (user_id, mission_id) DO NOTHING
                RETURNING id
                """
            ),
            {
                "uid": user_id,
                "mid": mission_key,
                "xp": xp_reward,
                "shards": shard_reward,
            },
        )
    ).first()
    is_new = inserted is not None

    if is_new:
        # 보상 적용
        await session.execute(
            text(
                """
                UPDATE user_stats
                SET exp = exp + :xp,
                    score = score + :shards,
                    last_quest_id = :mid,
                    last_evaluation_source = 'gais',
                    last_activity_at = now()
                WHERE user_id = :uid
                """
            ),
            {"uid": user_id, "xp": xp_reward, "shards": shard_reward, "mid": mission_key},
        )
        # level 재계산
        cur = (
            await session.execute(
                text("SELECT exp FROM user_stats WHERE user_id = :uid"),
                {"uid": user_id},
            )
        ).first()
        new_xp = int(cur[0]) if cur else 0
        new_level = _level_from_xp(new_xp)
        await session.execute(
            text("UPDATE user_stats SET level = :lv WHERE user_id = :uid"),
            {"uid": user_id, "lv": new_level},
        )

    stats = await _load_user_stats(session, user_id)
    state_block = _compose_state(user_handle, stats)

    payload = {
        "mission_id": mission_id_raw,
        "quest_key": mission_key,
        "user_id": user_handle,
        "idempotent_replay": (not is_new),
    }
    await _log_event(session, user_id, "complete_quest", payload)

    return {
        "ok": True,
        "user_id": user_handle,
        "quest_id": mission_id_raw,
        "mission_id": mission_id_raw,
        "gained_xp": xp_reward if is_new else 0,
        "gained_stars": shard_reward if is_new else 0,
        "reward_text": (
            f"보상 획득: XP +{xp_reward}, 별파편 +{shard_reward}." if is_new
            else "이미 완료한 미션입니다."
        ),
        "state": state_block,
        "idempotent_replay": (not is_new),
    }


@app.post("/missions/complete")
async def missions_complete(
    body: MissionCompleteRequest, session: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    return await _complete_mission_core(session, body.user_id, body.mission_id)


@app.post("/quests/complete")
async def quests_complete(
    body: QuestCompleteRequest, session: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    return await _complete_mission_core(session, body.user_id, body.quest_id)


@app.get("/shop")
async def shop(session: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    s = get_settings()
    handle = s.demo_user_handle
    user_id = await get_or_create_user_id(session, handle)

    rows = (
        await session.execute(
            text(
                """
                SELECT c.id, c.name, c.price, c.required_level,
                       (uc.id IS NOT NULL) AS owned
                FROM cosmetics c
                LEFT JOIN user_cosmetics uc
                  ON uc.cosmetic_id = c.id AND uc.user_id = :uid
                WHERE c.active = TRUE
                ORDER BY c.required_level ASC, c.id ASC
                """
            ),
            {"uid": user_id},
        )
    ).mappings().all()

    items = [
        {
            "id": r["id"],
            "name": r["name"],
            "cost": int(r["price"]),
            "required_level": int(r["required_level"]),
            "owned": bool(r["owned"]),
        }
        for r in rows
    ]
    stats = await _load_user_stats(session, user_id)
    return {
        "user_id": handle,
        "items": items,
        "currency": int(stats.get("score") or 0),
    }


# ---------------------------------------------------------------------------
# Error handler (consistent shape with previous version)
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
def http_exc_handler(_: Any, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )
