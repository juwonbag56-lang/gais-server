"""GAIS / Ascend backend.

Single-file FastAPI server with:
- persistent user state (SQLite)
- face evaluation (deterministic fallback, optional OpenAI-compatible vision hook)
- evolution preview
- quest generation and completion
- battle / unlock flow
- compatibility endpoints for the existing Ascend Flutter client

Run:
    pip install fastapi uvicorn pydantic
    uvicorn gais_fastapi_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

APP_NAME = "GAIS"
APP_VERSION = "2.0.0"
DB_PATH = Path(os.getenv("GAIS_DB_PATH", "./gais.db"))
LOG_JSON = os.getenv("GAIS_JSON_LOG", "1") == "1"
MAX_LEVEL = int(os.getenv("GAIS_MAX_LEVEL", "100"))
DEFAULT_XP_PER_QUEST = int(os.getenv("GAIS_XP_PER_QUEST", "25"))
DEFAULT_STAR_PER_QUEST = int(os.getenv("GAIS_STAR_PER_QUEST", "5"))
VISION_API_URL = os.getenv("VISION_API_URL", "").strip()
VISION_API_KEY = os.getenv("VISION_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
VISION_MODEL = os.getenv("VISION_MODEL", os.getenv("MODEL_NAME", "gpt-4.1-mini"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("GAIS_CORS_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="GAIS API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in ALLOWED_ORIGINS else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS quest_completions (
    quest_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


@contextmanager
def db_conn() -> Iterable[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db_conn() as conn:
        conn.executescript(SCHEMA_SQL)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class FaceEvaluateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    image_ref: str = Field(..., min_length=1, description="image url / local path / asset id")
    gender: Optional[Literal["male", "female", "other"]] = None


class EvolutionRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    direction: Literal["style", "skin", "mood", "overall"] = "overall"
    days: int = Field(7, ge=1, le=90)


class QuestRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    focus: Optional[Literal["style", "skin", "mood", "overall"]] = None


class QuestCompleteRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    quest_id: str = Field(..., min_length=1)


class BattleRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    opponent_rank: int = Field(..., ge=1, le=100)


class UnlockRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    item_id: str = Field(..., min_length=1)


class ResetRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


class MissionCompleteRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    mission_id: str = Field(..., min_length=1)


class FaceState(BaseModel):
    skin: int = 50
    symmetry: int = 50
    vibe: int = 50
    style: int = 50
    cleanliness: int = 50


class UserProfile(BaseModel):
    user_id: str
    level: int = 1
    xp: int = 0
    stars: int = 0
    face_score: int = 50
    rank: str = "D"
    stats: FaceState = Field(default_factory=FaceState)
    weak_points: List[str] = Field(default_factory=list)
    unlocked: List[str] = Field(default_factory=list)
    mood: str = "neutral"
    battlepass_level: int = 0
    currency: int = 0
    last_updated: float = 0.0
    last_quest_id: str = ""
    last_evaluation_source: str = "fallback"


class GAISStateResponse(BaseModel):
    user_id: str
    level: int
    xp: int
    stars: int
    currency: int
    battlepass_level: int
    stats: Dict[str, int]
    mood: str
    face_score: int
    rank: str
    weak_points: List[str]
    unlocked: List[str]
    last_updated: float
    last_quest_id: str
    last_evaluation_source: str


# -----------------------------------------------------------------------------
# App bootstrap
# -----------------------------------------------------------------------------

app_ready_at = time.time()
init_db()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

RANKS: List[Tuple[int, str]] = [
    (95, "S"),
    (85, "A"),
    (75, "B"),
    (60, "C"),
    (45, "D"),
    (0, "E"),
]

MOOD_BY_RANK = {
    "S": "radiant",
    "A": "determined",
    "B": "focused",
    "C": "neutral",
    "D": "weary",
    "E": "tired",
}

UNLOCKS_BY_LEVEL = {
    3: ["hair_style_01"],
    5: ["glasses_01"],
    10: ["premium_style_01"],
    20: ["battle_pass_skin_01"],
}


def default_stats() -> Dict[str, int]:
    return {
        "skin": 50,
        "symmetry": 50,
        "vibe": 50,
        "style": 50,
        "cleanliness": 50,
    }


@dataclass
class CompletionResult:
    quest_id: str
    gained_xp: int
    gained_stars: int
    title: str
    stat_gain: Dict[str, int]


class GAISCore:
    def load_user(self, user_id: str) -> UserProfile:
        with db_conn() as conn:
            row = conn.execute("SELECT payload FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            profile = UserProfile(user_id=user_id)
            profile.stats = FaceState(**default_stats())
            self.save_user(profile)
            return profile
        payload = json.loads(row["payload"])
        if isinstance(payload.get("stats"), dict):
            payload["stats"] = FaceState(**{**default_stats(), **payload["stats"]})
        return UserProfile.model_validate(payload)

    def save_user(self, profile: UserProfile) -> None:
        profile.last_updated = time.time()
        with db_conn() as conn:
            conn.execute(
                """
                INSERT INTO users(user_id, payload, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (profile.user_id, json.dumps(profile.model_dump(), ensure_ascii=False), profile.last_updated),
            )

    def log_event(self, user_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO events(user_id, event_type, payload, created_at) VALUES(?, ?, ?, ?)",
                (user_id, event_type, json.dumps(payload, ensure_ascii=False), time.time()),
            )

    def _rank_from_score(self, score: int) -> str:
        for threshold, rank in RANKS:
            if score >= threshold:
                return rank
        return "E"

    def _level_from_xp(self, xp: int) -> int:
        return min(MAX_LEVEL, 1 + xp // 100)

    def _weak_points(self, stats: Dict[str, int]) -> List[str]:
        return [k for k, _ in sorted(stats.items(), key=lambda kv: kv[1])[:3]]

    def _seed(self, *parts: str) -> int:
        digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
        return int(digest[:12], 16)

    def _state_dict(self, profile: UserProfile) -> Dict[str, Any]:
        return {
            "user_id": profile.user_id,
            "level": profile.level,
            "xp": profile.xp,
            "stars": profile.stars,
            "currency": profile.currency,
            "battlepass_level": profile.battlepass_level,
            "stats": profile.stats.model_dump(),
            "mood": profile.mood,
            "face_score": profile.face_score,
            "rank": profile.rank,
            "weak_points": profile.weak_points,
            "unlocked": profile.unlocked,
            "last_updated": profile.last_updated,
            "last_quest_id": profile.last_quest_id,
            "last_evaluation_source": profile.last_evaluation_source,
        }

    def _persist_profile_after_stat_update(self, profile: UserProfile) -> None:
        profile.face_score = int(round(sum(profile.stats.model_dump().values()) / len(profile.stats.model_dump())))
        profile.rank = self._rank_from_score(profile.face_score)
        profile.weak_points = self._weak_points(profile.stats.model_dump())
        profile.mood = MOOD_BY_RANK[profile.rank]
        profile.level = self._level_from_xp(profile.xp)
        profile.battlepass_level = max(profile.battlepass_level, profile.level)
        for threshold, items in UNLOCKS_BY_LEVEL.items():
            if profile.level >= threshold:
                for item in items:
                    if item not in profile.unlocked:
                        profile.unlocked.append(item)
        self.save_user(profile)

    def _quest_catalog(self, user_id: str, focus: str, weak_points: List[str]) -> List[Dict[str, Any]]:
        base: Dict[str, List[Dict[str, Any]]] = {
            "skin": [
                self._quest(user_id, "skin_cleanse", "세안 루틴 2회", 25, 5, {"skin": 4, "cleanliness": 4}),
                self._quest(user_id, "water_habit", "물 2L 마시기", 20, 3, {"skin": 2, "cleanliness": 2}),
                self._quest(user_id, "sleep_reset", "수면 7시간 확보", 30, 5, {"skin": 3, "vibe": 3}),
            ],
            "style": [
                self._quest(user_id, "fit_check", "오늘 옷핏 점검", 25, 4, {"style": 4, "vibe": 2}),
                self._quest(user_id, "hair_fix", "헤어 정리 10분", 20, 3, {"style": 3, "cleanliness": 1}),
                self._quest(user_id, "accessory_test", "안경/악세서리 한 번 착용해보기", 30, 5, {"style": 3, "vibe": 2}),
            ],
            "mood": [
                self._quest(user_id, "walk_15", "15분 걷기", 20, 3, {"vibe": 4, "symmetry": 1}),
                self._quest(user_id, "posture_reset", "자세 교정 5분", 25, 4, {"symmetry": 3, "vibe": 2}),
                self._quest(user_id, "journal", "짧게 기록 3줄", 15, 2, {"vibe": 2, "style": 1}),
            ],
            "overall": [
                self._quest(user_id, "basic_reset", "세안 + 수면 + 물 루틴", 35, 6, {"skin": 2, "cleanliness": 2, "vibe": 2}),
                self._quest(user_id, "style_reset", "헤어 + 옷 정리", 30, 5, {"style": 3, "cleanliness": 2}),
                self._quest(user_id, "confidence", "자기 점검 5분", 20, 3, {"vibe": 3, "symmetry": 1}),
            ],
        }
        quests = list(base[focus])
        if weak_points:
            quests[0]["focus_hint"] = weak_points[0]
        return quests

    def _quest(self, user_id: str, quest_key: str, title: str, xp: int, stars: int, stat_gain: Dict[str, int]) -> Dict[str, Any]:
        return {
            "id": f"{user_id}:{quest_key}",
            "quest_key": quest_key,
            "title": title,
            "xp": xp,
            "stars": stars,
            "stat_gain": stat_gain,
        }

    def _quest_by_id(self, user_id: str, quest_id: str) -> Optional[Dict[str, Any]]:
        for focus in ("skin", "style", "mood", "overall"):
            for quest in self._quest_catalog(user_id, focus, []):
                if quest["id"] == quest_id or quest["quest_key"] == quest_id:
                    return quest
        return None

    def _focus_from_weak_points(self, weak_or_stats: Any) -> str:
        if isinstance(weak_or_stats, list) and weak_or_stats:
            wp = weak_or_stats[0]
        elif isinstance(weak_or_stats, dict) and weak_or_stats:
            wp = min(weak_or_stats.items(), key=lambda kv: kv[1])[0]
        else:
            return "overall"
        mapping = {
            "skin": "skin",
            "cleanliness": "skin",
            "style": "style",
            "vibe": "mood",
            "symmetry": "mood",
        }
        return mapping.get(wp, "overall")

    def _future_story(self, direction: str, days: int, future_rank: str, future_stats: Dict[str, int]) -> str:
        focus_text = {
            "style": "스타일이 먼저 정돈되고",
            "skin": "피부 인상이 먼저 맑아지고",
            "mood": "전체 분위기가 더 선명해지고",
            "overall": "전반 인상이 고르게 상승합니다",
        }[direction]
        best = max(future_stats.items(), key=lambda kv: kv[1])[0]
        return f"{days}일 후에는 {focus_text}. 가장 강한 축은 {best}이고, 예상 등급은 {future_rank}입니다."

    def _evaluation_hint(self, rank: str, weak_points: List[str]) -> str:
        return f"현재는 {rank} 등급입니다. 우선 {', '.join(weak_points)}를 개선하면 가장 큰 변화가 납니다."

    def _all_unlockables(self) -> List[str]:
        return ["hair_style_01", "glasses_01", "premium_style_01", "battle_pass_skin_01"]

    def _vision_fallback(self, req: FaceEvaluateRequest) -> Dict[str, Any]:
        seed = self._seed(req.user_id, req.image_ref)
        face_score = 40 + (seed % 56)
        stats = {
            "skin": 35 + ((seed // 3) % 61),
            "symmetry": 35 + ((seed // 7) % 61),
            "vibe": 35 + ((seed // 11) % 61),
            "style": 35 + ((seed // 13) % 61),
            "cleanliness": 35 + ((seed // 17) % 61),
        }
        rank = self._rank_from_score(face_score)
        weak_points = self._weak_points(stats)
        return {
            "face_score": face_score,
            "rank": rank,
            "stats": stats,
            "weak_points": weak_points,
            "mood": MOOD_BY_RANK[rank],
            "source": "fallback",
        }

    def _vision_remote(self, req: FaceEvaluateRequest) -> Optional[Dict[str, Any]]:
        if not VISION_API_URL or not VISION_API_KEY:
            return None
        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict cosmetic/character evaluation engine. "
                        "Return only JSON with keys: face_score, rank, stats, weak_points, mood, hint. "
                        "stats must include skin, symmetry, vibe, style, cleanliness with integer 0-100."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Evaluate this user face image for GAIS. user_id={req.user_id}, gender={req.gender or 'unknown'}. "
                                "Focus on cosmetic / character growth, not identity or sensitive traits."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": req.image_ref},
                        },
                    ],
                },
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            VISION_API_URL.rstrip("/") + "/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VISION_API_KEY}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            if isinstance(content, str):
                parsed = json.loads(content)
            else:
                parsed = content
            stats = {
                k: int(parsed.get("stats", {}).get(k, 50))
                for k in ["skin", "symmetry", "vibe", "style", "cleanliness"]
            }
            face_score = int(parsed.get("face_score", int(sum(stats.values()) / len(stats))))
            rank = str(parsed.get("rank", self._rank_from_score(face_score)))
            weak_points = list(parsed.get("weak_points", self._weak_points(stats)))
            mood = str(parsed.get("mood", MOOD_BY_RANK.get(rank, "neutral")))
            hint = str(parsed.get("hint", ""))
            return {
                "face_score": face_score,
                "rank": rank,
                "stats": stats,
                "weak_points": weak_points,
                "mood": mood,
                "hint": hint,
                "source": "vision_api",
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    def evaluate_face(self, req: FaceEvaluateRequest) -> Dict[str, Any]:
        profile = self.load_user(req.user_id)
        result = self._vision_remote(req) or self._vision_fallback(req)
        profile.face_score = int(result["face_score"])
        profile.rank = str(result["rank"])
        profile.stats = FaceState(**{**default_stats(), **result["stats"]})
        profile.weak_points = list(result["weak_points"])
        profile.mood = str(result["mood"])
        profile.last_evaluation_source = str(result.get("source", "fallback"))
        self._persist_profile_after_stat_update(profile)
        self.log_event(req.user_id, "evaluate_face", req.model_dump())
        return {
            "user_id": req.user_id,
            "face_score": profile.face_score,
            "rank": profile.rank,
            "stats": profile.stats.model_dump(),
            "weak_points": profile.weak_points,
            "mood": profile.mood,
            "hint": result.get("hint") or self._evaluation_hint(profile.rank, profile.weak_points),
            "source": profile.last_evaluation_source,
        }

    def evolve_future(self, req: EvolutionRequest) -> Dict[str, Any]:
        profile = self.load_user(req.user_id)
        stats = profile.stats.model_dump()
        target = dict(stats)
        delta_map = {
            "style": {"style": 10, "vibe": 4},
            "skin": {"skin": 10, "cleanliness": 5},
            "mood": {"vibe": 10, "symmetry": 3},
            "overall": {"skin": 6, "symmetry": 6, "vibe": 6, "style": 6, "cleanliness": 6},
        }
        for key, delta in delta_map[req.direction].items():
            target[key] = min(100, target[key] + delta)

        future_score = min(100, int(round(sum(target.values()) / len(target))))
        future_rank = self._rank_from_score(future_score)
        hairstyle = "soft_wave" if req.direction in ("style", "overall") else "clean_cut"
        glasses = "glasses_01" if profile.level >= 5 else "none"
        preview_prompt = (
            f"A {req.days}-day future-self preview for GAIS: direction={req.direction}, "
            f"rank={future_rank}, hairstyle={hairstyle}, glasses={glasses}, mood={MOOD_BY_RANK[future_rank]}."
        )
        self.log_event(req.user_id, "evolve_future", req.model_dump())
        return {
            "user_id": req.user_id,
            "direction": req.direction,
            "days": req.days,
            "future_score": future_score,
            "future_rank": future_rank,
            "future_stats": target,
            "preview_prompt": preview_prompt,
            "story": self._future_story(req.direction, req.days, future_rank, target),
        }

    def generate_quests(self, req: QuestRequest) -> Dict[str, Any]:
        profile = self.load_user(req.user_id)
        focus = req.focus or self._focus_from_weak_points(profile.weak_points or profile.stats.model_dump())
        quests = self._quest_catalog(req.user_id, focus, profile.weak_points)
        self.log_event(req.user_id, "generate_quests", req.model_dump())
        return {"user_id": req.user_id, "focus": focus, "quests": quests}

    def complete_quest(self, req: QuestCompleteRequest) -> Dict[str, Any]:
        existing = self._read_completion(req.quest_id)
        if existing is not None:
            return existing

        profile = self.load_user(req.user_id)
        quest = self._quest_by_id(req.user_id, req.quest_id)
        if quest is None:
            raise HTTPException(status_code=404, detail="Quest not found")

        gain_xp = int(quest.get("xp", DEFAULT_XP_PER_QUEST))
        gain_stars = int(quest.get("stars", DEFAULT_STAR_PER_QUEST))
        profile.xp += gain_xp
        profile.stars += gain_stars
        profile.currency += gain_stars
        profile.level = self._level_from_xp(profile.xp)
        profile.mood = "determined"
        profile.last_quest_id = req.quest_id

        stats = profile.stats.model_dump()
        for key, inc in quest["stat_gain"].items():
            stats[key] = min(100, stats.get(key, 50) + int(inc))
        profile.stats = FaceState(**stats)
        self._persist_profile_after_stat_update(profile)
        self._store_completion(req.quest_id, req.user_id, {"quest": quest, "state": self._state_dict(profile)})
        self.log_event(req.user_id, "complete_quest", req.model_dump())
        return {
            "user_id": req.user_id,
            "quest_id": req.quest_id,
            "gained_xp": gain_xp,
            "gained_stars": gain_stars,
            "state": self._state_dict(profile),
            "reward_text": self._reward_text(gain_xp, gain_stars, profile.level),
        }

    def battle(self, req: BattleRequest) -> Dict[str, Any]:
        profile = self.load_user(req.user_id)
        power = profile.face_score + profile.level * 2 + profile.stars // 10
        opponent_power = req.opponent_rank * 2 + 40
        win = power >= opponent_power
        if win:
            profile.stars += 10
            profile.currency += 10
            profile.xp += 15
            outcome = "win"
        else:
            profile.xp += 5
            profile.mood = "motivated"
            outcome = "loss"
        profile.level = self._level_from_xp(profile.xp)
        self._persist_profile_after_stat_update(profile)
        self.log_event(req.user_id, "battle", req.model_dump())
        return {
            "user_id": req.user_id,
            "outcome": outcome,
            "power": power,
            "opponent_power": opponent_power,
            "state": self._state_dict(profile),
        }

    def unlock_item(self, req: UnlockRequest) -> Dict[str, Any]:
        profile = self.load_user(req.user_id)
        if req.item_id not in self._all_unlockables():
            raise HTTPException(status_code=400, detail="Unknown item")
        if req.item_id not in profile.unlocked:
            profile.unlocked.append(req.item_id)
            self.save_user(profile)
        self.log_event(req.user_id, "unlock_item", req.model_dump())
        return {"user_id": req.user_id, "unlocked": profile.unlocked}

    def reset_user(self, req: ResetRequest) -> Dict[str, Any]:
        profile = UserProfile(user_id=req.user_id)
        profile.stats = FaceState(**default_stats())
        self.save_user(profile)
        self.log_event(req.user_id, "reset_user", req.model_dump())
        return {"ok": True, "state": self._state_dict(profile)}

    def get_state(self, user_id: str) -> GAISStateResponse:
        profile = self.load_user(user_id)
        return GAISStateResponse(**self._state_dict(profile))

    def get_events(self, user_id: str, limit: int = 50) -> Dict[str, Any]:
        with db_conn() as conn:
            rows = conn.execute(
                "SELECT event_type, payload, created_at FROM events WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return {
            "user_id": user_id,
            "events": [
                {
                    "event_type": row["event_type"],
                    "payload": json.loads(row["payload"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ],
        }

    def list_missions(self, user_id: str) -> Dict[str, Any]:
        profile = self.load_user(user_id)
        focus = self._focus_from_weak_points(profile.weak_points or profile.stats.model_dump())
        quests = self._quest_catalog(user_id, focus, profile.weak_points)
        completed_keys = self._completed_keys(user_id)
        missions = [q for q in quests if q["quest_key"] not in completed_keys]
        self.log_event(user_id, "list_missions", {"focus": focus})
        return {
            "user_id": user_id,
            "today_completed": len(completed_keys),
            "today_total": len(quests),
            "missions": missions,
            "focus": focus,
        }

    def complete_mission(self, req: MissionCompleteRequest) -> Dict[str, Any]:
        # Backward-compatible alias for the Flutter client.
        return self.complete_quest(QuestCompleteRequest(user_id=req.user_id, quest_id=req.mission_id))

    def get_profile_alias(self, user_id: str) -> Dict[str, Any]:
        profile = self.load_user(user_id)
        state = self._state_dict(profile)
        return {
            "user": {
                "id": profile.user_id,
                "name": "Server User",
                "level": profile.level,
                "xp": profile.xp,
                "stars": profile.stars,
                "mood": profile.mood,
                "rank": profile.rank,
                "stats": profile.stats.model_dump(),
                "unlocked": profile.unlocked,
                "battlepass_level": profile.battlepass_level,
                "currency": profile.currency,
            },
            "state": state,
        }

    def _completed_keys(self, user_id: str) -> List[str]:
        with db_conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM quest_completions WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        keys: List[str] = []
        for row in rows:
            payload = json.loads(row["payload"])
            quest = payload.get("quest", {})
            if isinstance(quest, dict) and quest.get("quest_key"):
                keys.append(str(quest["quest_key"]))
        return keys

    def _store_completion(self, quest_id: str, user_id: str, payload: Dict[str, Any]) -> None:
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO quest_completions(quest_id, user_id, payload, created_at) VALUES(?, ?, ?, ?)",
                (quest_id, user_id, json.dumps(payload, ensure_ascii=False), time.time()),
            )

    def _read_completion(self, quest_id: str) -> Optional[Dict[str, Any]]:
        with db_conn() as conn:
            row = conn.execute("SELECT payload FROM quest_completions WHERE quest_id = ?", (quest_id,)).fetchone()
        if row is None:
            return None
        return json.loads(row["payload"])

    def _reward_text(self, xp: int, stars: int, level: int) -> str:
        if level >= 10:
            return f"레벨 {level} 달성! XP +{xp}, 별파편 +{stars}."
        return f"보상 획득: XP +{xp}, 별파편 +{stars}."


core = GAISCore()

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "app": APP_NAME, "version": APP_VERSION, "uptime_sec": round(time.time() - app_ready_at, 3)}


@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "app": APP_NAME, "version": APP_VERSION}


@app.post("/evaluate")
def evaluate(req: FaceEvaluateRequest) -> Dict[str, Any]:
    return core.evaluate_face(req)


@app.post("/evolve")
def evolve(req: EvolutionRequest) -> Dict[str, Any]:
    return core.evolve_future(req)


@app.post("/quest")
def quest(req: QuestRequest) -> Dict[str, Any]:
    return core.generate_quests(req)


@app.post("/complete")
def complete(req: QuestCompleteRequest) -> Dict[str, Any]:
    return core.complete_quest(req)


@app.post("/battle")
def battle(req: BattleRequest) -> Dict[str, Any]:
    return core.battle(req)


@app.post("/unlock")
def unlock(req: UnlockRequest) -> Dict[str, Any]:
    return core.unlock_item(req)


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    return core.reset_user(req)


@app.get("/state/{user_id}")
def state(user_id: str) -> GAISStateResponse:
    return core.get_state(user_id)


@app.get("/events/{user_id}")
def events(user_id: str, limit: int = 50) -> Dict[str, Any]:
    return core.get_events(user_id, limit)


# -----------------------------------------------------------------------------
# Backward-compatible Ascend endpoints
# -----------------------------------------------------------------------------


@app.get("/me")
def me(user_id: str = "user-1") -> Dict[str, Any]:
    return core.get_profile_alias(user_id)


@app.get("/missions")
def missions(user_id: str = "user-1") -> Dict[str, Any]:
    return core.list_missions(user_id)


@app.post("/missions/complete")
def missions_complete(req: MissionCompleteRequest) -> Dict[str, Any]:
    return core.complete_mission(req)


@app.get("/shop")
def shop(user_id: str = "user-1") -> Dict[str, Any]:
    profile = core.load_user(user_id)
    items = [
        {"id": "hair_style_01", "name": "Soft Wave Hair", "cost": 30, "required_level": 3, "owned": "hair_style_01" in profile.unlocked},
        {"id": "glasses_01", "name": "Clean Glasses", "cost": 45, "required_level": 5, "owned": "glasses_01" in profile.unlocked},
        {"id": "premium_style_01", "name": "Premium Aura Set", "cost": 120, "required_level": 10, "owned": "premium_style_01" in profile.unlocked},
        {"id": "battle_pass_skin_01", "name": "Ascendant Skin", "cost": 200, "required_level": 20, "owned": "battle_pass_skin_01" in profile.unlocked},
    ]
    return {"user_id": user_id, "items": items, "currency": profile.currency}


@app.get("/app/meta")
def app_meta() -> Dict[str, Any]:
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "db_path": str(DB_PATH),
        "vision_enabled": bool(VISION_API_URL and VISION_API_KEY),
    }


# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------


@app.exception_handler(HTTPException)
def http_exc_handler(_: Any, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gais_fastapi_server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
