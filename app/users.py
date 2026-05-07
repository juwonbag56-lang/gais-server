"""
Demo-user adapter.

Flutter 앱은 임의의 문자열 user_id(예: "user-1")를 보냅니다.
스키마는 users.id가 UUID이고 email/password_hash가 NOT NULL이므로
이 어댑터에서 handle("user-1") ↔ UUID를 매핑합니다.

규칙:
- email = f"{handle}@local"  (CITEXT, 유니크)
- display_name = handle (단, 2~50자 제약 → "u_<handle>"로 패딩 보정)
- password_hash = "!"  (placeholder; 로그인 비활성)
- user_stats 행도 함께 생성
"""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings


def _safe_display_name(handle: str) -> str:
    base = handle.strip() or "user"
    if len(base) < 2:
        base = f"u_{base}"
    return base[:50]


async def get_or_create_user_id(session: AsyncSession, handle: str) -> uuid.UUID:
    """handle("user-1") → users.id(UUID). 없으면 자동 생성 + user_stats 시드."""
    settings = get_settings()
    email = f"{handle}@local"
    display = _safe_display_name(handle)
    pwd = settings.demo_user_password_hash

    # 1) 이미 있으면 반환
    row = (
        await session.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": email},
        )
    ).first()
    if row:
        return row[0]

    # 2) 없으면 생성 (race 조건 대비 ON CONFLICT)
    inserted = (
        await session.execute(
            text(
                """
                INSERT INTO users (email, password_hash, display_name)
                VALUES (:email, :pwd, :display)
                ON CONFLICT (email) DO UPDATE SET email = EXCLUDED.email
                RETURNING id
                """
            ),
            {"email": email, "pwd": pwd, "display": display},
        )
    ).first()
    user_id: uuid.UUID = inserted[0]

    # user_stats 시드 (없으면)
    await session.execute(
        text(
            """
            INSERT INTO user_stats (user_id)
            VALUES (:user_id)
            ON CONFLICT (user_id) DO NOTHING
            """
        ),
        {"user_id": user_id},
    )
    return user_id


async def get_user_handle_by_id(session: AsyncSession, user_id: uuid.UUID) -> Optional[str]:
    row = (
        await session.execute(
            text("SELECT email FROM users WHERE id = :id"),
            {"id": user_id},
        )
    ).first()
    if not row:
        return None
    email = str(row[0])
    if email.endswith("@local"):
        return email[: -len("@local")]
    return email
