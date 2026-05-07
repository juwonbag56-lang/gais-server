"""
Runtime configuration loaded from environment / .env.
"""
from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

load_dotenv()  # .env 파일이 있으면 로드


def _normalize_async_dsn(raw: str) -> str:
    """Render Postgres URL(`postgres://...` 또는 `postgresql://...`)을
    SQLAlchemy async용 `postgresql+asyncpg://...`로 정규화한다.
    asyncpg가 모르는 sslmode 같은 query 파라미터는 안전하게 제거한다.
    """
    if not raw:
        return raw
    parts = urlsplit(raw)
    scheme = parts.scheme
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+asyncpg"
    elif scheme.startswith("postgresql+"):
        # 이미 드라이버가 지정된 경우는 그대로 사용
        pass
    # asyncpg는 libpq 스타일 query 파라미터(sslmode 등)를 받지 않음 → 제거
    query = ""
    return urlunsplit((scheme, parts.netloc, parts.path, query, parts.fragment))


def _normalize_sync_dsn(raw: str) -> str:
    """Alembic용 동기 드라이버(psycopg2 미설치 환경에서도 동작하는 sync mode 회피).
    여기서는 alembic도 async 엔진으로 실행하므로 굳이 사용하지 않지만, 호환을 위해 보관.
    """
    if not raw:
        return raw
    parts = urlsplit(raw)
    scheme = parts.scheme
    if scheme in ("postgres", "postgresql+asyncpg"):
        scheme = "postgresql"
    return urlunsplit((scheme, parts.netloc, parts.path, "", parts.fragment))


class Settings:
    def __init__(self) -> None:
        self.database_url_raw: str = os.getenv("DATABASE_URL", "")
        self.database_url_async: str = _normalize_async_dsn(self.database_url_raw)
        self.database_url_sync: str = _normalize_sync_dsn(self.database_url_raw)
        self.run_migrations_on_startup: bool = (
            os.getenv("RUN_MIGRATIONS_ON_STARTUP", "true").lower() == "true"
        )
        self.demo_user_handle: str = os.getenv("DEMO_USER_HANDLE", "user-1")
        self.demo_user_email: str = f"{self.demo_user_handle}@local"
        self.demo_user_password_hash: str = "!"  # 데모 유저: 외부 로그인 불가용 placeholder
        self.demo_user_display_name: str = "Server User"

        self.db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
        self.db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
        self.db_pool_recycle: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))
        self.db_pool_pre_ping: bool = (
            os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"
        )

        self.app_version: str = os.getenv("APP_VERSION", "3.0.0")
        self.vision_api_url: str = os.getenv("VISION_API_URL", "")
        self.vision_api_key: str = os.getenv("VISION_API_KEY", "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
