"""
Alembic environment.

Alembic은 동기 드라이버를 선호한다. 우리는 .env의 DATABASE_URL을 받아
psycopg2 또는 asyncpg 중 사용 가능한 것을 자동 선택해 sync URL을 만든다.

- psycopg2/psycopg가 없으면 asyncpg를 sync 래핑(SQLAlchemy 2.0의 'asyncpg' 드라이버는
  sync API로도 동작하지 않으므로) 대신 직접 asyncpg로 schema.sql을 실행하는 fallback을 사용한다.
"""
from __future__ import annotations

import os
from logging.config import fileConfig
from urllib.parse import urlsplit, urlunsplit

from alembic import context

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _normalize(url: str, scheme: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((scheme, parts.netloc, parts.path, "", parts.fragment))


def _resolve_db_url() -> str:
    raw = os.getenv("DATABASE_URL", "") or config.get_main_option("sqlalchemy.url", "") or ""
    if not raw:
        raise RuntimeError("DATABASE_URL is not set for migrations")
    return raw


def run_migrations_offline() -> None:
    raise RuntimeError("Offline mode not supported for this project")


def run_migrations_online() -> None:
    """asyncpg를 사용해 raw SQL 마이그레이션을 실행한다.

    Alembic의 표준 흐름은 SQLAlchemy 엔진을 만들어 connection을 넘기는 것이지만,
    우리 마이그레이션 스크립트는 direct asyncpg connection을 받아 실행하도록 작성되어 있다.
    그러므로 여기서는 raw 모드로 동작한다 — alembic의 버전 테이블만 직접 관리한다.
    """
    import asyncio

    import asyncpg

    raw = _resolve_db_url()
    dsn = _normalize(raw, "postgresql")  # asyncpg는 'postgres'/'postgresql' 모두 수용

    async def _ensure_version_table_and_run():
        conn = await asyncpg.connect(dsn=dsn)
        try:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alembic_version (
                    version_num VARCHAR(32) PRIMARY KEY
                );
                """
            )
            current = await conn.fetchval("SELECT version_num FROM alembic_version LIMIT 1")
            from alembic.script import ScriptDirectory

            script = ScriptDirectory.from_config(config)
            head = script.get_current_head()
            if current == head:
                return

            # 모든 head 이전 단계 적용 (linear 단일 chain 가정)
            steps = list(reversed(list(script.walk_revisions(base="base", head="heads"))))
            for rev in steps:
                if current and rev.revision == current:
                    # 이미 적용된 시점을 만나면 다음 리비전부터 적용
                    idx = steps.index(rev)
                    steps = steps[idx + 1:]
                    break

            for rev in steps:
                module = rev.module
                upgrade_fn = getattr(module, "upgrade_async", None)
                if upgrade_fn is None:
                    raise RuntimeError(
                        f"Revision {rev.revision} must define an `async def upgrade_async(conn)`"
                    )
                # 트랜잭션 단위로 실행
                async with conn.transaction():
                    await upgrade_fn(conn)
                    await conn.execute("DELETE FROM alembic_version")
                    await conn.execute(
                        "INSERT INTO alembic_version(version_num) VALUES($1)",
                        rev.revision,
                    )
        finally:
            await conn.close()

    asyncio.run(_ensure_version_table_and_run())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
