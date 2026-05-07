"""initial schema (full DDL from db/schema.sql)

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-05-07

"""
from __future__ import annotations

from pathlib import Path

# revision identifiers, used by Alembic.
revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def _load_schema_sql() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "db" / "schema.sql").read_text(encoding="utf-8")


async def upgrade_async(conn) -> None:
    sql = _load_schema_sql()
    # asyncpg는 BEGIN/COMMIT을 본문에서 사용하면 거부한다 (이미 트랜잭션 컨텍스트).
    cleaned = "\n".join(
        line for line in sql.splitlines()
        if line.strip().upper() not in ("BEGIN;", "COMMIT;")
    )
    await conn.execute(cleaned)


async def downgrade_async(conn) -> None:
    # 단일 초기 마이그레이션이므로 downgrade는 의도적으로 비워둠.
    raise NotImplementedError("Initial schema downgrade not supported")
