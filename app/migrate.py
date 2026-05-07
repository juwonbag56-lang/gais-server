"""
Alembic upgrade head, callable from FastAPI startup.

Alembic은 동기 호출이라, 별도 스레드에서 실행한다.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config

logger = logging.getLogger("gais.migrate")


def _alembic_cfg() -> Config:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = Config(str(repo_root / "alembic.ini"))
    # script_location을 절대경로로 강제 (현재 작업 디렉토리에 영향받지 않도록)
    cfg.set_main_option("script_location", str(repo_root / "alembic"))
    return cfg


def _upgrade_sync() -> None:
    cfg = _alembic_cfg()
    command.upgrade(cfg, "head")


async def run_alembic_upgrade() -> None:
    logger.info("Running alembic upgrade head ...")
    await asyncio.to_thread(_upgrade_sync)
