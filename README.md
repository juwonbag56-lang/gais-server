# GAIS Server (PostgreSQL Edition)

FastAPI + SQLAlchemy Async + asyncpg + Alembic.

## 환경변수

| 변수 | 필수 | 설명 |
|---|---|---|
| `DATABASE_URL` | ✅ | Render Postgres "Internal Database URL"을 그대로 붙여넣기 |
| `RUN_MIGRATIONS_ON_STARTUP` | ⚪ | 기본 `true`. 부팅 시 `alembic upgrade head` 자동 실행 |
| `DEMO_USER_HANDLE` | ⚪ | Flutter가 보내는 user_id 문자열. 기본 `user-1` |
| `DB_POOL_SIZE` | ⚪ | 기본 `5` |
| `DB_MAX_OVERFLOW` | ⚪ | 기본 `10` |
| `DB_POOL_RECYCLE` | ⚪ | 기본 `1800` (초) |
| `DB_POOL_PRE_PING` | ⚪ | 기본 `true` |

## Render 배포

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/health`
- 첫 부팅 시 Alembic이 `db/schema.sql`을 그대로 적용합니다.

## API (응답 shape 변경 금지)

- `GET /health`
- `GET /app/meta`
- `GET /me`
- `GET /state/{user_id}`
- `GET /events/{user_id}`
- `GET /missions`
- `POST /missions/complete` body: `{ "user_id": "...", "mission_id": "..." }`
- `GET /shop`

## 응답 어댑터 정책

- Flutter는 `user_id="user-1"`을 보냄. 서버는 이를 `users.email='user-1@local'` UUID로 매핑.
- DB의 `user_stats.exp/score`는 응답 시 `xp/stars`로 매핑.
- 새 스키마에 없는 `stats/face_score/rank/unlocked/weak_points/mood/last_quest_id/...`는
  기존 Flutter 호환을 위해 기본값으로 합성 응답.
