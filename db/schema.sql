BEGIN;

-- =========================================================
-- Extensions
-- =========================================================
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS citext;

-- =========================================================
-- Updated-at helper
-- =========================================================
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =========================================================
-- Users
-- =========================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email CITEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name VARCHAR(50) NOT NULL CHECK (char_length(display_name) BETWEEN 2 AND 50),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_display_name ON users (display_name);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users (is_active);

DROP TRIGGER IF EXISTS trg_users_updated_at ON users;
CREATE TRIGGER trg_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- User stats / RPG state
-- =========================================================
CREATE TABLE IF NOT EXISTS user_stats (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    level INTEGER NOT NULL DEFAULT 1 CHECK (level >= 1),
    exp INTEGER NOT NULL DEFAULT 0 CHECK (exp >= 0),
    score INTEGER NOT NULL DEFAULT 0 CHECK (score >= 0),
    streak INTEGER NOT NULL DEFAULT 0 CHECK (streak >= 0),
    coins INTEGER NOT NULL DEFAULT 0 CHECK (coins >= 0),
    rank_points INTEGER NOT NULL DEFAULT 0 CHECK (rank_points >= 0),
    current_title VARCHAR(100),
    last_activity_at TIMESTAMPTZ,
    battlepass_level INTEGER NOT NULL DEFAULT 0 CHECK (battlepass_level >= 0),
    currency INTEGER NOT NULL DEFAULT 0 CHECK (currency >= 0),
    mood VARCHAR(40) NOT NULL DEFAULT 'neutral',
    last_quest_id TEXT,
    last_evaluation_source VARCHAR(40) NOT NULL DEFAULT 'fallback',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_stats_score ON user_stats (score DESC);
CREATE INDEX IF NOT EXISTS idx_user_stats_level ON user_stats (level DESC);
CREATE INDEX IF NOT EXISTS idx_user_stats_rank_points ON user_stats (rank_points DESC);

DROP TRIGGER IF EXISTS trg_user_stats_updated_at ON user_stats;
CREATE TRIGGER trg_user_stats_updated_at
BEFORE UPDATE ON user_stats
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- Daily check-ins / activity history
-- =========================================================
CREATE TABLE IF NOT EXISTS daily_checkins (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    checkin_date DATE NOT NULL,
    exp_gained INTEGER NOT NULL DEFAULT 0 CHECK (exp_gained >= 0),
    score_gained INTEGER NOT NULL DEFAULT 0 CHECK (score_gained >= 0),
    note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, checkin_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_checkins_user_date ON daily_checkins (user_id, checkin_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_checkins_date ON daily_checkins (checkin_date DESC);

-- =========================================================
-- App event logs
-- =========================================================
CREATE TABLE IF NOT EXISTS app_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    event_name VARCHAR(80) NOT NULL,
    event_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_app_events_user_created_at ON app_events (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_app_events_name_created_at ON app_events (event_name, created_at DESC);

-- =========================================================
-- AI analysis runs
-- =========================================================
CREATE TABLE IF NOT EXISTS ai_analysis_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    input_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_name VARCHAR(80),
    status VARCHAR(30) NOT NULL DEFAULT 'queued',
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ai_analysis_runs_user_created_at ON ai_analysis_runs (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_runs_status_created_at ON ai_analysis_runs (status, created_at DESC);

DROP TRIGGER IF EXISTS trg_ai_analysis_runs_updated_at ON ai_analysis_runs;
CREATE TRIGGER trg_ai_analysis_runs_updated_at
BEFORE UPDATE ON ai_analysis_runs
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

-- =========================================================
-- Refresh tokens (store hashed token, not raw token)
-- =========================================================
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_jti CITEXT NOT NULL UNIQUE,
    token_hash TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens (user_id);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires_at ON refresh_tokens (expires_at);

-- =========================================================
-- GAIS / Ascend content tables
-- =========================================================

-- User face uploads / image metadata
CREATE TABLE IF NOT EXISTS user_face_uploads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    s3_key TEXT NOT NULL,
    s3_url TEXT,
    mime_type VARCHAR(80),
    file_size_bytes BIGINT CHECK (file_size_bytes >= 0),
    width INTEGER CHECK (width >= 0),
    height INTEGER CHECK (height >= 0),
    blurhash TEXT,
    sha256 CHAR(64),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_user_face_uploads_user_uploaded_at
    ON user_face_uploads (user_id, uploaded_at DESC);

CREATE INDEX IF NOT EXISTS idx_user_face_uploads_sha256
    ON user_face_uploads (sha256);

CREATE INDEX IF NOT EXISTS idx_user_face_uploads_deleted
    ON user_face_uploads (is_deleted);

-- Face analysis results
CREATE TABLE IF NOT EXISTS face_analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    upload_id UUID REFERENCES user_face_uploads(id) ON DELETE SET NULL,
    score NUMERIC(5,4) NOT NULL CHECK (score >= 0 AND score <= 1),
    confidence NUMERIC(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    uncertainty NUMERIC(5,4) NOT NULL DEFAULT 0 CHECK (uncertainty >= 0 AND uncertainty <= 1),
    percentile NUMERIC(5,2) CHECK (percentile >= 0 AND percentile <= 100),
    elo_rating INTEGER NOT NULL DEFAULT 1500,
    model_version VARCHAR(40),
    explanation_vector JSONB NOT NULL DEFAULT '[]'::jsonb,
    top_positive_features JSONB NOT NULL DEFAULT '[]'::jsonb,
    top_negative_features JSONB NOT NULL DEFAULT '[]'::jsonb,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_face_analysis_results_user_created_at
    ON face_analysis_results (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_face_analysis_results_upload_id
    ON face_analysis_results (upload_id);

CREATE INDEX IF NOT EXISTS idx_face_analysis_results_elo_rating
    ON face_analysis_results (elo_rating DESC);

-- Pairwise labels (manual training data)
CREATE TABLE IF NOT EXISTS pair_labels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    image_a_id UUID NOT NULL REFERENCES user_face_uploads(id) ON DELETE CASCADE,
    image_b_id UUID NOT NULL REFERENCES user_face_uploads(id) ON DELETE CASCADE,
    winner CHAR(1) NOT NULL CHECK (winner IN ('A', 'B', 'T', 'U')),
    confidence NUMERIC(5,4) NOT NULL DEFAULT 1 CHECK (confidence >= 0 AND confidence <= 1),
    quality_score NUMERIC(5,4) NOT NULL DEFAULT 1 CHECK (quality_score >= 0 AND quality_score <= 1),
    difficulty_score NUMERIC(5,4) NOT NULL DEFAULT 0 CHECK (difficulty_score >= 0 AND difficulty_score <= 1),
    label_source VARCHAR(30) NOT NULL DEFAULT 'human',
    demographic_tag VARCHAR(50),
    pose_tag VARCHAR(50),
    lighting_tag VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_pair_not_same_image CHECK (image_a_id <> image_b_id)
);

CREATE INDEX IF NOT EXISTS idx_pair_labels_user_created_at
    ON pair_labels (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_pair_labels_quality_score
    ON pair_labels (quality_score DESC);

CREATE INDEX IF NOT EXISTS idx_pair_labels_difficulty_score
    ON pair_labels (difficulty_score DESC);

CREATE INDEX IF NOT EXISTS idx_pair_labels_winner
    ON pair_labels (winner);

-- Hard negative memory buffer
CREATE TABLE IF NOT EXISTS hard_case_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pair_label_id UUID REFERENCES pair_labels(id) ON DELETE SET NULL,
    image_a_id UUID NOT NULL REFERENCES user_face_uploads(id) ON DELETE CASCADE,
    image_b_id UUID NOT NULL REFERENCES user_face_uploads(id) ON DELETE CASCADE,
    difficulty_score NUMERIC(5,4) NOT NULL CHECK (difficulty_score >= 0 AND difficulty_score <= 1),
    uncertainty NUMERIC(5,4) NOT NULL DEFAULT 0 CHECK (uncertainty >= 0 AND uncertainty <= 1),
    elo_gap NUMERIC(10,4) NOT NULL DEFAULT 0,
    embedding_a JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding_b JSONB NOT NULL DEFAULT '[]'::jsonb,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_hard_case_memory_difficulty
    ON hard_case_memory (difficulty_score DESC);

CREATE INDEX IF NOT EXISTS idx_hard_case_memory_uncertainty
    ON hard_case_memory (uncertainty DESC);

CREATE INDEX IF NOT EXISTS idx_hard_case_memory_created_at
    ON hard_case_memory (created_at DESC);

-- ELO history
CREATE TABLE IF NOT EXISTS elo_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    face_a_id UUID REFERENCES user_face_uploads(id) ON DELETE CASCADE,
    face_b_id UUID REFERENCES user_face_uploads(id) ON DELETE CASCADE,
    rating_a_before INTEGER NOT NULL,
    rating_b_before INTEGER NOT NULL,
    rating_a_after INTEGER NOT NULL,
    rating_b_after INTEGER NOT NULL,
    result CHAR(1) NOT NULL CHECK (result IN ('A', 'B', 'T')),
    k_factor INTEGER NOT NULL DEFAULT 32,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_elo_history_user_created_at
    ON elo_history (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_elo_history_face_a
    ON elo_history (face_a_id);

CREATE INDEX IF NOT EXISTS idx_elo_history_face_b
    ON elo_history (face_b_id);

-- Missions / quests
CREATE TABLE IF NOT EXISTS missions (
    id TEXT PRIMARY KEY,
    title VARCHAR(120) NOT NULL,
    description TEXT,
    focus VARCHAR(30) NOT NULL CHECK (focus IN ('style', 'skin', 'mood', 'overall')),
    xp_reward INTEGER NOT NULL DEFAULT 0 CHECK (xp_reward >= 0),
    shard_reward INTEGER NOT NULL DEFAULT 0 CHECK (shard_reward >= 0),
    difficulty INTEGER NOT NULL DEFAULT 1 CHECK (difficulty >= 1),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_missions_focus_active
    ON missions (focus, active);

DROP TRIGGER IF EXISTS trg_missions_updated_at ON missions;
CREATE TRIGGER trg_missions_updated_at
BEFORE UPDATE ON missions
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

CREATE TABLE IF NOT EXISTS user_missions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mission_id TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'available' CHECK (status IN ('available', 'claimed', 'completed', 'expired')),
    progress INTEGER NOT NULL DEFAULT 0 CHECK (progress >= 0),
    max_progress INTEGER NOT NULL DEFAULT 1 CHECK (max_progress >= 1),
    claimed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, mission_id)
);

CREATE INDEX IF NOT EXISTS idx_user_missions_user_status
    ON user_missions (user_id, status);

CREATE INDEX IF NOT EXISTS idx_user_missions_mission_id
    ON user_missions (mission_id);

DROP TRIGGER IF EXISTS trg_user_missions_updated_at ON user_missions;
CREATE TRIGGER trg_user_missions_updated_at
BEFORE UPDATE ON user_missions
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

CREATE TABLE IF NOT EXISTS mission_completions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mission_id TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    xp_gained INTEGER NOT NULL DEFAULT 0 CHECK (xp_gained >= 0),
    shards_gained INTEGER NOT NULL DEFAULT 0 CHECK (shards_gained >= 0),
    raw_response JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, mission_id)
);

CREATE INDEX IF NOT EXISTS idx_mission_completions_user_created_at
    ON mission_completions (user_id, created_at DESC);

-- Battles / validation loops
CREATE TABLE IF NOT EXISTS battles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    opponent_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    challenger_upload_id UUID REFERENCES user_face_uploads(id) ON DELETE SET NULL,
    opponent_upload_id UUID REFERENCES user_face_uploads(id) ON DELETE SET NULL,
    result VARCHAR(10) NOT NULL CHECK (result IN ('win', 'loss', 'draw')),
    elo_delta INTEGER NOT NULL DEFAULT 0,
    reward_xp INTEGER NOT NULL DEFAULT 0 CHECK (reward_xp >= 0),
    reward_shards INTEGER NOT NULL DEFAULT 0 CHECK (reward_shards >= 0),
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_battles_user_created_at
    ON battles (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_battles_result_created_at
    ON battles (result, created_at DESC);

-- Cosmetics / battle pass
CREATE TABLE IF NOT EXISTS cosmetics (
    id TEXT PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    category VARCHAR(50) NOT NULL,
    required_level INTEGER NOT NULL DEFAULT 1 CHECK (required_level >= 1),
    price INTEGER NOT NULL DEFAULT 0 CHECK (price >= 0),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_cosmetics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    cosmetic_id TEXT NOT NULL REFERENCES cosmetics(id) ON DELETE CASCADE,
    equipped BOOLEAN NOT NULL DEFAULT FALSE,
    acquired_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    equipped_at TIMESTAMPTZ,
    UNIQUE (user_id, cosmetic_id)
);

CREATE INDEX IF NOT EXISTS idx_user_cosmetics_user_equipped
    ON user_cosmetics (user_id, equipped);

-- =========================================================
-- Leaderboard view
-- =========================================================
CREATE OR REPLACE VIEW leaderboard AS
SELECT
    u.id AS user_id,
    u.display_name,
    u.email,
    s.level,
    s.exp,
    s.score,
    s.streak,
    s.coins,
    s.rank_points,
    s.current_title,
    s.last_activity_at,
    s.battlepass_level,
    s.currency,
    s.mood,
    u.created_at
FROM users u
JOIN user_stats s ON s.user_id = u.id
WHERE u.is_active = TRUE
ORDER BY s.rank_points DESC, s.score DESC, s.level DESC, u.created_at ASC;

-- =========================================================
-- Helpful seed rows (optional)
-- =========================================================
INSERT INTO missions (id, title, description, focus, xp_reward, shard_reward, difficulty, active)
VALUES
    ('walk_15', '15분 걷기', '가볍게 15분 산책한다.', 'mood', 20, 3, 1, TRUE),
    ('posture_reset', '자세 교정', '자세를 5분간 교정한다.', 'mood', 25, 4, 1, TRUE),
    ('journal', '짧은 기록', '오늘의 상태를 3줄로 기록한다.', 'mood', 15, 2, 1, TRUE),
    ('skin_cleanse', '세안 루틴', '세안 루틴을 2회 실행한다.', 'skin', 25, 5, 2, TRUE),
    ('fit_check', '옷핏 점검', '오늘의 옷핏을 점검한다.', 'style', 25, 4, 2, TRUE)
ON CONFLICT (id) DO NOTHING;

INSERT INTO cosmetics (id, name, category, required_level, price, active)
VALUES
    ('hair_style_01', 'Soft Wave Hair', 'hair', 3, 30, TRUE),
    ('glasses_01', 'Clean Glasses', 'accessory', 5, 45, TRUE),
    ('premium_style_01', 'Premium Aura Set', 'outfit', 10, 120, TRUE),
    ('battle_pass_skin_01', 'Ascendant Skin', 'skin', 20, 200, TRUE)
ON CONFLICT (id) DO NOTHING;

COMMIT;
