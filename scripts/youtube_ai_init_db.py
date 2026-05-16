import argparse
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = BASE_DIR / "workspaces" / "youtube_ai"


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS oauth_accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL DEFAULT 'google',
    account_email TEXT,
    token_path TEXT,
    scopes TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS channels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    handle TEXT,
    account_id INTEGER,
    status TEXT NOT NULL DEFAULT 'active',
    sync_enabled INTEGER NOT NULL DEFAULT 1,
    timezone TEXT NOT NULL DEFAULT 'Asia/Taipei',
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(account_id) REFERENCES oauth_accounts(id)
);

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    video_id TEXT NOT NULL UNIQUE,
    title TEXT,
    description TEXT,
    published_at TEXT,
    duration_seconds INTEGER,
    thumbnail_url TEXT,
    tags_json TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(channel_id) REFERENCES channels(channel_id)
);

CREATE TABLE IF NOT EXISTS video_daily_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    metric_date TEXT NOT NULL,
    views INTEGER DEFAULT 0,
    impressions INTEGER DEFAULT 0,
    ctr REAL,
    average_view_duration_seconds REAL,
    average_view_percentage REAL,
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    subscribers_gained INTEGER DEFAULT 0,
    subscribers_lost INTEGER DEFAULT 0,
    lifetime_subscribers_gained INTEGER DEFAULT 0,
    lifetime_subscribers_lost INTEGER DEFAULT 0,
    raw_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(video_id, metric_date),
    FOREIGN KEY(video_id) REFERENCES videos(video_id)
);

CREATE TABLE IF NOT EXISTS traffic_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT,
    video_id TEXT NOT NULL,
    metric_date TEXT NOT NULL,
    scope TEXT DEFAULT 'video',
    source_type TEXT NOT NULL,
    views INTEGER DEFAULT 0,
    watch_time_minutes REAL DEFAULT 0,
    average_view_duration_seconds REAL DEFAULT 0,
    raw_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(channel_id) REFERENCES channels(channel_id),
    FOREIGN KEY(video_id) REFERENCES videos(video_id)
);

CREATE TABLE IF NOT EXISTS audience_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    metric_date TEXT NOT NULL,
    segment_type TEXT NOT NULL,
    segment_value TEXT NOT NULL,
    percentage REAL,
    raw_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(video_id)
);

CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    video_id TEXT,
    comment_id TEXT NOT NULL UNIQUE,
    author_name TEXT,
    text TEXT NOT NULL,
    like_count INTEGER DEFAULT 0,
    published_at TEXT,
    sentiment TEXT,
    faq_category TEXT,
    raw_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(channel_id) REFERENCES channels(channel_id),
    FOREIGN KEY(video_id) REFERENCES videos(video_id)
);

CREATE TABLE IF NOT EXISTS video_metadata_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    title TEXT,
    description TEXT,
    tags_json TEXT,
    thumbnail_url TEXT,
    raw_json TEXT,
    FOREIGN KEY(video_id) REFERENCES videos(video_id)
);

CREATE TABLE IF NOT EXISTS ai_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT,
    video_id TEXT,
    insight_type TEXT NOT NULL,
    severity TEXT,
    title TEXT NOT NULL,
    summary TEXT,
    recommendation_json TEXT,
    source_metrics_json TEXT,
    model TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    FOREIGN KEY(channel_id) REFERENCES channels(channel_id),
    FOREIGN KEY(video_id) REFERENCES videos(video_id)
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    insight_id INTEGER,
    channel_id TEXT,
    video_id TEXT,
    task_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'todo',
    due_at TEXT,
    external_target TEXT,
    external_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(insight_id) REFERENCES ai_insights(id)
);

CREATE TABLE IF NOT EXISTS action_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT,
    video_id TEXT,
    task_id INTEGER,
    action_type TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT,
    action_at TEXT NOT NULL,
    actor TEXT NOT NULL DEFAULT 'manual',
    notes TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);

CREATE TABLE IF NOT EXISTS ab_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT,
    video_id TEXT,
    action_log_id INTEGER,
    test_type TEXT NOT NULL,
    baseline_json TEXT,
    result_24h_json TEXT,
    result_48h_json TEXT,
    verdict TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(action_log_id) REFERENCES action_logs(id)
);

CREATE TABLE IF NOT EXISTS daily_briefings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    briefing_date TEXT NOT NULL,
    channel_id TEXT,
    summary TEXT NOT NULL,
    todo_json TEXT,
    delivery_target TEXT,
    delivery_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT,
    conversation_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    decision_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS etl_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT,
    run_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    rows_written INTEGER DEFAULT 0,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_video_daily_metrics_video_date ON video_daily_metrics(video_id, metric_date);
CREATE INDEX IF NOT EXISTS idx_ai_insights_channel_type ON ai_insights(channel_id, insight_type);
CREATE INDEX IF NOT EXISTS idx_tasks_status_due ON tasks(status, due_at);
CREATE INDEX IF NOT EXISTS idx_comments_video ON comments(video_id);
"""


POSTGRES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS oauth_accounts (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT 'google',
    account_email TEXT,
    token_path TEXT,
    scopes TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS channels (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    channel_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    handle TEXT,
    account_id BIGINT REFERENCES oauth_accounts(id),
    status TEXT NOT NULL DEFAULT 'active',
    sync_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    timezone TEXT NOT NULL DEFAULT 'Asia/Taipei',
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    channel_id TEXT NOT NULL REFERENCES channels(channel_id),
    video_id TEXT NOT NULL UNIQUE,
    title TEXT,
    description TEXT,
    published_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    thumbnail_url TEXT,
    tags_json JSONB,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS video_daily_metrics (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    metric_date DATE NOT NULL,
    views INTEGER DEFAULT 0,
    impressions INTEGER DEFAULT 0,
    ctr DOUBLE PRECISION,
    average_view_duration_seconds DOUBLE PRECISION,
    average_view_percentage DOUBLE PRECISION,
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    subscribers_gained INTEGER DEFAULT 0,
    subscribers_lost INTEGER DEFAULT 0,
    lifetime_subscribers_gained INTEGER DEFAULT 0,
    lifetime_subscribers_lost INTEGER DEFAULT 0,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    UNIQUE(video_id, metric_date)
);

CREATE TABLE IF NOT EXISTS traffic_sources (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    channel_id TEXT REFERENCES channels(channel_id),
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    metric_date DATE NOT NULL,
    scope TEXT DEFAULT 'video',
    source_type TEXT NOT NULL,
    views INTEGER DEFAULT 0,
    watch_time_minutes DOUBLE PRECISION DEFAULT 0,
    average_view_duration_seconds DOUBLE PRECISION DEFAULT 0,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS audience_segments (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    metric_date DATE NOT NULL,
    segment_type TEXT NOT NULL,
    segment_value TEXT NOT NULL,
    percentage DOUBLE PRECISION,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS comments (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    channel_id TEXT NOT NULL REFERENCES channels(channel_id),
    video_id TEXT REFERENCES videos(video_id),
    comment_id TEXT NOT NULL UNIQUE,
    author_name TEXT,
    text TEXT NOT NULL,
    like_count INTEGER DEFAULT 0,
    published_at TIMESTAMPTZ,
    sentiment TEXT,
    faq_category TEXT,
    raw_json JSONB,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS video_metadata_snapshots (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    snapshot_at TIMESTAMPTZ NOT NULL,
    title TEXT,
    description TEXT,
    tags_json JSONB,
    thumbnail_url TEXT,
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS ai_insights (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    channel_id TEXT REFERENCES channels(channel_id),
    video_id TEXT REFERENCES videos(video_id),
    insight_type TEXT NOT NULL,
    severity TEXT,
    title TEXT NOT NULL,
    summary TEXT,
    recommendation_json JSONB,
    source_metrics_json JSONB,
    model TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    insight_id BIGINT REFERENCES ai_insights(id),
    channel_id TEXT REFERENCES channels(channel_id),
    video_id TEXT REFERENCES videos(video_id),
    task_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'todo',
    due_at TIMESTAMPTZ,
    external_target TEXT,
    external_id TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS action_logs (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    channel_id TEXT REFERENCES channels(channel_id),
    video_id TEXT REFERENCES videos(video_id),
    task_id BIGINT REFERENCES tasks(id),
    action_type TEXT NOT NULL,
    before_json JSONB,
    after_json JSONB,
    action_at TIMESTAMPTZ NOT NULL,
    actor TEXT NOT NULL DEFAULT 'manual',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS ab_tests (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    channel_id TEXT REFERENCES channels(channel_id),
    video_id TEXT REFERENCES videos(video_id),
    action_log_id BIGINT REFERENCES action_logs(id),
    test_type TEXT NOT NULL,
    baseline_json JSONB,
    result_24h_json JSONB,
    result_48h_json JSONB,
    verdict TEXT,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_briefings (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    briefing_date DATE NOT NULL,
    channel_id TEXT REFERENCES channels(channel_id),
    summary TEXT NOT NULL,
    todo_json JSONB,
    delivery_target TEXT,
    delivery_status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_memory (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    channel_id TEXT REFERENCES channels(channel_id),
    conversation_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    decision_json JSONB,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS etl_runs (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    channel_id TEXT REFERENCES channels(channel_id),
    run_type TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    rows_written INTEGER DEFAULT 0,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_video_daily_metrics_video_date ON video_daily_metrics(video_id, metric_date);
CREATE INDEX IF NOT EXISTS idx_ai_insights_channel_type ON ai_insights(channel_id, insight_type);
CREATE INDEX IF NOT EXISTS idx_tasks_status_due ON tasks(status, due_at);
CREATE INDEX IF NOT EXISTS idx_comments_video ON comments(video_id);
"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_workspace(workspace: Path) -> None:
    for rel in [
        "config",
        "etl_runs",
        "metadata_snapshots",
        "insights",
        "tasks",
        "action_logs",
        "briefings",
        "experiments",
        "memory",
        "exports",
    ]:
        (workspace / rel).mkdir(parents=True, exist_ok=True)


def init_channels_file(workspace: Path) -> Path:
    path = workspace / "config" / "channels.json"
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "channels": [],
                    "updated_at": now_iso(),
                    "notes": "Add YouTube channels from the Streamlit dashboard.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return path


def init_db(workspace: Path) -> Path:
    db_path = workspace / "youtube_ai.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    schema_path = workspace / "youtube_ai_schema.sql"
    schema_path.write_text(SCHEMA_SQL.strip() + "\n", encoding="utf-8")
    postgres_schema_path = workspace / "youtube_ai_supabase_schema.sql"
    postgres_schema_path.write_text(POSTGRES_SCHEMA_SQL.strip() + "\n", encoding="utf-8")
    return db_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize YouTube AI Growth Expert database.")
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", str(DEFAULT_WORKSPACE)))
    args = parser.parse_args()

    workspace = Path(args.workspace_root).resolve()
    init_workspace(workspace)
    channels_path = init_channels_file(workspace)
    db_path = init_db(workspace)
    print(f"workspace={workspace}")
    print(f"database={db_path}")
    print(f"schema={workspace / 'youtube_ai_schema.sql'}")
    print(f"channels={channels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
