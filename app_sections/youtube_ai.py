import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

from app_utils.filesystem import log_file_for_stage

ROOT: Path | None = None
WS_ROOT: str = "workspaces/youtube_ai"
runner = None


def configure_youtube_ai(root: Path, workspace_root: str, stage_runner) -> None:
    global ROOT, WS_ROOT, runner
    ROOT = root
    WS_ROOT = workspace_root
    runner = stage_runner


def _root() -> Path:
    if ROOT is None:
        raise RuntimeError("YouTube AI section is not configured. Call configure_youtube_ai first.")
    return ROOT


def read_json_file(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def youtube_ai_workspace_path() -> Path:
    return _root() / WS_ROOT

def youtube_ai_db_path() -> Path:
    env_path = os.environ.get("YOUTUBE_AI_SQLITE_PATH", "workspaces/youtube_ai/youtube_ai.db")
    path = Path(env_path)
    return path if path.is_absolute() else _root() / path

def youtube_ai_channels_path() -> Path:
    return youtube_ai_workspace_path() / "config" / "channels.json"

def youtube_ai_settings_path() -> Path:
    return youtube_ai_workspace_path() / "config" / "settings.json"

def default_youtube_ai_settings() -> dict:
    return {
        "etl_days": 32,
        "comments_per_video": 10,
        "etl_log_auto_refresh": True,
        "updated_at": "",
    }

def read_youtube_ai_settings() -> dict:
    settings = default_youtube_ai_settings()
    payload = read_json_file(youtube_ai_settings_path()) or {}
    if isinstance(payload, dict):
        settings.update({k: v for k, v in payload.items() if k in settings})
    return settings

def write_youtube_ai_settings(settings: dict) -> Path:
    path = youtube_ai_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = default_youtube_ai_settings()
    payload.update(settings)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def read_youtube_ai_channels() -> list[dict]:
    payload = read_json_file(youtube_ai_channels_path()) or {}
    channels = payload.get("channels") if isinstance(payload, dict) else []
    return channels if isinstance(channels, list) else []

def write_youtube_ai_channels(channels: list[dict]) -> Path:
    path = youtube_ai_channels_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"channels": channels, "updated_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path

def sync_youtube_ai_channels_to_db(channels: list[dict]) -> None:
    import sqlite3

    db_path = youtube_ai_db_path()
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    try:
        now = datetime.now().isoformat(timespec="seconds")
        for item in channels:
            channel_id = str(item.get("channel_id", "")).strip()
            if not channel_id:
                continue
            conn.execute(
                """
                INSERT INTO channels(channel_id, title, handle, status, sync_enabled, timezone, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    title=excluded.title,
                    handle=excluded.handle,
                    status=excluded.status,
                    sync_enabled=excluded.sync_enabled,
                    timezone=excluded.timezone,
                    notes=excluded.notes,
                    updated_at=excluded.updated_at
                """,
                (
                    channel_id,
                    str(item.get("title", "")).strip() or channel_id,
                    str(item.get("handle", "")).strip(),
                    str(item.get("status", "active")).strip() or "active",
                    1 if item.get("sync_enabled", True) else 0,
                    str(item.get("timezone", "Asia/Taipei")).strip() or "Asia/Taipei",
                    str(item.get("notes", "")).strip(),
                    now,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()

def youtube_ai_db_table_counts(db_path: Path) -> pd.DataFrame:
    import sqlite3

    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
            conn,
        )["name"].tolist()
        rows = []
        for table in tables:
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            rows.append({"資料表": table, "筆數": count})
        return pd.DataFrame(rows)
    finally:
        conn.close()

def youtube_ai_recent_etl_runs(db_path: Path, limit: int = 5) -> pd.DataFrame:
    import sqlite3

    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(
            """
            SELECT id, run_type, started_at, ended_at, status, rows_written, error
            FROM etl_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        ).fillna("")
    finally:
        conn.close()

def youtube_ai_select_channel(channels: list[dict], key: str = "youtube_ai_selected_channel_id") -> tuple[str, list[dict]]:
    options = [("全部頻道", "")]
    for channel in channels:
        channel_id = str(channel.get("channel_id", "")).strip()
        if not channel_id:
            continue
        title = str(channel.get("title", "")).strip() or channel_id
        handle = str(channel.get("handle", "")).strip()
        label = f"{title} ({handle})" if handle else title
        options.append((label, channel_id))

    current = st.session_state.get(key, "")
    ids = [item[1] for item in options]
    index = ids.index(current) if current in ids else 0
    selected_label = st.selectbox("Channel", [item[0] for item in options], index=index, key=f"{key}_label")
    selected_id = dict(options).get(selected_label, "")
    st.session_state[key] = selected_id
    if selected_id:
        return selected_id, [channel for channel in channels if str(channel.get("channel_id", "")).strip() == selected_id]
    return "", channels

def youtube_ai_channel_metrics_summary(db_path: Path, channel_id: str = "") -> pd.DataFrame:
    import sqlite3

    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        params = []
        where = ""
        if channel_id:
            where = "WHERE v.channel_id = ?"
            params.append(channel_id)
        query = f"""
            SELECT
                COUNT(DISTINCT v.video_id) AS videos,
                COALESCE(SUM(latest.views), 0) AS latest_views,
                COALESCE(SUM(latest.likes), 0) AS latest_likes,
                COALESCE(SUM(latest.comments), 0) AS latest_comments,
                MAX(latest.metric_date) AS latest_metric_date
            FROM videos v
            LEFT JOIN (
                SELECT m.*
                FROM video_daily_metrics m
                JOIN (
                    SELECT video_id, MAX(metric_date) AS metric_date
                    FROM video_daily_metrics
                    GROUP BY video_id
                ) lm
                  ON lm.video_id = m.video_id
                 AND lm.metric_date = m.metric_date
            ) latest
              ON latest.video_id = v.video_id
            {where}
        """
        return pd.read_sql_query(query, conn, params=params).fillna("")
    finally:
        conn.close()

def youtube_ai_video_performance(db_path: Path, channel_id: str = "") -> pd.DataFrame:
    import sqlite3

    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        params = []
        where = ""
        if channel_id:
            where = "WHERE v.channel_id = ?"
            params.append(channel_id)
        query = f"""
            SELECT
                c.title AS channel_title,
                v.channel_id,
                v.video_id,
                v.title,
                v.published_at,
                v.thumbnail_url,
                m.metric_date,
                COALESCE(m.views, 0) AS views,
                COALESCE(m.impressions, 0) AS impressions,
                COALESCE(m.ctr, 0) AS ctr,
                COALESCE(m.likes, 0) AS likes,
                COALESCE(m.comments, 0) AS comments,
                COALESCE(m.shares, 0) AS shares,
                COALESCE(m.average_view_duration_seconds, 0) AS average_view_duration_seconds,
                COALESCE(m.average_view_percentage, 0) AS average_view_percentage,
                COALESCE(m.subscribers_gained, 0) AS subscribers_gained,
                COALESCE(m.subscribers_lost, 0) AS subscribers_lost,
                COALESCE(m.lifetime_subscribers_gained, m.subscribers_gained, 0) AS lifetime_subscribers_gained,
                COALESCE(m.lifetime_subscribers_lost, m.subscribers_lost, 0) AS lifetime_subscribers_lost,
                COALESCE(m.raw_json, '') AS raw_json
            FROM videos v
            LEFT JOIN channels c ON c.channel_id = v.channel_id
            LEFT JOIN video_daily_metrics m ON m.video_id = v.video_id
            {where}
            ORDER BY v.channel_id, v.video_id, m.metric_date
        """
        df = pd.read_sql_query(query, conn, params=params).fillna("")
    finally:
        conn.close()

    if df.empty:
        return df
    df["metric_date"] = df["metric_date"].astype(str)
    for col in [
        "views",
        "impressions",
        "ctr",
        "likes",
        "comments",
        "shares",
        "average_view_duration_seconds",
        "average_view_percentage",
        "subscribers_gained",
        "subscribers_lost",
        "lifetime_subscribers_gained",
        "lifetime_subscribers_lost",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = df[df["metric_date"].str.len() > 0].copy()
    if df.empty:
        return df
    latest = df.sort_values(["video_id", "metric_date"]).groupby("video_id", as_index=False).tail(1).copy()
    latest["interactions"] = latest["likes"] + latest["comments"] + latest["shares"]
    latest["影片連結"] = latest["video_id"].astype(str).apply(lambda video_id: f"https://www.youtube.com/watch?v={video_id}" if video_id else "")
    def estimated_minutes(row: pd.Series) -> float:
        try:
            raw = json.loads(str(row.get("raw_json") or "{}"))
            value = raw.get("estimatedMinutesWatched")
            if value not in (None, ""):
                return float(value)
        except Exception:
            pass
        return float(row.get("views", 0) or 0) * float(row.get("average_view_duration_seconds", 0) or 0) / 60.0
    latest["estimated_minutes_watched"] = latest.apply(estimated_minutes, axis=1)
    latest["watch_hours"] = latest["estimated_minutes_watched"] / 60.0
    latest["views_from_impressions"] = latest.apply(
        lambda row: int(round(float(row["impressions"]) * float(row["ctr"]) / 100.0)) if float(row["impressions"]) > 0 else 0,
        axis=1,
    )
    latest["watch_hours_from_impressions"] = latest.apply(
        lambda row: float(row["views_from_impressions"]) * float(row["average_view_duration_seconds"]) / 3600.0,
        axis=1,
    )
    latest["engagement_rate"] = latest.apply(
        lambda row: round((float(row["interactions"]) / float(row["views"]) * 100), 2) if float(row["views"]) > 0 else 0,
        axis=1,
    )
    latest["net_subscribers"] = latest["subscribers_gained"] - latest["subscribers_lost"]
    latest["lifetime_net_subscribers"] = latest["lifetime_subscribers_gained"] - latest["lifetime_subscribers_lost"]
    latest = latest.sort_values(["views", "interactions"], ascending=[False, False])
    return latest.reset_index(drop=True)

YOUTUBE_TRAFFIC_SOURCE_LABELS = {
    "ADVERTISING": "廣告",
    "ANNOTATION": "註解",
    "CAMPAIGN_CARD": "資訊卡",
    "END_SCREEN": "片尾畫面",
    "EXT_URL": "外部連結",
    "HASHTAGS": "Hashtags",
    "NO_LINK_OTHER": "其他 YouTube 功能",
    "NO_LINK_EMBEDDED": "嵌入播放器",
    "NO_LINK_YT_OTHER": "其他 YouTube 來源",
    "NOTIFICATION": "通知",
    "PLAYLIST": "播放清單",
    "PROMOTED": "付費推廣",
    "RELATED_VIDEO": "推薦影片",
    "SHORTS": "Shorts",
    "SOUND_PAGE": "音效頁",
    "SUBSCRIBER": "訂閱內容",
    "YT_CHANNEL": "頻道頁",
    "YT_SEARCH": "YouTube 搜尋",
    "YT_OTHER_PAGE": "其他 YouTube 頁面",
}

def youtube_ai_traffic_sources(db_path: Path, channel_id: str = "", scope: str = "channel", video_id: str = "") -> pd.DataFrame:
    import sqlite3

    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        params = [scope]
        where = "WHERE scope = ?"
        if channel_id:
            where += " AND channel_id = ?"
            params.append(channel_id)
        if video_id:
            where += " AND video_id = ?"
            params.append(video_id)
        query = f"""
            SELECT
                source_type,
                SUM(views) AS views,
                SUM(watch_time_minutes) AS watch_time_minutes,
                AVG(average_view_duration_seconds) AS average_view_duration_seconds,
                MAX(metric_date) AS metric_date
            FROM traffic_sources
            {where}
            GROUP BY source_type
            ORDER BY views DESC
        """
        df = pd.read_sql_query(query, conn, params=params).fillna("")
    finally:
        conn.close()
    if df.empty:
        return df
    for col in ["views", "watch_time_minutes", "average_view_duration_seconds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    total_views = float(df["views"].sum())
    total_watch = float(df["watch_time_minutes"].sum())
    df["source_label"] = df["source_type"].map(YOUTUBE_TRAFFIC_SOURCE_LABELS).fillna(df["source_type"])
    df["views_share"] = df["views"].apply(lambda value: round(float(value) / total_views * 100, 2) if total_views else 0)
    df["watch_time_hours"] = df["watch_time_minutes"] / 60.0
    df["watch_time_share"] = df["watch_time_minutes"].apply(lambda value: round(float(value) / total_watch * 100, 2) if total_watch else 0)
    return df

def youtube_ai_video_top_traffic_sources(db_path: Path, channel_id: str = "") -> pd.DataFrame:
    import sqlite3

    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        params = []
        where = "WHERE scope = 'video'"
        if channel_id:
            where += " AND channel_id = ?"
            params.append(channel_id)
        query = f"""
            SELECT video_id, source_type, SUM(views) AS views
            FROM traffic_sources
            {where}
            GROUP BY video_id, source_type
            ORDER BY video_id, views DESC
        """
        df = pd.read_sql_query(query, conn, params=params).fillna("")
    finally:
        conn.close()
    if df.empty:
        return df
    df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0)
    top = df.sort_values(["video_id", "views"], ascending=[True, False]).groupby("video_id", as_index=False).head(1).copy()
    top["主要流量來源"] = top["source_type"].map(YOUTUBE_TRAFFIC_SOURCE_LABELS).fillna(top["source_type"])
    top["主要來源 Views"] = top["views"].astype(int)
    return top[["video_id", "主要流量來源", "主要來源 Views"]]

def render_youtube_ai_reports(db_path: Path, channel_id: str, visible_channels: list[dict]) -> None:
    st.markdown("**頻道健康總覽**")
    performance = youtube_ai_video_performance(db_path, channel_id)
    if performance.empty:
        st.info("尚未有可用的影片指標。請先在 Pipeline Manager 執行 Daily ETL。")
        return

    total_subscribers = sum(int(channel.get("subscriber_count", 0) or 0) for channel in visible_channels)
    total_views = int(performance["views"].sum())
    total_watch_hours = float(performance["watch_hours"].sum())
    total_interactions = int(performance["interactions"].sum())
    avg_duration = float(performance["average_view_duration_seconds"].mean())
    avg_percentage = float(performance["average_view_percentage"].mean())
    health_rows = [
        {"指標": "目前訂閱數", "數值": f"{total_subscribers:,}", "說明": "來自 Channel 同步資料"},
        {"指標": "累積觀看時數", "數值": f"{total_watch_hours:,.1f}", "說明": "本次 ETL 區間內每支影片最新指標彙總"},
        {"指標": "觀看數", "數值": f"{total_views:,}", "說明": "本次 ETL 區間彙總"},
        {"指標": "互動數", "數值": f"{total_interactions:,}", "說明": "按讚 + 留言 + 分享"},
        {"指標": "平均觀看秒數", "數值": f"{avg_duration:.1f}", "說明": "影片平均"},
        {"指標": "平均觀看比例", "數值": f"{avg_percentage:.1f}%", "說明": "影片平均"},
    ]
    st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True)

    st.caption("健康總覽使用每支影片最新一筆 ETL 指標彙總，適合快速判斷目前頻道內容池的整體狀態。")

    st.markdown("**流量來源輪廓**")
    traffic_df = youtube_ai_traffic_sources(db_path, channel_id, scope="channel")
    if traffic_df.empty:
        st.info("尚未有流量來源資料。請重新執行 Daily ETL 以同步 insightTrafficSourceType。")
    else:
        traffic_table = traffic_df[
            ["source_label", "views", "views_share", "watch_time_hours", "watch_time_share", "average_view_duration_seconds"]
        ].rename(
            columns={
                "source_label": "流量來源",
                "views": "Views",
                "views_share": "Views 佔比%",
                "watch_time_hours": "Watch time hours",
                "watch_time_share": "Watch time 佔比%",
                "average_view_duration_seconds": "Average view duration",
            }
        )
        traffic_table["Watch time hours"] = pd.to_numeric(traffic_table["Watch time hours"], errors="coerce").fillna(0).round(1)
        traffic_table["Average view duration"] = pd.to_numeric(traffic_table["Average view duration"], errors="coerce").fillna(0).round(1)
        st.dataframe(traffic_table, use_container_width=True, hide_index=True)
        chart_df = traffic_table.set_index("流量來源")[["Views"]]
        st.bar_chart(chart_df)

    st.markdown("**Impressions 漏斗**")
    total_impressions = int(performance["impressions"].sum())
    total_views_from_impressions = int(performance["views_from_impressions"].sum())
    total_watch_hours_from_impressions = float(performance["watch_hours_from_impressions"].sum())
    overall_ctr = (total_views_from_impressions / total_impressions * 100.0) if total_impressions else 0.0
    funnel_rows = [
        {"階段": "Impressions", "數值": f"{total_impressions:,}" if total_impressions else "尚未由 API 提供", "轉換率": "100%" if total_impressions else "—"},
        {"階段": "Click-through rate", "數值": f"{overall_ctr:.2f}%" if total_impressions else "尚未由 API 提供", "轉換率": f"{overall_ctr:.2f}%" if total_impressions else "—"},
        {"階段": "Views from impressions", "數值": f"{total_views_from_impressions:,}" if total_impressions else "尚未由 API 提供", "轉換率": f"{overall_ctr:.2f}%" if total_impressions else "—"},
        {"階段": "Watch time from impressions (hours)", "數值": f"{total_watch_hours_from_impressions:,.1f}" if total_impressions else "尚未由 API 提供", "轉換率": "—"},
    ]
    st.dataframe(pd.DataFrame(funnel_rows), use_container_width=True, hide_index=True)
    if total_impressions:
        st.progress(min(max(overall_ctr / 100.0, 0.0), 1.0), text=f"Impressions → Views CTR {overall_ctr:.2f}%")
    else:
        st.caption("YouTube Analytics API 目前未回傳 YouTube Studio 的 Impressions / CTR 漏斗指標；欄位保留，待後續資料源支援後會自動呈現。")

    st.markdown("**影片表現排行**")
    with st.expander("欄位說明", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {"欄位": "Channel", "說明": "影片所屬的 YouTube 頻道。"},
                    {"欄位": "影片標題", "說明": "目前同步到資料庫的影片標題。"},
                    {"欄位": "影片連結", "說明": "可直接開啟 YouTube 影片頁面的連結。"},
                    {"欄位": "上架日期", "說明": "影片發布到 YouTube 的時間。"},
                    {"欄位": "指標日期", "說明": "這筆排行資料使用的 ETL 指標日期。"},
                    {"欄位": "觀看數", "說明": "該影片在目前同步指標中的觀看數。"},
                    {"欄位": "Impressions", "說明": "YouTube 顯示影片縮圖的次數；目前官方 Analytics API 未回傳時會顯示 0。"},
                    {"欄位": "Click-through rate", "說明": "從 Impressions 點進觀看的比例。"},
                    {"欄位": "Views from impressions", "說明": "由 Impressions 估算或資料源回傳的觀看數。"},
                    {"欄位": "Watch time from impressions (hours)", "說明": "由 Impressions 帶來的觀看時數。"},
                    {"欄位": "按讚 / 留言 / 分享", "說明": "該影片的互動資料。"},
                    {"欄位": "互動率%", "說明": "(按讚 + 留言 + 分享) / 觀看數 * 100。"},
                    {"欄位": "平均觀看秒數", "說明": "觀眾平均觀看此影片的秒數。"},
                    {"欄位": "平均觀看比例%", "說明": "觀眾平均看完影片長度的百分比。"},
                    {"欄位": "累積訂閱增加", "說明": "此影片自上架以來帶來的新訂閱數。"},
                    {"欄位": "累積訂閱流失", "說明": "此影片自上架以來造成的取消訂閱數。"},
                    {"欄位": "累積淨訂閱", "說明": "累積訂閱增加 - 累積訂閱流失。"},
                    {"欄位": "觀看時數", "說明": "此影片在 ETL 區間內累積的觀看時數。"},
                    {"欄位": "主要流量來源", "說明": "該影片觀看數最高的 insightTrafficSourceType。"},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    rank_cols = st.columns([1, 1, 1, 1])
    rank_mode = rank_cols[0].selectbox("排行指標", ["觀看數", "累積訂閱增加", "累積淨訂閱", "互動率", "平均觀看比例", "平均觀看秒數"], key="youtube_ai_rank_metric")
    rank_direction = rank_cols[1].selectbox("排行方向", ["Top", "Bottom"], key="youtube_ai_rank_direction")
    limit = int(rank_cols[2].slider("顯示影片數", min_value=5, max_value=50, value=10, step=5, key="youtube_ai_rank_limit"))
    freeze_view = rank_cols[3].toggle("凍結識別欄", value=True, key="youtube_ai_rank_freeze_view")

    metric_map = {
        "觀看數": "views",
        "累積訂閱增加": "lifetime_subscribers_gained",
        "累積淨訂閱": "lifetime_net_subscribers",
        "互動率": "engagement_rate",
        "平均觀看比例": "average_view_percentage",
        "平均觀看秒數": "average_view_duration_seconds",
    }
    sort_col = metric_map[rank_mode]
    ranked = performance.sort_values(sort_col, ascending=(rank_direction == "Bottom")).head(limit).copy()
    top_sources = youtube_ai_video_top_traffic_sources(db_path, channel_id)
    if not top_sources.empty:
        ranked = ranked.merge(top_sources, on="video_id", how="left")
    else:
        ranked["主要流量來源"] = ""
        ranked["主要來源 Views"] = 0
    table = ranked[
        [
            "channel_title",
            "title",
            "影片連結",
            "published_at",
            "metric_date",
            "views",
            "impressions",
            "ctr",
            "views_from_impressions",
            "watch_hours_from_impressions",
            "likes",
            "comments",
            "shares",
            "engagement_rate",
            "average_view_duration_seconds",
            "average_view_percentage",
            "lifetime_subscribers_gained",
            "lifetime_subscribers_lost",
            "lifetime_net_subscribers",
            "watch_hours",
            "主要流量來源",
            "主要來源 Views",
        ]
    ].rename(
        columns={
            "channel_title": "Channel",
            "title": "影片標題",
            "published_at": "上架日期",
            "metric_date": "指標日期",
            "views": "觀看數",
            "impressions": "Impressions",
            "ctr": "Click-through rate",
            "views_from_impressions": "Views from impressions",
            "watch_hours_from_impressions": "Watch time from impressions (hours)",
            "likes": "按讚",
            "comments": "留言",
            "shares": "分享",
            "engagement_rate": "互動率%",
            "average_view_duration_seconds": "平均觀看秒數",
            "average_view_percentage": "平均觀看比例%",
            "lifetime_subscribers_gained": "累積訂閱增加",
            "lifetime_subscribers_lost": "累積訂閱流失",
            "lifetime_net_subscribers": "累積淨訂閱",
            "watch_hours": "觀看時數",
        }
    )
    table["上架日期"] = pd.to_datetime(table["上架日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    table["上架日期"] = table["上架日期"].fillna("")
    table["觀看時數"] = pd.to_numeric(table["觀看時數"], errors="coerce").fillna(0).round(1)
    table["Click-through rate"] = pd.to_numeric(table["Click-through rate"], errors="coerce").fillna(0).round(2)
    table["Watch time from impressions (hours)"] = pd.to_numeric(table["Watch time from impressions (hours)"], errors="coerce").fillna(0).round(1)
    link_config = {"影片連結": st.column_config.LinkColumn("影片連結", display_text="開啟影片")}
    if freeze_view:
        st.caption("凍結識別欄視圖：左側保留 Channel / 影片 / 連結 / 上架日期，右側顯示可橫向瀏覽的數據欄位。")
        left_cols = ["Channel", "影片標題", "影片連結", "上架日期"]
        metric_cols = [col for col in table.columns if col not in left_cols]
        left_panel, right_panel = st.columns([1.35, 2.65])
        with left_panel:
            st.dataframe(table[left_cols], use_container_width=True, hide_index=True, height=420, column_config=link_config)
        with right_panel:
            st.dataframe(table[metric_cols], use_container_width=True, hide_index=True, height=420)
    else:
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config=link_config,
        )

    chart_df = ranked[["title", sort_col]].copy()
    chart_df["title"] = chart_df["title"].astype(str).str.slice(0, 40)
    chart_df = chart_df.rename(columns={"title": "影片", sort_col: rank_mode}).set_index("影片")
    st.bar_chart(chart_df)

def render_youtube_ai_dashboard() -> None:
    st.subheader("YouTube AI 增長專家系統")
    st.caption("頻道健康狀態、最近同步結果與後續 AI 診斷摘要。")

    workspace = youtube_ai_workspace_path()
    db_path = youtube_ai_db_path()
    channels = read_youtube_ai_channels()
    selected_channel_id, visible_channels = youtube_ai_select_channel(channels)
    recent_runs = youtube_ai_recent_etl_runs(db_path)
    summary_df = youtube_ai_channel_metrics_summary(db_path, selected_channel_id)
    summary = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    latest_status = "尚無紀錄"
    latest_ended = "—"
    if not recent_runs.empty:
        latest_status = str(recent_runs.iloc[0].get("status") or "—")
        latest_ended = str(recent_runs.iloc[0].get("ended_at") or "—")

    status_rows = [
        {"項目": "已同步 Channel", "值": len(visible_channels)},
        {"項目": "影片數", "值": int(summary.get("videos") or 0)},
        {"項目": "最近 ETL", "值": latest_status},
        {"項目": "完成時間", "值": latest_ended},
        {"項目": "指標日期", "值": str(summary.get("latest_metric_date") or "—")},
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    if visible_channels:
        st.markdown("**已同步 Channel**")
        df = pd.DataFrame(visible_channels)
        display_cols = [c for c in ["title", "handle", "channel_id", "subscriber_count", "video_count", "view_count", "updated_at"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("尚未同步 Channel。請到 Settings 進行 OAuth 授權與 Channel 同步。")

    if not recent_runs.empty:
        st.markdown("**最近 ETL 執行紀錄**")
        st.dataframe(recent_runs, use_container_width=True, hide_index=True)
    else:
        st.caption("尚無 ETL 執行紀錄。")

    if db_path.exists():
        with st.expander("資料庫摘要", expanded=False):
            st.dataframe(youtube_ai_db_table_counts(db_path), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**報表開發區**")
    render_youtube_ai_reports(db_path, selected_channel_id, visible_channels)

def render_youtube_ai_settings() -> None:
    st.subheader("YouTube AI Settings")
    st.caption("資料庫、OAuth、Channel 同步與 Daily ETL 預設參數。")

    workspace = youtube_ai_workspace_path()
    db_path = youtube_ai_db_path()
    channels_path = youtube_ai_channels_path()
    settings = read_youtube_ai_settings()

    cols = st.columns(4)
    cols[0].metric("DB Provider", os.environ.get("YOUTUBE_AI_DB_PROVIDER", "sqlite"))
    cols[1].metric("Project", os.environ.get("YOUTUBE_AI_PROJECT_NAME", "YoutubeAnalysis"))
    cols[2].metric("Settings", "已建立" if youtube_ai_settings_path().exists() else "尚未建立")
    cols[3].metric("Channels", len(read_youtube_ai_channels()))

    st.markdown("**資料庫資訊**")
    info_rows = [
        {"項目": "Workspace", "值": str(workspace)},
        {"項目": "SQLite DB", "值": str(db_path)},
        {"項目": "Schema SQL", "值": str(workspace / "youtube_ai_schema.sql")},
        {"項目": "Channels Config", "值": str(channels_path)},
        {"項目": "Settings Config", "值": str(youtube_ai_settings_path())},
        {"項目": "Supabase URL", "值": os.environ.get("YOUTUBE_AI_SUPABASE_URL", "") or "尚未設定"},
    ]
    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

    init_step = {"name": "初始化 YouTube AI 資料庫", "type": "python", "script": "scripts/youtube_ai_init_db.py", "args": []}
    init_state = runner.get_substep_state({"path": workspace, "ep": 0, "start": 0, "end": 0}, "yt_init")
    if st.button("建立 / 更新資料庫", disabled=bool(init_state.get("running")), key="youtube_ai_init_db"):
        workspace.mkdir(parents=True, exist_ok=True)
        res = runner.start_substep({"path": workspace, "ep": 0, "start": 0, "end": 0}, "yt_init", init_step)
        if res.get("ok"):
            st.success(f"已啟動資料庫初始化：{res.get('log')}")
        else:
            st.error(f"初始化失敗：{res.get('message')}")
    if init_state.get("running"):
        st.info(f"資料庫初始化執行中，PID {init_state.get('pid')}")

    st.divider()
    st.markdown("**OAuth 授權與 Channel 同步**")
    client_secret_path = _root() / "client_secret.json"
    oauth_cols = st.columns(4)
    oauth_cols[0].metric("client_secret.json", "已找到" if client_secret_path.exists() else "缺少")
    oauth_cols[1].metric("OAuth Tokens", len(list((workspace / "oauth_tokens").glob("token_youtube_ai_*.pickle"))) if (workspace / "oauth_tokens").exists() else 0)
    oauth_cols[2].metric("已同步 Channel", len(read_youtube_ai_channels()))
    oauth_cols[3].metric("Scopes", "YouTube + Analytics")
    if not client_secret_path.exists():
        st.warning(f"找不到 OAuth client_secret.json：{client_secret_path}。請先放入 Google OAuth Desktop Client 設定檔。")
    st.caption("新增監控頻道會透過 OAuth 2.0 授權後呼叫 YouTube API 自動取得。不同 Google 帳號請使用不同授權標籤，避免 token 互相覆蓋。")

    oauth_label = st.text_input(
        "授權標籤",
        value="capital_currents",
        key="youtube_ai_oauth_account_label",
        help="建議使用頻道或帳號易懂代號，例如 capital_currents、digital_10。不同頻道請使用不同標籤。",
    )
    oauth_args = ["--account-label", oauth_label.strip() or "default"]

    sync_state = runner.get_substep_state({"path": workspace, "ep": 0, "start": 0, "end": 0}, "yt_oauth_sync")
    oc1, oc2 = st.columns([1, 1])
    with oc1:
        if st.button("新增 / 同步監控頻道", disabled=bool(sync_state.get("running")) or not client_secret_path.exists(), key="youtube_ai_oauth_sync"):
            workspace.mkdir(parents=True, exist_ok=True)
            res = runner.start_substep(
                {"path": workspace, "ep": 0, "start": 0, "end": 0},
                "yt_oauth_sync",
                {"name": "新增 / 同步監控頻道", "type": "python", "script": "scripts/youtube_ai_sync_channels.py", "args": oauth_args},
            )
            if res.get("ok"):
                st.success(f"已啟動監控頻道同步：{res.get('log')}")
            else:
                st.error(f"監控頻道同步啟動失敗：{res.get('message')}")
    with oc2:
        if st.button("新增其他 Google 帳號 / 重新授權", disabled=bool(sync_state.get("running")) or not client_secret_path.exists(), key="youtube_ai_oauth_reauth"):
            res = runner.start_substep(
                {"path": workspace, "ep": 0, "start": 0, "end": 0},
                "yt_oauth_sync",
                {"name": "新增其他 Google 帳號 / 重新授權", "type": "python", "script": "scripts/youtube_ai_sync_channels.py", "args": oauth_args + ["--force-reauth"]},
            )
            if res.get("ok"):
                st.success(f"已啟動重新授權：{res.get('log')}")
            else:
                st.error(f"重新授權啟動失敗：{res.get('message')}")
    if sync_state.get("running"):
        st.info(f"OAuth 同步執行中，PID {sync_state.get('pid')}。若瀏覽器授權視窗未自動開啟，請查看執行環境。")

    st.markdown("**已授權 / 已同步 Channel**")
    channels = read_youtube_ai_channels()
    if channels:
        df = pd.DataFrame(channels)
        display_cols = [c for c in ["channel_id", "title", "handle", "subscriber_count", "video_count", "view_count", "status", "updated_at"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("尚未同步 Channel。請按「OAuth 授權並同步 Channel」。")

    st.divider()
    st.markdown("**Daily ETL 預設參數**")
    with st.form("youtube_ai_settings_form"):
        c1, c2, c3 = st.columns(3)
        etl_days = int(c1.number_input("預設同步天數", min_value=1, max_value=90, value=int(settings.get("etl_days", 32)), step=1))
        comments_per_video = int(c2.number_input("每支影片留言數", min_value=0, max_value=100, value=int(settings.get("comments_per_video", 10)), step=5))
        etl_log_auto_refresh = c3.checkbox("ETL Log 預設自動刷新", value=bool(settings.get("etl_log_auto_refresh", True)))
        submitted = st.form_submit_button("儲存 Settings")
    if submitted:
        saved_path = write_youtube_ai_settings(
            {
                "etl_days": etl_days,
                "comments_per_video": comments_per_video,
                "etl_log_auto_refresh": etl_log_auto_refresh,
            }
        )
        st.success(f"已儲存 Settings：{saved_path}")

def render_youtube_ai_pipeline_manager() -> None:
    st.subheader("YouTube AI Pipeline Manager")
    st.caption("人工執行資料同步、查看 log 與最近執行紀錄。")

    workspace = youtube_ai_workspace_path()
    db_path = youtube_ai_db_path()
    settings = read_youtube_ai_settings()
    etl_state = runner.get_substep_state({"path": workspace, "ep": 0, "start": 0, "end": 0}, "yt_daily_etl")
    etl_log_path = Path(etl_state.get("log") or log_file_for_stage(workspace, "subyt_daily_etl"))

    with st.expander("Daily ETL 指標同步", expanded=False):
        st.caption("同步 views、watch time、traffic source、訂閱歸因、影片 metadata 與留言。")
        e1, e2, e3 = st.columns([1, 1, 2])
        with e1:
            etl_days = int(st.number_input("同步天數", min_value=1, max_value=90, value=int(settings.get("etl_days", 32)), step=1, key="youtube_ai_etl_days"))
        with e2:
            comments_per_video = int(st.number_input("每支影片留言數", min_value=0, max_value=100, value=int(settings.get("comments_per_video", 10)), step=5, key="youtube_ai_comments_per_video"))
        with e3:
            st.write("")
            st.write("")
            if st.button("執行 Daily ETL", disabled=bool(etl_state.get("running")), key="youtube_ai_run_daily_etl"):
                res = runner.start_substep(
                    {"path": workspace, "ep": 0, "start": 0, "end": 0},
                    "yt_daily_etl",
                    {
                        "name": "自動化指標抓取 Daily ETL",
                        "type": "python",
                        "script": "scripts/youtube_ai_daily_etl.py",
                        "args": ["--days", str(etl_days), "--comments-per-video", str(comments_per_video)],
                    },
                )
                if res.get("ok"):
                    st.success(f"已啟動 Daily ETL：{res.get('log')}")
                else:
                    st.error(f"Daily ETL 啟動失敗：{res.get('message')}")
        if etl_state.get("running"):
            st.info(f"Daily ETL 執行中，PID {etl_state.get('pid')}。")

        auto_refresh_etl = st.toggle(
            "自動刷新 Daily ETL Log",
            value=bool(etl_state.get("running")) or bool(settings.get("etl_log_auto_refresh", True)),
            key="youtube_ai_etl_auto_refresh",
            help="Daily ETL 執行中時，每 2 秒刷新狀態與最近 log。",
        )
        etl_refresh_interval = 2 if (auto_refresh_etl and bool(etl_state.get("running"))) else None

        @st.fragment(run_every=etl_refresh_interval)
        def render_youtube_ai_etl_log():
            latest_state = runner.get_substep_state({"path": workspace, "ep": 0, "start": 0, "end": 0}, "yt_daily_etl")
            latest_log_path = Path(latest_state.get("log") or log_file_for_stage(workspace, "subyt_daily_etl"))
            c1, c2, c3 = st.columns(3)
            c1.metric("ETL 狀態", "執行中" if latest_state.get("running") else "閒置")
            c2.metric("PID", latest_state.get("pid") or "—")
            c3.metric("Log", latest_log_path.name)
            if latest_log_path.exists():
                try:
                    log_text = latest_log_path.read_text(encoding="utf-8", errors="ignore")
                except Exception as exc:
                    st.error(f"讀取 log 失敗：{exc}")
                    return
                st.download_button(
                    "下載 Daily ETL Log",
                    data=log_text.encode("utf-8"),
                    file_name=latest_log_path.name,
                    key="youtube_ai_download_etl_log",
                )
                st.code(log_text[-8000:] if log_text else "Log 目前是空的。", language="text")
            else:
                st.caption(f"尚未建立 Daily ETL log：{latest_log_path}")

        with st.expander("Daily ETL Log", expanded=False):
            render_youtube_ai_etl_log()

    with st.expander("Reporting API Reach Reports", expanded=False):
        st.caption("預計同步 YouTube Reporting API bulk CSV 的 video_thumbnail_impressions 與 video_thumbnail_impressions_ctr。")
        st.info("Reporting API 是非同步批次報告。第一次建立 job 後通常需要等待 YouTube 產生日報，Impressions / CTR 也可能有 2–3 天延遲。")
        reach_job_state = runner.get_substep_state({"path": workspace, "ep": 0, "start": 0, "end": 0}, "yt_reach_jobs")
        reach_import_state = runner.get_substep_state({"path": workspace, "ep": 0, "start": 0, "end": 0}, "yt_reach_import")
        r1, r2 = st.columns(2)
        with r1:
            if st.button("建立 / 確認 Reach Report Job", disabled=bool(reach_job_state.get("running")), key="youtube_ai_reach_job"):
                res = runner.start_substep(
                    {"path": workspace, "ep": 0, "start": 0, "end": 0},
                    "yt_reach_jobs",
                    {
                        "name": "建立 / 確認 Reach Report Job",
                        "type": "python",
                        "script": "scripts/youtube_ai_reporting_reach_jobs.py",
                        "args": [],
                    },
                )
                if res.get("ok"):
                    st.success(f"已啟動 Reach Report Job 檢查：{res.get('log')}")
                else:
                    st.error(f"啟動失敗：{res.get('message')}")
        with r2:
            if st.button("下載 / 匯入 Reach Reports", disabled=bool(reach_import_state.get("running")), key="youtube_ai_reach_import"):
                res = runner.start_substep(
                    {"path": workspace, "ep": 0, "start": 0, "end": 0},
                    "yt_reach_import",
                    {
                        "name": "下載 / 匯入 Reach Reports",
                        "type": "python",
                        "script": "scripts/youtube_ai_reporting_reach_import.py",
                        "args": [],
                    },
                )
                if res.get("ok"):
                    st.success(f"已啟動 Reach Reports 匯入：{res.get('log')}")
                else:
                    st.error(f"啟動失敗：{res.get('message')}")
        if reach_job_state.get("running"):
            st.info(f"Reach Report Job 檢查執行中，PID {reach_job_state.get('pid')}。")
        if reach_import_state.get("running"):
            st.info(f"Reach Reports 匯入執行中，PID {reach_import_state.get('pid')}。")

        reach_logs = [
            ("Reach Job Log", Path(reach_job_state.get("log") or log_file_for_stage(workspace, "subyt_reach_jobs"))),
            ("Reach Import Log", Path(reach_import_state.get("log") or log_file_for_stage(workspace, "subyt_reach_import"))),
        ]
        for label, log_path in reach_logs:
            with st.expander(label, expanded=False):
                if log_path.exists():
                    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
                    st.code(log_text[-8000:] if log_text else "Log 目前是空的。", language="text")
                else:
                    st.caption(f"尚未建立 log：{log_path}")

    with st.expander("最近執行紀錄", expanded=False):
        recent_runs = youtube_ai_recent_etl_runs(db_path)
        if not recent_runs.empty:
            st.dataframe(recent_runs, use_container_width=True, hide_index=True)
        else:
            st.caption("尚無 ETL 執行紀錄。")

