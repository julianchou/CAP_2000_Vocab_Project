import argparse
import json
import os
import pickle
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from google.auth.transport.requests import Request
from googleapiclient.discovery import build


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = BASE_DIR / "workspaces" / "youtube_ai"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_pickle_credentials(token_path: Path):
    if not token_path.exists():
        raise FileNotFoundError(f"missing OAuth token: {token_path}")
    with token_path.open("rb") as f:
        creds = pickle.load(f)
    if creds and creds.expired and getattr(creds, "refresh_token", None):
        creds.refresh(Request())
        with token_path.open("wb") as f:
            pickle.dump(creds, f)
    if not creds or not creds.valid:
        raise RuntimeError(f"invalid OAuth token: {token_path}")
    return creds


def db_connect(workspace: Path):
    db_path = workspace / "youtube_ai.db"
    if not db_path.exists():
        raise FileNotFoundError(f"missing database: {db_path}. Run database initialization first.")
    conn = sqlite3.connect(db_path)
    ensure_metric_columns(conn)
    conn.commit()
    return conn


def ensure_metric_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(video_daily_metrics)").fetchall()}
    for column in ["lifetime_subscribers_gained", "lifetime_subscribers_lost"]:
        if column not in existing:
            conn.execute(f"ALTER TABLE video_daily_metrics ADD COLUMN {column} INTEGER DEFAULT 0")
    traffic_existing = {row[1] for row in conn.execute("PRAGMA table_info(traffic_sources)").fetchall()}
    traffic_columns = {
        "channel_id": "TEXT",
        "scope": "TEXT DEFAULT 'video'",
        "average_view_duration_seconds": "REAL DEFAULT 0",
    }
    for column, ddl in traffic_columns.items():
        if column not in traffic_existing:
            conn.execute(f"ALTER TABLE traffic_sources ADD COLUMN {column} {ddl}")


def oauth_token_rows(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, account_email, token_path
        FROM oauth_accounts
        WHERE status='active' AND token_path IS NOT NULL
        ORDER BY id DESC
        """
    ).fetchall()
    seen = set()
    result = []
    for account_id, account_email, token_path in rows:
        if token_path in seen:
            continue
        seen.add(token_path)
        result.append({"account_id": account_id, "account_email": account_email or "", "token_path": token_path})
    return result


def fetch_mine_channels(youtube) -> list[dict]:
    resp = youtube.channels().list(part="id,snippet,statistics,contentDetails", mine=True, maxResults=50).execute()
    channels = []
    for item in resp.get("items", []):
        snippet = item.get("snippet") or {}
        stats = item.get("statistics") or {}
        content_details = item.get("contentDetails") or {}
        related_playlists = content_details.get("relatedPlaylists") or {}
        channels.append(
            {
                "channel_id": item.get("id", ""),
                "title": snippet.get("title", ""),
                "handle": snippet.get("customUrl", ""),
                "uploads_playlist_id": related_playlists.get("uploads", ""),
                "subscriber_count": int(stats.get("subscriberCount", 0) or 0),
                "video_count": int(stats.get("videoCount", 0) or 0),
                "view_count": int(stats.get("viewCount", 0) or 0),
            }
        )
    return [item for item in channels if item.get("channel_id")]


def analytics_rows(yta, start_date: str, end_date: str) -> list[dict]:
    metric_sets = [
        [
            "views",
            "estimatedMinutesWatched",
            "averageViewDuration",
            "averageViewPercentage",
            "likes",
            "comments",
            "shares",
        ],
        [
            "views",
            "estimatedMinutesWatched",
            "averageViewDuration",
            "likes",
            "comments",
            "shares",
        ],
        [
            "views",
            "estimatedMinutesWatched",
            "averageViewDuration",
            "averageViewPercentage",
        ],
        ["views", "estimatedMinutesWatched", "averageViewDuration"],
    ]
    errors = []
    for metrics in metric_sets:
        try:
            resp = yta.reports().query(
                ids="channel==MINE",
                startDate=start_date,
                endDate=end_date,
                metrics=",".join(metrics),
                dimensions="video",
                sort="-views",
                maxResults=200,
            ).execute()
            headers = [col.get("name", "") for col in resp.get("columnHeaders", [])]
            rows = []
            for values in resp.get("rows", []) or []:
                rows.append(dict(zip(headers, values)))
            return rows
        except Exception as exc:
            errors.append(f"{','.join(metrics)} => {exc}")
    raise RuntimeError("All YouTube Analytics video-level queries failed. " + " | ".join(errors))


def analytics_subscriber_rows(yta, start_date: str, end_date: str) -> list[dict]:
    resp = yta.reports().query(
        ids="channel==MINE",
        startDate=start_date,
        endDate=end_date,
        metrics="subscribersGained,subscribersLost",
        dimensions="video",
        sort="-subscribersGained",
        maxResults=200,
    ).execute()
    headers = [col.get("name", "") for col in resp.get("columnHeaders", [])]
    rows = []
    for values in resp.get("rows", []) or []:
        rows.append(dict(zip(headers, values)))
    return rows


def analytics_channel_traffic_source_rows(yta, start_date: str, end_date: str) -> list[dict]:
    resp = yta.reports().query(
        ids="channel==MINE",
        startDate=start_date,
        endDate=end_date,
        metrics="views,estimatedMinutesWatched,averageViewDuration",
        dimensions="insightTrafficSourceType",
        sort="-views",
        maxResults=200,
    ).execute()
    headers = [col.get("name", "") for col in resp.get("columnHeaders", [])]
    return [dict(zip(headers, values)) for values in resp.get("rows", []) or []]


def analytics_video_traffic_source_rows(yta, start_date: str, end_date: str, video_ids: list[str]) -> list[dict]:
    rows = []
    for video_id in video_ids[:50]:
        resp = yta.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics="views,estimatedMinutesWatched,averageViewDuration",
            dimensions="insightTrafficSourceType",
            filters=f"video=={video_id}",
            sort="-views",
            maxResults=25,
        ).execute()
        headers = [col.get("name", "") for col in resp.get("columnHeaders", [])]
        for values in resp.get("rows", []) or []:
            item = dict(zip(headers, values))
            item["video"] = video_id
            rows.append(item)
    return rows


def merge_subscriber_rows(rows: list[dict], subscriber_rows: list[dict]) -> None:
    row_by_video = {str(row.get("video", "")).strip(): row for row in rows if row.get("video")}
    for sub_row in subscriber_rows:
        video_id = str(sub_row.get("video", "")).strip()
        if not video_id or video_id not in row_by_video:
            continue
        row_by_video[video_id]["subscribersGained"] = sub_row.get("subscribersGained", 0)
        row_by_video[video_id]["subscribersLost"] = sub_row.get("subscribersLost", 0)


def merge_lifetime_subscriber_rows(rows: list[dict], subscriber_rows: list[dict]) -> None:
    row_by_video = {str(row.get("video", "")).strip(): row for row in rows if row.get("video")}
    for sub_row in subscriber_rows:
        video_id = str(sub_row.get("video", "")).strip()
        if not video_id or video_id not in row_by_video:
            continue
        row_by_video[video_id]["lifetimeSubscribersGained"] = sub_row.get("subscribersGained", 0)
        row_by_video[video_id]["lifetimeSubscribersLost"] = sub_row.get("subscribersLost", 0)


def earliest_video_date(conn: sqlite3.Connection, fallback: str) -> str:
    value = conn.execute("SELECT substr(MIN(published_at), 1, 10) FROM videos WHERE published_at IS NOT NULL AND published_at != ''").fetchone()[0]
    return value or fallback


def fetch_upload_playlist_video_ids(youtube, playlist_id: str, limit: int) -> list[str]:
    if not playlist_id:
        return []
    video_ids = []
    page_token = None
    while len(video_ids) < limit:
        resp = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=min(50, max(limit - len(video_ids), 1)),
            pageToken=page_token,
        ).execute()
        for item in resp.get("items", []) or []:
            video_id = ((item.get("contentDetails") or {}).get("videoId") or "").strip()
            if video_id:
                video_ids.append(video_id)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return video_ids


def chunked(values: list[str], size: int = 50):
    for idx in range(0, len(values), size):
        yield values[idx: idx + size]


def fetch_video_metadata(youtube, video_ids: list[str]) -> list[dict]:
    out = []
    for chunk in chunked(video_ids, 50):
        resp = youtube.videos().list(part="snippet,contentDetails,statistics", id=",".join(chunk), maxResults=50).execute()
        for item in resp.get("items", []) or []:
            snippet = item.get("snippet") or {}
            stats = item.get("statistics") or {}
            thumbnails = snippet.get("thumbnails") or {}
            thumb = (thumbnails.get("maxres") or thumbnails.get("high") or thumbnails.get("medium") or {}).get("url", "")
            out.append(
                {
                    "video_id": item.get("id", ""),
                    "channel_id": snippet.get("channelId", ""),
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "thumbnail_url": thumb,
                    "tags": snippet.get("tags") or [],
                    "statistics": stats,
                    "raw": item,
                }
            )
    return out


def fetch_recent_comments(youtube, video_ids: list[str], per_video: int) -> list[dict]:
    comments = []
    if per_video <= 0:
        return comments
    for video_id in video_ids:
        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(per_video, 100),
                order="time",
                textFormat="plainText",
            ).execute()
        except Exception as exc:
            comments.append({"video_id": video_id, "error": str(exc)})
            continue
        for item in resp.get("items", []) or []:
            top = ((item.get("snippet") or {}).get("topLevelComment") or {})
            snippet = top.get("snippet") or {}
            comments.append(
                {
                    "comment_id": top.get("id", item.get("id", "")),
                    "video_id": video_id,
                    "channel_id": snippet.get("channelId", ""),
                    "author_name": snippet.get("authorDisplayName", ""),
                    "text": snippet.get("textDisplay", ""),
                    "like_count": int(snippet.get("likeCount", 0) or 0),
                    "published_at": snippet.get("publishedAt", ""),
                    "raw": item,
                }
            )
    return comments


def upsert_channel(conn, channel: dict, account_id: int) -> None:
    now = now_iso()
    conn.execute(
        """
        INSERT INTO channels(channel_id, title, handle, account_id, status, sync_enabled, timezone, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'active', 1, 'Asia/Taipei', ?, ?, ?)
        ON CONFLICT(channel_id) DO UPDATE SET
            title=excluded.title,
            handle=excluded.handle,
            account_id=excluded.account_id,
            notes=excluded.notes,
            updated_at=excluded.updated_at
        """,
        (
            channel["channel_id"],
            channel.get("title", "") or channel["channel_id"],
            channel.get("handle", ""),
            account_id,
            f"ETL synced. subscribers={channel.get('subscriber_count', 0)}, videos={channel.get('video_count', 0)}",
            now,
            now,
        ),
    )


def upsert_video(conn, item: dict) -> None:
    now = now_iso()
    conn.execute(
        """
        INSERT INTO videos(channel_id, video_id, title, description, published_at, thumbnail_url, tags_json, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            channel_id=excluded.channel_id,
            title=excluded.title,
            description=excluded.description,
            published_at=excluded.published_at,
            thumbnail_url=excluded.thumbnail_url,
            tags_json=excluded.tags_json,
            updated_at=excluded.updated_at
        """,
        (
            item.get("channel_id", ""),
            item.get("video_id", ""),
            item.get("title", ""),
            item.get("description", ""),
            item.get("published_at", ""),
            item.get("thumbnail_url", ""),
            json.dumps(item.get("tags") or [], ensure_ascii=False),
            now,
            now,
        ),
    )
    conn.execute(
        """
        INSERT INTO video_metadata_snapshots(video_id, snapshot_at, title, description, tags_json, thumbnail_url, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.get("video_id", ""),
            now,
            item.get("title", ""),
            item.get("description", ""),
            json.dumps(item.get("tags") or [], ensure_ascii=False),
            item.get("thumbnail_url", ""),
            json.dumps(item.get("raw") or {}, ensure_ascii=False),
        ),
    )


def upsert_metric(conn, row: dict, video_channel: dict[str, str]) -> None:
    video_id = str(row.get("video", "")).strip()
    metric_date = str(row.get("day", "") or row.get("metric_date", "")).strip()
    if not video_id or not metric_date:
        return
    if video_id not in video_channel:
        return
    now = now_iso()
    conn.execute(
        """
        INSERT INTO video_daily_metrics(
            video_id, metric_date, views, impressions, ctr, average_view_duration_seconds, average_view_percentage,
            likes, comments, shares, subscribers_gained, subscribers_lost,
            lifetime_subscribers_gained, lifetime_subscribers_lost, raw_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id, metric_date) DO UPDATE SET
            views=excluded.views,
            impressions=excluded.impressions,
            ctr=excluded.ctr,
            average_view_duration_seconds=excluded.average_view_duration_seconds,
            average_view_percentage=excluded.average_view_percentage,
            likes=excluded.likes,
            comments=excluded.comments,
            shares=excluded.shares,
            subscribers_gained=excluded.subscribers_gained,
            subscribers_lost=excluded.subscribers_lost,
            lifetime_subscribers_gained=excluded.lifetime_subscribers_gained,
            lifetime_subscribers_lost=excluded.lifetime_subscribers_lost,
            raw_json=excluded.raw_json
        """,
        (
            video_id,
            metric_date,
            int(row.get("views", 0) or 0),
            int(row.get("impressions", 0) or 0),
            float(row.get("impressionClickThroughRate", row.get("ctr", 0)) or 0),
            float(row.get("averageViewDuration", 0) or 0),
            float(row.get("averageViewPercentage", 0) or 0),
            int(row.get("likes", 0) or 0),
            int(row.get("comments", 0) or 0),
            int(row.get("shares", 0) or 0),
            int(row.get("subscribersGained", 0) or 0),
            int(row.get("subscribersLost", 0) or 0),
            int(row.get("lifetimeSubscribersGained", row.get("subscribersGained", 0)) or 0),
            int(row.get("lifetimeSubscribersLost", row.get("subscribersLost", 0)) or 0),
            json.dumps(row, ensure_ascii=False),
            now,
        ),
    )


def upsert_public_stats_metric(conn, item: dict, metric_date: str) -> None:
    video_id = str(item.get("video_id", "")).strip()
    if not video_id:
        return
    stats = item.get("statistics") or {}
    now = now_iso()
    conn.execute(
        """
        INSERT INTO video_daily_metrics(
            video_id, metric_date, views, ctr, average_view_duration_seconds, average_view_percentage,
            likes, comments, shares, subscribers_gained, subscribers_lost, raw_json, created_at
        )
        VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?, 0, 0, 0, ?, ?)
        ON CONFLICT(video_id, metric_date) DO UPDATE SET
            views=excluded.views,
            likes=excluded.likes,
            comments=excluded.comments,
            raw_json=excluded.raw_json
        """,
        (
            video_id,
            metric_date,
            int(stats.get("viewCount", 0) or 0),
            int(stats.get("likeCount", 0) or 0),
            int(stats.get("commentCount", 0) or 0),
            json.dumps({"source": "youtube_data_api_public_statistics", "statistics": stats}, ensure_ascii=False),
            now,
        ),
    )


def update_lifetime_subscribers(conn, subscriber_rows: list[dict], metric_date: str, video_channel: dict[str, str]) -> int:
    updated = 0
    now = now_iso()
    for row in subscriber_rows:
        video_id = str(row.get("video", "")).strip()
        if not video_id or video_id not in video_channel:
            continue
        gained = int(row.get("subscribersGained", 0) or 0)
        lost = int(row.get("subscribersLost", 0) or 0)
        conn.execute(
            """
            INSERT INTO video_daily_metrics(
                video_id, metric_date, lifetime_subscribers_gained, lifetime_subscribers_lost, raw_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id, metric_date) DO UPDATE SET
                lifetime_subscribers_gained=excluded.lifetime_subscribers_gained,
                lifetime_subscribers_lost=excluded.lifetime_subscribers_lost,
                raw_json=COALESCE(video_daily_metrics.raw_json, excluded.raw_json)
            """,
            (
                video_id,
                metric_date,
                gained,
                lost,
                json.dumps({"lifetime_subscribers": row}, ensure_ascii=False),
                now,
            ),
        )
        updated += 1
    return updated


def replace_traffic_sources(
    conn,
    channel_ids: list[str],
    metric_date: str,
    channel_rows: list[dict],
    video_rows: list[dict],
    video_channel: dict[str, str],
) -> int:
    now = now_iso()
    written = 0
    for channel_id in channel_ids:
        conn.execute("DELETE FROM traffic_sources WHERE metric_date=? AND channel_id=?", (metric_date, channel_id))

    for row in channel_rows:
        source_type = str(row.get("insightTrafficSourceType", "")).strip() or "UNKNOWN"
        for channel_id in channel_ids:
            conn.execute(
                """
                INSERT INTO traffic_sources(
                    channel_id, video_id, metric_date, scope, source_type, views,
                    watch_time_minutes, average_view_duration_seconds, raw_json, created_at
                )
                VALUES (?, ?, ?, 'channel', ?, ?, ?, ?, ?, ?)
                """,
                (
                    channel_id,
                    "__CHANNEL__",
                    metric_date,
                    source_type,
                    int(row.get("views", 0) or 0),
                    float(row.get("estimatedMinutesWatched", 0) or 0),
                    float(row.get("averageViewDuration", 0) or 0),
                    json.dumps(row, ensure_ascii=False),
                    now,
                ),
            )
            written += 1

    for row in video_rows:
        video_id = str(row.get("video", "")).strip()
        channel_id = video_channel.get(video_id, "")
        if not video_id or not channel_id:
            continue
        source_type = str(row.get("insightTrafficSourceType", "")).strip() or "UNKNOWN"
        conn.execute(
            """
            INSERT INTO traffic_sources(
                channel_id, video_id, metric_date, scope, source_type, views,
                watch_time_minutes, average_view_duration_seconds, raw_json, created_at
            )
            VALUES (?, ?, ?, 'video', ?, ?, ?, ?, ?, ?)
            """,
            (
                channel_id,
                video_id,
                metric_date,
                source_type,
                int(row.get("views", 0) or 0),
                float(row.get("estimatedMinutesWatched", 0) or 0),
                float(row.get("averageViewDuration", 0) or 0),
                json.dumps(row, ensure_ascii=False),
                now,
            ),
        )
        written += 1
    return written


def insert_comments(conn, items: list[dict]) -> int:
    written = 0
    now = now_iso()
    for item in items:
        if item.get("error") or not item.get("comment_id"):
            continue
        conn.execute(
            """
            INSERT INTO comments(channel_id, video_id, comment_id, author_name, text, like_count, published_at, raw_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(comment_id) DO UPDATE SET
                like_count=excluded.like_count,
                raw_json=excluded.raw_json
            """,
            (
                item.get("channel_id", ""),
                item.get("video_id", ""),
                item.get("comment_id", ""),
                item.get("author_name", ""),
                item.get("text", ""),
                int(item.get("like_count", 0) or 0),
                item.get("published_at", ""),
                json.dumps(item.get("raw") or {}, ensure_ascii=False),
                now,
            ),
        )
        written += 1
    return written


def run_etl(workspace: Path, days: int, comments_per_video: int) -> dict:
    workspace.mkdir(parents=True, exist_ok=True)
    end_day = date.today() - timedelta(days=1)
    start_day = end_day - timedelta(days=max(days, 1) - 1)
    started_at = now_iso()
    summary = {
        "started_at": started_at,
        "ended_at": "",
        "status": "running",
        "start_date": start_day.isoformat(),
        "end_date": end_day.isoformat(),
        "accounts": 0,
        "channels": 0,
        "videos": 0,
        "metric_rows": 0,
        "traffic_source_rows": 0,
        "comments": 0,
        "errors": [],
        "warnings": [],
    }
    conn = db_connect(workspace)
    run_id = None
    try:
        cur = conn.execute(
            "INSERT INTO etl_runs(channel_id, run_type, started_at, status) VALUES (NULL, 'daily_etl', ?, 'running')",
            (started_at,),
        )
        run_id = cur.lastrowid
        token_rows = oauth_token_rows(conn)
        if not token_rows:
            raise RuntimeError("No active OAuth token found. Run OAuth channel sync first.")
        for token_row in token_rows:
            summary["accounts"] += 1
            try:
                creds = load_pickle_credentials(Path(token_row["token_path"]))
                youtube = build("youtube", "v3", credentials=creds)
                yta = build("youtubeAnalytics", "v2", credentials=creds)
                channels = fetch_mine_channels(youtube)
                for channel in channels:
                    upsert_channel(conn, channel, int(token_row["account_id"]))
                summary["channels"] += len(channels)
                rows = []
                try:
                    rows = analytics_rows(yta, start_day.isoformat(), end_day.isoformat())
                    try:
                        subscriber_rows = analytics_subscriber_rows(yta, start_day.isoformat(), end_day.isoformat())
                        merge_subscriber_rows(rows, subscriber_rows)
                    except Exception as sub_exc:
                        summary["warnings"].append(
                            {
                                "account": token_row.get("account_email", ""),
                                "warning": f"YouTube Analytics subscriber-by-video query unavailable. {sub_exc}",
                            }
                        )
                    for row in rows:
                        row["metric_date"] = end_day.isoformat()
                    video_ids = sorted({str(row.get("video", "")).strip() for row in rows if row.get("video")})
                except Exception as exc:
                    warning = (
                        "YouTube Analytics API unavailable; falling back to YouTube Data API public statistics. "
                        f"{exc}"
                    )
                    summary["warnings"].append({"account": token_row.get("account_email", ""), "warning": warning})
                    video_ids = []
                    for channel in channels:
                        video_ids.extend(fetch_upload_playlist_video_ids(youtube, channel.get("uploads_playlist_id", ""), limit=100))
                    video_ids = sorted(set(video_ids))
                metadata = fetch_video_metadata(youtube, video_ids)
                video_channel = {}
                for item in metadata:
                    upsert_video(conn, item)
                    video_channel[item["video_id"]] = item.get("channel_id", "")
                try:
                    active_channel_ids = [str(channel.get("channel_id", "")).strip() for channel in channels if channel.get("channel_id")]
                    channel_traffic_rows = analytics_channel_traffic_source_rows(yta, start_day.isoformat(), end_day.isoformat())
                    video_traffic_rows = analytics_video_traffic_source_rows(yta, start_day.isoformat(), end_day.isoformat(), video_ids)
                    summary["traffic_source_rows"] += replace_traffic_sources(
                        conn,
                        active_channel_ids,
                        end_day.isoformat(),
                        channel_traffic_rows,
                        video_traffic_rows,
                        video_channel,
                    )
                except Exception as traffic_exc:
                    summary["warnings"].append(
                        {
                            "account": token_row.get("account_email", ""),
                            "warning": f"YouTube Analytics traffic source query unavailable. {traffic_exc}",
                        }
                    )
                try:
                    lifetime_start = earliest_video_date(conn, start_day.isoformat())
                    lifetime_subscriber_rows = analytics_subscriber_rows(yta, lifetime_start, end_day.isoformat())
                    merge_lifetime_subscriber_rows(rows, lifetime_subscriber_rows)
                    update_lifetime_subscribers(conn, lifetime_subscriber_rows, end_day.isoformat(), video_channel)
                except Exception as life_exc:
                    summary["warnings"].append(
                        {
                            "account": token_row.get("account_email", ""),
                            "warning": f"YouTube Analytics lifetime subscriber-by-video query unavailable. {life_exc}",
                        }
                    )
                if rows:
                    for row in rows:
                        upsert_metric(conn, row, video_channel)
                else:
                    for item in metadata:
                        upsert_public_stats_metric(conn, item, end_day.isoformat())
                comments = fetch_recent_comments(youtube, video_ids[:25], comments_per_video)
                comment_count = insert_comments(conn, comments)
                summary["videos"] += len(metadata)
                summary["metric_rows"] += len(rows) if rows else len(metadata)
                summary["comments"] += comment_count
                conn.commit()
            except Exception as exc:
                summary["errors"].append({"account": token_row.get("account_email", ""), "error": str(exc)})
        if summary["errors"] and (summary["videos"] or summary["metric_rows"] or summary["comments"]):
            summary["status"] = "partial_success"
        elif summary["errors"]:
            summary["status"] = "error"
        elif summary["warnings"]:
            summary["status"] = "partial_success"
        else:
            summary["status"] = "success"
    except Exception as exc:
        summary["status"] = "error"
        summary["errors"].append({"error": str(exc)})
    finally:
        summary["ended_at"] = now_iso()
        if run_id is not None:
            conn.execute(
                "UPDATE etl_runs SET ended_at=?, status=?, rows_written=?, error=? WHERE id=?",
                (
                    summary["ended_at"],
                    summary["status"],
                    int(summary["metric_rows"]) + int(summary["videos"]) + int(summary["comments"]),
                    json.dumps(summary["errors"], ensure_ascii=False) if summary["errors"] else "",
                    run_id,
                ),
            )
            conn.commit()
        conn.close()
    out_dir = workspace / "etl_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["output_path"] = str(out_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="YouTube AI Growth Expert Daily ETL.")
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", str(DEFAULT_WORKSPACE)))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--comments-per-video", type=int, default=10)
    args = parser.parse_args()

    summary = run_etl(Path(args.workspace_root).resolve(), days=max(args.days, 1), comments_per_video=max(args.comments_per_video, 0))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
