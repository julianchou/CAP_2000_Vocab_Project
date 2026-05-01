import argparse
import json
import os
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = BASE_DIR / "workspaces" / "youtube_ai"
CLIENT_SECRETS_FILE = BASE_DIR / "client_secret.json"
SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def token_path(workspace: Path, account_label: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in account_label).strip("_") or "default"
    return workspace / "oauth_tokens" / f"token_youtube_ai_{safe}.pickle"


def load_credentials(workspace: Path, account_label: str, force_reauth: bool):
    if not CLIENT_SECRETS_FILE.exists():
        raise FileNotFoundError(f"missing client_secret.json: {CLIENT_SECRETS_FILE}")
    path = token_path(workspace, account_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    creds = None
    if path.exists() and not force_reauth:
        with path.open("rb") as f:
            creds = pickle.load(f)
    if creds and creds.expired and getattr(creds, "refresh_token", None):
        creds.refresh(Request())
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_FILE), SCOPES)
        creds = flow.run_local_server(port=0)
    with path.open("wb") as f:
        pickle.dump(creds, f)
    return creds, path


def fetch_mine_channels(creds) -> list[dict]:
    youtube = build("youtube", "v3", credentials=creds)
    channels = []
    page_token = None
    while True:
        resp = youtube.channels().list(
            part="id,snippet,statistics,contentDetails",
            mine=True,
            maxResults=50,
            pageToken=page_token,
        ).execute()
        for item in resp.get("items", []):
            snippet = item.get("snippet") or {}
            stats = item.get("statistics") or {}
            channels.append(
                {
                    "channel_id": item.get("id", ""),
                    "title": snippet.get("title", ""),
                    "handle": snippet.get("customUrl", ""),
                    "description": snippet.get("description", ""),
                    "thumbnail_url": ((snippet.get("thumbnails") or {}).get("high") or {}).get("url", ""),
                    "subscriber_count": int(stats.get("subscriberCount", 0) or 0),
                    "video_count": int(stats.get("videoCount", 0) or 0),
                    "view_count": int(stats.get("viewCount", 0) or 0),
                    "status": "active",
                    "sync_enabled": True,
                    "timezone": "Asia/Taipei",
                    "oauth_source": "youtube.channels.list(mine=true)",
                    "updated_at": now_iso(),
                }
            )
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return [item for item in channels if item.get("channel_id")]


def read_channels_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    channels = payload.get("channels") if isinstance(payload, dict) else []
    return channels if isinstance(channels, list) else []


def write_channels_file(workspace: Path, channels: list[dict]) -> Path:
    path = workspace / "config" / "channels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"channels": channels, "updated_at": now_iso(), "source": "oauth"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def sync_db(workspace: Path, channels: list[dict], token_file: Path, account_label: str) -> None:
    db_path = workspace / "youtube_ai.db"
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    try:
        now = now_iso()
        cur = conn.execute(
            """
            INSERT INTO oauth_accounts(provider, account_email, token_path, scopes, status, created_at, updated_at)
            VALUES ('google', ?, ?, ?, 'active', ?, ?)
            """,
            (account_label, str(token_file), json.dumps(SCOPES, ensure_ascii=False), now, now),
        )
        account_id = cur.lastrowid
        for item in channels:
            conn.execute(
                """
                INSERT INTO channels(channel_id, title, handle, account_id, status, sync_enabled, timezone, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    title=excluded.title,
                    handle=excluded.handle,
                    account_id=excluded.account_id,
                    status=excluded.status,
                    sync_enabled=excluded.sync_enabled,
                    timezone=excluded.timezone,
                    notes=excluded.notes,
                    updated_at=excluded.updated_at
                """,
                (
                    item.get("channel_id", ""),
                    item.get("title", "") or item.get("channel_id", ""),
                    item.get("handle", ""),
                    account_id,
                    "active",
                    1,
                    item.get("timezone", "Asia/Taipei"),
                    f"OAuth synced. subscribers={item.get('subscriber_count', 0)}, videos={item.get('video_count', 0)}",
                    now,
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def merge_channels(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = {str(item.get("channel_id", "")).strip(): item for item in existing if str(item.get("channel_id", "")).strip()}
    for item in incoming:
        merged[str(item.get("channel_id", "")).strip()] = item
    return list(merged.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorize YouTube OAuth and sync channels for YouTube AI Growth Expert.")
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", str(DEFAULT_WORKSPACE)))
    parser.add_argument("--account-label", default="default")
    parser.add_argument("--force-reauth", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace_root).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    creds, saved_token = load_credentials(workspace, args.account_label, args.force_reauth)
    incoming = fetch_mine_channels(creds)
    channels_path = workspace / "config" / "channels.json"
    merged = merge_channels(read_channels_file(channels_path), incoming)
    saved_channels = write_channels_file(workspace, merged)
    sync_db(workspace, incoming, saved_token, args.account_label)
    print(f"token={saved_token}")
    print(f"channels={saved_channels}")
    print(f"synced_channel_count={len(incoming)}")
    for item in incoming:
        print(f"- {item.get('title')} ({item.get('channel_id')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
