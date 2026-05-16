import argparse
import csv
import io
import json
import os
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

from google.auth.transport.requests import AuthorizedSession, Request
from googleapiclient.discovery import build


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = BASE_DIR / "workspaces" / "youtube_ai"
REACH_KEYWORDS = ("reach", "impression", "thumbnail")


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


def ensure_metric_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(video_daily_metrics)").fetchall()}
    for column in ["lifetime_subscribers_gained", "lifetime_subscribers_lost"]:
        if column not in existing:
            conn.execute(f"ALTER TABLE video_daily_metrics ADD COLUMN {column} INTEGER DEFAULT 0")


def api_list_all(request_factory, collection_key: str, **extra_kwargs) -> list[dict]:
    items = []
    page_token = None
    while True:
        kwargs = dict(extra_kwargs)
        if page_token:
            kwargs["pageToken"] = page_token
        resp = request_factory(**kwargs).execute()
        items.extend(resp.get(collection_key, []) or [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            return items


def list_jobs(service) -> list[dict]:
    def make_request(**kwargs):
        return service.jobs().list(**kwargs)

    return api_list_all(make_request, "jobs")


def list_reports(service, job_id: str) -> list[dict]:
    def make_request(**kwargs):
        return service.jobs().reports().list(jobId=job_id, **kwargs)

    return api_list_all(make_request, "reports")


def is_reach_job(job: dict) -> bool:
    text = f"{job.get('id', '')} {job.get('name', '')} {job.get('reportTypeId', '')}".lower()
    return "youtube_ai_reach" in text or any(keyword in text for keyword in REACH_KEYWORDS)


def first_value(row: dict, names: list[str], default: str = "") -> str:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name])
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return str(value)
    return default


def parse_int(value: str) -> int:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except Exception:
        return 0


def parse_float(value: str) -> float:
    try:
        return float(str(value).replace("%", "").replace(",", "").strip())
    except Exception:
        return 0.0


def report_date(report: dict, row: dict) -> str:
    value = first_value(row, ["date", "day"])
    if value:
        return value[:10]
    start_time = str(report.get("startTime") or "")
    return start_time[:10]


def import_csv(conn, report: dict, csv_text: str) -> int:
    reader = csv.DictReader(io.StringIO(csv_text))
    written = 0
    now = now_iso()
    for row in reader:
        video_id = first_value(row, ["video_id", "videoId", "video"])
        metric_date = report_date(report, row)
        if not video_id or not metric_date:
            continue
        impressions = parse_int(
            first_value(
                row,
                [
                    "video_thumbnail_impressions",
                    "videoThumbnailImpressions",
                    "video_thumbnail_impressions_count",
                ],
            )
        )
        ctr = parse_float(
            first_value(
                row,
                [
                    "video_thumbnail_impressions_ctr",
                    "video_thumbnail_impressions_click_rate",
                    "videoThumbnailImpressionsClickRate",
                    "videoThumbnailImpressionsCtr",
                ],
            )
        )
        if impressions == 0 and ctr == 0:
            continue
        exists = conn.execute("SELECT 1 FROM videos WHERE video_id=? LIMIT 1", (video_id,)).fetchone()
        if not exists:
            continue
        conn.execute(
            """
            INSERT INTO video_daily_metrics(video_id, metric_date, impressions, ctr, raw_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id, metric_date) DO UPDATE SET
                impressions=excluded.impressions,
                ctr=excluded.ctr,
                raw_json=COALESCE(video_daily_metrics.raw_json, excluded.raw_json)
            """,
            (video_id, metric_date, impressions, ctr, json.dumps(row, ensure_ascii=False), now),
        )
        written += 1
    return written


def download_report(creds, download_url: str) -> str:
    session = AuthorizedSession(creds)
    resp = session.get(download_url)
    resp.raise_for_status()
    return resp.text


def run(workspace: Path) -> dict:
    db_path = workspace / "youtube_ai.db"
    if not db_path.exists():
        raise FileNotFoundError(f"missing database: {db_path}")
    started_at = now_iso()
    summary = {
        "started_at": started_at,
        "ended_at": "",
        "status": "running",
        "accounts": 0,
        "jobs": 0,
        "reports": 0,
        "downloaded_reports": 0,
        "rows_written": 0,
        "warnings": [],
        "errors": [],
    }
    conn = sqlite3.connect(db_path)
    ensure_metric_columns(conn)
    try:
        token_rows = oauth_token_rows(conn)
        if not token_rows:
            raise RuntimeError("No active OAuth token found. Run OAuth channel sync first.")
        raw_dir = workspace / "reporting_reach_reports"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for token_row in token_rows:
            account_label = token_row.get("account_email") or f"account_{token_row.get('account_id')}"
            summary["accounts"] += 1
            try:
                creds = load_pickle_credentials(Path(token_row["token_path"]))
                service = build("youtubereporting", "v1", credentials=creds)
                jobs = [job for job in list_jobs(service) if is_reach_job(job)]
                summary["jobs"] += len(jobs)
                if not jobs:
                    summary["warnings"].append({"account": account_label, "warning": "No reach/impression reporting job found."})
                    continue
                for job in jobs:
                    reports = list_reports(service, job.get("id", ""))
                    summary["reports"] += len(reports)
                    for report in reports:
                        download_url = report.get("downloadUrl", "")
                        report_id = report.get("id", "")
                        if not download_url or not report_id:
                            continue
                        raw_path = raw_dir / f"{account_label}_{job.get('reportTypeId', 'report')}_{report_id}.csv"
                        if raw_path.exists():
                            csv_text = raw_path.read_text(encoding="utf-8", errors="ignore")
                        else:
                            csv_text = download_report(creds, download_url)
                            raw_path.write_text(csv_text, encoding="utf-8")
                            summary["downloaded_reports"] += 1
                        summary["rows_written"] += import_csv(conn, report, csv_text)
                conn.commit()
            except Exception as exc:
                summary["errors"].append({"account": account_label, "error": str(exc)})
    finally:
        conn.close()

    summary["ended_at"] = now_iso()
    if summary["errors"] and summary["rows_written"]:
        summary["status"] = "partial_success"
    elif summary["errors"]:
        summary["status"] = "error"
    elif summary["warnings"]:
        summary["status"] = "partial_success"
    else:
        summary["status"] = "success"
    out_dir = workspace / "etl_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"reporting_reach_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["output_path"] = str(out_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and import YouTube Reporting API Reach reports.")
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", str(DEFAULT_WORKSPACE)))
    args = parser.parse_args()
    summary = run(Path(args.workspace_root).resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") in {"success", "partial_success"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
