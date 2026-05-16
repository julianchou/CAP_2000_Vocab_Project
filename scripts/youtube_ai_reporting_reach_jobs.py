import argparse
import json
import os
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

from google.auth.transport.requests import Request
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


def api_list_all(request_factory, collection_key: str) -> list[dict]:
    items = []
    page_token = None
    while True:
        kwargs = {}
        if page_token:
            kwargs["pageToken"] = page_token
        resp = request_factory(**kwargs).execute()
        items.extend(resp.get(collection_key, []) or [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            return items


def list_report_types(service) -> list[dict]:
    def make_request(**kwargs):
        return service.reportTypes().list(**kwargs)

    return api_list_all(make_request, "reportTypes")


def list_jobs(service) -> list[dict]:
    def make_request(**kwargs):
        return service.jobs().list(**kwargs)

    return api_list_all(make_request, "jobs")


def is_reach_report(report_type: dict) -> bool:
    text = f"{report_type.get('id', '')} {report_type.get('name', '')}".lower()
    return any(keyword in text for keyword in REACH_KEYWORDS)


def ensure_jobs(service, account_label: str, report_types: list[dict]) -> list[dict]:
    existing_jobs = list_jobs(service)
    existing_by_type = {job.get("reportTypeId"): job for job in existing_jobs if job.get("reportTypeId")}
    results = []
    for report_type in report_types:
        report_type_id = report_type.get("id", "")
        if not report_type_id:
            continue
        if report_type.get("systemManaged"):
            results.append({"action": "skipped_system_managed", "report_type": report_type, "job": {}})
            continue
        if report_type_id in existing_by_type:
            job = existing_by_type[report_type_id]
            results.append({"action": "exists", "report_type": report_type, "job": job})
            continue
        name = f"youtube_ai_reach_{account_label}_{report_type_id}"[:100]
        job = service.jobs().create(body={"reportTypeId": report_type_id, "name": name}).execute()
        results.append({"action": "created", "report_type": report_type, "job": job})
    return results


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
        "available_report_types": 0,
        "reach_report_types": 0,
        "jobs_created": 0,
        "jobs_existing": 0,
        "warnings": [],
        "errors": [],
        "accounts_detail": [],
    }
    conn = sqlite3.connect(db_path)
    try:
        token_rows = oauth_token_rows(conn)
    finally:
        conn.close()
    if not token_rows:
        raise RuntimeError("No active OAuth token found. Run OAuth channel sync first.")

    for token_row in token_rows:
        account_label = token_row.get("account_email") or f"account_{token_row.get('account_id')}"
        account_detail = {"account": account_label, "report_types": [], "jobs": []}
        summary["accounts"] += 1
        try:
            creds = load_pickle_credentials(Path(token_row["token_path"]))
            service = build("youtubereporting", "v1", credentials=creds)
            report_types = list_report_types(service)
            reach_types = [item for item in report_types if is_reach_report(item)]
            summary["available_report_types"] += len(report_types)
            summary["reach_report_types"] += len(reach_types)
            account_detail["report_types"] = reach_types
            if not reach_types:
                summary["warnings"].append(
                    {
                        "account": account_label,
                        "warning": "No Reporting API report type containing reach/impression/thumbnail was available for this account.",
                    }
                )
            job_results = ensure_jobs(service, account_label, reach_types)
            for item in job_results:
                if item["action"] == "created":
                    summary["jobs_created"] += 1
                else:
                    summary["jobs_existing"] += 1
            account_detail["jobs"] = job_results
        except Exception as exc:
            summary["errors"].append({"account": account_label, "error": str(exc)})
        summary["accounts_detail"].append(account_detail)

    summary["ended_at"] = now_iso()
    if summary["errors"] and (summary["jobs_created"] or summary["jobs_existing"]):
        summary["status"] = "partial_success"
    elif summary["errors"]:
        summary["status"] = "error"
    elif summary["warnings"]:
        summary["status"] = "partial_success"
    else:
        summary["status"] = "success"

    out_dir = workspace / "etl_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"reporting_reach_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["output_path"] = str(out_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify YouTube Reporting API Reach report jobs.")
    parser.add_argument("--workspace-root", default=os.environ.get("CAP_WORKSPACE_ROOT", str(DEFAULT_WORKSPACE)))
    args = parser.parse_args()
    summary = run(Path(args.workspace_root).resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") in {"success", "partial_success"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
