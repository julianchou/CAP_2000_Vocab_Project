import os
import sys
from pathlib import Path
import json
import time
import streamlit as st
import pandas as pd

from app_utils.filesystem import (
    list_episode_dirs,
    parse_episode_info,
    read_status,
    write_status,
    video_output_path,
    subtitles_fixed_path,
    images_dir,
    inputs_txt_path,
    infer_stage_statuses,
    manual_marker_path,
    set_manual_marker,
    has_manual_marker,
)
from app_utils.pipeline import StageRunner, load_stage_config
from app_utils.finops import load_cost_model, estimate_batch_cost, gating
from app_utils.ui_helpers import with_emoji
from app_utils.substeps import (
    load_substeps_config,
    evaluate_substeps,
    evaluate_substeps_debug,
)

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="CAP 2000 AI Video CMS", layout="wide")

st.sidebar.title("CAP 2000 控制台")
section = st.sidebar.radio("功能模組", [
    "📊 Dashboard",
    "⚙️ Pipeline Manager",
    "💰 FinOps Monitor",
    "🎬 Studio & Publisher",
])

# Common data
stage_cfg = load_stage_config(ROOT)
runner = StageRunner(ROOT)
cost_model = load_cost_model(ROOT)
substeps_map = load_substeps_config(ROOT)


# --- helpers ---

def scan_episodes():
    eps = []
    for p in list_episode_dirs(ROOT):
        info = parse_episode_info(p)
        raw_prev = read_status(p)
        computed = infer_stage_statuses(p, prev=raw_prev)
        eps.append({
            "Ep": f"Ep{info['ep']:02d}",
            "Range": f"{info['start']:04d}-{info['end']:04d}",
            "Current": computed,
            "_raw": info,
        })
    return eps


def stage_table(eps):
    rows = []
    for e in eps:
        stg = e["Current"]
        latest = None
        for sid in ["1", "1.5", "2", "3", "4", "5", "6", "7"]:
            s = stg.get(sid, "pending")
            if s in ("running", "error", "done"):
                latest = f"{sid}:{s}"
        rows.append({
            "Ep": e["Ep"],
            "Range": e["Range"],
            "1": with_emoji(stg.get("1", "pending")),
            "1.5": with_emoji(stg.get("1.5", "pending")),
            "2": with_emoji(stg.get("2", "pending")),
            "3": with_emoji(stg.get("3", "pending")),
            "4": with_emoji(stg.get("4", "pending")),
            "5": with_emoji(stg.get("5", "pending")),
            "6": with_emoji(stg.get("6", "pending")),
            "7": with_emoji(stg.get("7", "pending")),
            "Latest": latest or "—",
        })
    return pd.DataFrame(rows)



def compute_substeps_status(eps):
    """Return per-episode No.3–17 status using the same logic
    as the details panel (evaluate_substeps_debug)."""
    out = []
    for e in eps:
        info = e["_raw"]
        dbg = evaluate_substeps_debug(info["path"], substeps_map)
        status = {}
        for row in dbg:
            k = str(row.get("no"))
            if k and str(k).strip().isdigit():
                k = str(int(str(k)))
            status[k] = "done" if row.get("ok") else "pending"
        out.append({
            "Ep": e["Ep"],
            "Range": e["Range"],
            "status": status,
            "_path": str(info["path"]),
            "_rows": dbg,
        })
    return out
def substeps_table(eps):
    rows = []
    snapshot = compute_substeps_status(eps)
    for item in snapshot:
        row = {"Ep": item["Ep"], "Range": item["Range"]}
        for i in range(3, 18):
            k = str(i)
            row[k] = with_emoji(item["status"].get(k, "pending"))
        rows.append(row)
    return pd.DataFrame(rows)


# --- UI ---
if section == "📊 Dashboard":
    st.header("專案與進度總覽")
    eps = scan_episodes()
    if not eps:
        st.info("尚未發現 workspace 內容。請先執行 setup_cap_2000_project.py 建立結構。")
    else:
        # 主表：顯示 15 個子步驟（No.3–17）
        st.caption(f"規則載入：{len(substeps_map)} 項")
        st.dataframe(substeps_table(eps), use_container_width=True)
        # 規則 (debug)：顯示從 YAML 載入的子步驟鍵值
        with st.expander("子步驟規則 (debug)", expanded=False):
            st.write([{ "no": getattr(s, "no", None), "key": getattr(s, "key", None), "name": getattr(s, "name", None)} for s in substeps_map])

        # 附表：原 1–7 階段狀態
        with st.expander("階段總覽 (1–7)", expanded=False):
            st.dataframe(stage_table(eps), use_container_width=True)

        # 手動確認控制（No.7 / No.13）
        with st.expander("手動確認工具 (僅 7 / 13)", expanded=False):
            eps_map = {f"Ep{e['_raw']['ep']:02d}": e for e in eps}
            ep_choice = st.selectbox("選擇集數以標記手動步驟", list(eps_map.keys()))
            target_m = eps_map[ep_choice]
            ep_info = target_m["_raw"]
            colA, colB, colC = st.columns([1, 1, 2])
            with colA:
                s7_cur = has_manual_marker(ep_info["path"], "7")
                s7_new = st.checkbox("Step 7 完成 (NotebookLM)", value=s7_cur)
            with colB:
                s13_cur = has_manual_marker(ep_info["path"], "13")
                s13_new = st.checkbox("Step 13 完成 (人工微調)", value=s13_cur)
            with colC:
                if st.button("更新手動標記"):
                    set_manual_marker(ep_info["path"], "7", s7_new)
                    set_manual_marker(ep_info["path"], "13", s13_new)
                    st.success("已更新手動標記，請重新展開刷新表格或切換頁籤。")

        # 檢核詳情 (可選集數)
        with st.expander("檢核詳情 (No.3–17)", expanded=False):
            ep_opts = {f"{e['Ep']} ({e['Range']})": e for e in eps}
            keys_list = list(ep_opts.keys())
            default_idx = 0
            for idx, k in enumerate(keys_list):
                if k.startswith("Ep04 "):
                    default_idx = idx
                    break
            choice = st.selectbox("選擇要檢核的集數", keys_list, index=default_idx)
            target = ep_opts.get(choice)
            if target:
                info = target["_raw"]
                st.caption(f"Episode Path: {info['path']}")
                dbg = evaluate_substeps_debug(info["path"], substeps_map)
                rows = []
                for item in dbg:
                    for p in item["patterns"]:
                        rows.append({
                            "No": item["no"],
                            "Name": item["name"],
                            "Rule": item["rule"],
                            "Step OK": "🟢" if item.get("ok") else "⚪",
                            "Pattern": p["pattern"],
                            "Nonempty": p["nonempty"],
                            "Matched Count": p["matched_count"],
                            "Matched (up to 10)": "\n".join(p["matched"]) if p["matched"] else "",
                            "Satisfied": "✅" if p["satisfied"] else "❌",
                        })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.caption("找不到可檢核的集數。")

        # 主表狀態來源 (debug 快照)
        with st.expander("主表狀態來源 (debug)", expanded=False):
            snap = compute_substeps_status(eps)
            ep_opts2 = {f"{s['Ep']} ({s['Range']})": s for s in snap}
            keylist2 = list(ep_opts2.keys())
            choice2 = st.selectbox("選擇集數查看主表來源", keylist2, index=0)
            srec = ep_opts2[choice2]
            st.caption(f"Path: {srec['_path']}")
            st.json(srec["status"])
            st.caption("raw rows from evaluate_substeps_debug:")
            st.json(srec.get("_rows", []))
            if st.button("輸出快照 JSON"):
                outp = ROOT / "app_assets" / f"substeps_snapshot_{int(time.time())}.json"
                outp.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
                st.success(f"已輸出 {outp}")

        # 批次任務啟動
        st.subheader("批次任務啟動")
        col1, col2, col3 = st.columns(3)
        with col1:
            start_ep = st.number_input("Start Ep", min_value=1, max_value=200, value=1)
        with col2:
            end_ep = st.number_input("End Ep", min_value=1, max_value=200, value=max(1, int(start_ep)))
        with col3:
            dry_run = st.checkbox("Dry run (不執行，只檢查腳本存在)", value=True)

    plan_eps = int(end_ep - start_ep + 1) if "start_ep" in locals() else 1
    gate = gating(cost_model, plan_eps)
    st.caption(f"成本試算：${gate['totals']['usd']} | Quota：{gate['totals']['quota']}")
    disabled = gate["over_budget"] or gate["over_quota"]
    if disabled:
        st.error("超過預設的成本或配額門檻，請調整參數或下修門檻後再執行。")

    if st.button("啟動流水線", disabled=disabled):
        eps = scan_episodes()
        for e in eps:
            epn = e["_raw"]["ep"]
            if start_ep <= epn <= end_ep:
                if dry_run:
                    st.write(f"[DRY] 檢查 {e['Ep']} 腳本可用性…")
                    for s in stage_cfg.get("stages", []):
                        st.write(f" - Stage {s['id']}: {s['name']}")
                else:
                    for sid in ["1", "1.5", "2", "3", "4", "5", "6", "7"]:
                        st.write(f"執行 {e['Ep']} - Stage {sid}")
                        runner.run_stage(e["_raw"], sid)
                        time.sleep(0.2)
        st.success("批次處理完成（或 DRY 模式檢查完成）。")


elif section == "⚙️ Pipeline Manager":
    st.header("單集階段控制與日誌")
    eps = scan_episodes()
    if not eps:
        st.info("尚未發現 workspace 內容。")
    else:
        options = {f"{e['Ep']} ({e['Range']})": e for e in eps}
        choice = st.selectbox("選擇集數", list(options.keys()))
        target = options[choice]
        info = target["_raw"]
        status = read_status(info["path"]) or {"stages": {}}

        st.write(f"當前狀態（檔案推斷）：{json.dumps(target['Current'], ensure_ascii=False)}")
        st.write(f"狀態檔（原始）：{json.dumps(status.get('stages', {}), ensure_ascii=False)}")

        cols = st.columns(4)
        buttons = ["1", "1.5", "2", "3", "4", "5", "6", "7"]
        for i, sid in enumerate(buttons):
            if cols[i % 4].button(f"Run Stage {sid}"):
                runner.run_stage(info, sid)
                st.experimental_rerun()

        st.subheader("日誌檢視")
        log_s = st.selectbox("選擇 Stage 日誌", buttons, index=0)
        log_path = info["path"] / "00_logs" / f"stage_{log_s.replace('.', '_')}.log"
        if log_path.exists():
            st.code(log_path.read_text(encoding="utf-8"), language="bash")
        else:
            st.caption("尚無日誌。")

elif section == "💰 FinOps Monitor":
    st.header("成本與配額控管 (試算)")
    episodes = st.number_input("預計生產集數", min_value=1, max_value=200, value=3)
    totals = estimate_batch_cost(cost_model, int(episodes))
    st.metric("預估成本 (USD)", totals["usd"])
    st.metric("YouTube Quota (units)", totals["quota"])
    gate = gating(cost_model, int(episodes))
    st.write("是否超過門檻：", gate["over_budget"], gate["over_quota"])
    st.caption("以上為規劃用估算值，實際計費以雲端供應商為準。")

elif section == "🎬 Studio & Publisher":
    st.header("影音預覽與發布")
    eps = scan_episodes()
    if not eps:
        st.info("尚未發現 workspace 內容。")
    else:
        options = {f"{e['Ep']} ({e['Range']})": e for e in eps}
        choice = st.selectbox("選擇集數", list(options.keys()))
        info = options[choice]["_raw"]

        # Video preview
        st.subheader("影片預覽")
        vpath = video_output_path(info["path"])
        if vpath.exists():
            st.video(str(vpath))
        else:
            st.caption(f"找不到影片：{vpath}")

        # Images grid
        st.subheader("分鏡圖片 (部分)")
        imgdir = images_dir(info["path"]) 
        if imgdir.exists():
            imgs = sorted([p for p in imgdir.glob("*.png")])[:24]
            if imgs:
                st.image([str(p) for p in imgs], width=160)
            else:
                st.caption("目前無圖片檔案。")
        else:
            st.caption("圖片資料夾不存在。")

        # SRT editor
        st.subheader("字幕編輯")
        srtp = subtitles_fixed_path(info["path"]) 
        srt_text = srtp.read_text(encoding="utf-8") if srtp.exists() else ""
        updated = st.text_area("notebooklm_audio_fixed.srt", value=srt_text, height=240)
        if st.button("儲存字幕"):
            srtp.parent.mkdir(parents=True, exist_ok=True)
            srtp.write_text(updated, encoding="utf-8")
            st.success("已儲存字幕。")

        # Publish settings
        st.subheader("發布設定")
        privacy = st.selectbox("隱私權", ["private", "unlisted", "public"], index=0)
        schedule_at = st.text_input("排程時間 (YYYY-MM-DD HH:MM)", value="")
        playlist_id = st.text_input("Playlist ID", value="")
        if st.button("寫入 upload_schedule.csv"):
            csv_path = ROOT / "upload_schedule.csv"
            line = f"{info['ep']},{vpath},{privacy},{schedule_at},{playlist_id}\n"
            if not csv_path.exists():
                csv_path.write_text("ep,video,privacy,publish_at,playlist_id\n", encoding="utf-8")
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(line)
            st.success(f"已寫入 {csv_path}")

st.caption("使用相同 Python 環境執行子流程：所有子程式皆以 sys.executable 觸發，並寫入各集數 00_logs/ 下的日誌。")



