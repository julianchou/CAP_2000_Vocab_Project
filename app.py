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
    log_file_for_stage,
)
from app_utils.pipeline import StageRunner, load_stage_config
import yaml
from app_utils.finops import load_cost_model, estimate_batch_cost, gating
from app_utils.ui_helpers import with_emoji
from app_utils.substeps import (
    load_substeps_config,
    evaluate_substeps,
    evaluate_substeps_debug,
)

ROOT = Path(__file__).resolve().parent

PROFILES_CFG = ROOT / "config" / "profiles.yaml"

def load_profiles():
    if PROFILES_CFG.exists():
        data = yaml.safe_load(PROFILES_CFG.read_text(encoding="utf-8")) or {}
        return data.get("profiles", [])
    return []

def get_profile_map():
    profs = load_profiles()
    return {p.get("id"): p for p in profs}

st.set_page_config(page_title="CAP 2000 AI Video CMS", layout="wide")

st.sidebar.title("CAP 2000 控制台")
section = st.sidebar.radio("功能模組", [
    "📊 Dashboard",
    "⚙️ Pipeline Manager",
    "💰 FinOps Monitor",
    "🎬 Studio & Publisher",
], key="nav_radio")

# Common data (profile-aware)
profiles = get_profile_map()
# Sidebar profile switch
profile_ids = list(profiles.keys()) or ["vocab"]
if "profile_id" not in st.session_state:
    st.session_state["profile_id"] = profile_ids[0]
choices = [profiles.get(pid, {"name": pid}).get("name", pid) + f" ({pid})" for pid in profile_ids]
idx_default = profile_ids.index(st.session_state["profile_id"]) if st.session_state["profile_id"] in profile_ids else 0
selected = st.sidebar.selectbox("專案環境", choices, index=idx_default, key="profile_select")
pid = selected.split("(")[-1].rstrip(")")
if pid in profiles and pid != st.session_state["profile_id"]:
    st.session_state["profile_id"] = pid
profile = profiles.get(st.session_state["profile_id"], {})
WS_ROOT = profile.get("workspace", "workspace")
PATH_STAGES = profile.get("stages")
PATH_SUBSTEPS = profile.get("substeps")
PATH_COSTS = profile.get("costs")
stage_cfg = load_stage_config(ROOT, PATH_STAGES)
runner = StageRunner(ROOT, PATH_STAGES, (ROOT / WS_ROOT))
cost_model = load_cost_model(ROOT, PATH_COSTS)
substeps_map = load_substeps_config(ROOT, PATH_SUBSTEPS)

# --- helpers ---

def scan_episodes():
    eps = []
    for p in list_episode_dirs(ROOT, WS_ROOT):
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

# convenience
STAGE_IDS = ["1","1.5","2","3","4","5","6","7"]

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
            ep_choice = st.selectbox("選擇集數以標記手動步驟", list(eps_map.keys()), key="manual_ep")
            target_m = eps_map[ep_choice]
            ep_info = target_m["_raw"]
            colA, colB, colC = st.columns([1, 1, 2])
            with colA:
                s7_cur = has_manual_marker(ep_info["path"], "7")
                s7_new = st.checkbox("Step 7 完成 (NotebookLM)", value=s7_cur, key="manual7")
            with colB:
                s13_cur = has_manual_marker(ep_info["path"], "13")
                s13_new = st.checkbox("Step 13 完成 (人工微調)", value=s13_cur, key="manual13")
            with colC:
                if st.button("更新手動標記", key="manual_update"):
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
            choice = st.selectbox("選擇要檢核的集數", keys_list, index=default_idx, key="detail_ep")
            target = ep_opts.get(choice)
            if target:
                info = target["_raw"]
                st.caption(f"Episode Path: {info['path']}")
                dbg = evaluate_substeps_debug(info["path"], substeps_map)
                rows = []
                for item in dbg:
                    pats = item.get("patterns", [])
                    nonempty_patterns = [p for p in pats if p.get("nonempty")]
                    rule = (item.get("rule") or "any").lower()
                    # Require Nonempty: 是否有任何 pattern 標註 nonempty:true
                    nonempty_required = len(nonempty_patterns) > 0
                    # Nonempty OK: 與子步驟 Rule 對齊
                    if not nonempty_patterns:
                        nonempty_ok = True
                    elif rule == "any":
                        nonempty_ok = any(p.get("satisfied") for p in nonempty_patterns)
                    else:
                        nonempty_ok = all(p.get("satisfied") for p in nonempty_patterns)
                    matched_count = sum(p.get("matched_count",0) for p in pats)
                    matched_paths = []
                    for p in pats:
                        matched_paths.extend(p.get("matched", []))
                    matched_paths = matched_paths[:10]
                    rows.append({
                        "No": item.get("no"),
                        "Name": item.get("name"),
                        "Rule": item.get("rule"),
                        "Step OK": "🟢" if item.get("ok") else "⚪",
                        "Require Nonempty": "✅" if nonempty_required else "",
                        "Nonempty OK": "✅" if nonempty_ok else "❌",
                        "Patterns": ", ".join([p.get("pattern","") for p in pats]),
                        "Matched Count": matched_count,
                        "Matched (up to 10)": "\n".join(matched_paths),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.caption("找不到可檢核的集數。")

# --- Pipeline Manager ---
elif section == "⚙️ Pipeline Manager":
    st.header("工作流引擎與日誌 (Pipeline Manager)")
    eps = scan_episodes()
    if not eps:
        st.info("尚未發現 workspace 內容。請先建立或同步集數資料夾。")
    else:
        ep_map = {f"Ep{e['_raw']['ep']:02d} ({e['Range']})": e for e in eps}
        ep_choice = st.selectbox("選擇集數", list(ep_map.keys()), key="pm_ep")
        target = ep_map[ep_choice]
        info = target["_raw"]
        st.caption(f"Episode Path: {info['path']}")

        # 對應關係 (依 profile)
        def get_mapping(profile_id: str):
            # mode: auto -> 可執行(stage)，manual -> 僅人工，na -> 無對應/外部
            mapping = {
                '1':  [3,4,5,6,7],
                '1.5':[8],
                '2':  [9],
                '3':  [10,11],
                '4':  [12,13,14],
                '5':  [15],
                '6':  [16],
                '7':  [17],
            }
            sub2stage = {}
            exec_mode = {}
            for sid, lst in mapping.items():
                for n in lst:
                    sub2stage[str(n)] = sid
            manual_only = {'7','13'}
            for n in range(3,18):
                ns = str(n)
                if ns in manual_only:
                    exec_mode[ns] = 'manual'
                elif ns in sub2stage:
                    exec_mode[ns] = 'auto'
                else:
                    exec_mode[ns] = 'na'
            return mapping, sub2stage, exec_mode
        stage2subs, sub2stage, exec_mode = get_mapping(st.session_state["profile_id"])
        stage_names = {str(s.get('id')): s.get('name','') for s in (stage_cfg.get('stages') or [])}

        tab_stage, tab_sub = st.tabs(["階段控制 (1–7)", "子步驟 (3–17)"])

        with tab_stage:
            cur = target["Current"]
            cols = st.columns(8)
            for i, sid in enumerate(STAGE_IDS):
                with cols[i]:
                    st.metric(label=f"Stage {sid}", value=cur.get(sid, "pending"))
            st.divider()
            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                sid = st.selectbox("選擇要執行的階段", STAGE_IDS, key="pm_stage")
            with c2:
                run_btn = st.button("執行所選階段", key="pm_run")
            with c3:
                st.write("「執行」後可於下方日誌檢視輸出。")
            if run_btn:
                status = runner.run_stage(info, sid)
                st.success(f"已執行 Stage {sid}，目前狀態：{status['stages'].get(sid)}")
            with st.expander("終端機日誌面板 (Log Viewer)", expanded=False):
                log_tabs = st.tabs([f"Stage {sid}" for sid in STAGE_IDS])
                for sid, tab in zip(STAGE_IDS, log_tabs):
                    with tab:
                        lp = log_file_for_stage(info["path"], sid)
                        if lp.exists():
                            st.download_button("下載 log", data=lp.read_bytes(), file_name=lp.name, key=f"dl_{sid}")
                            st.code(lp.read_text(encoding="utf-8", errors="ignore")[-5000:])
                        else:
                            st.caption("尚無此階段的日誌。")

        with tab_sub:
            st.caption("在此直接執行 3–16（第17發布請到 Studio & Publisher）。")
            dbg = evaluate_substeps_debug(info["path"], substeps_map)
            by_no = {str(r.get('no')): r for r in dbg}
            for n in range(3,18):
                ns = str(n)
                row = by_no.get(ns, {})
                pats = row.get('patterns', [])
                nonempty_patterns = [p for p in pats if p.get('nonempty')]
                rule = (row.get('rule') or 'any').lower()
                nonempty_required = len(nonempty_patterns) > 0
                if not nonempty_patterns:
                    nonempty_ok = True
                elif rule == 'any':
                    nonempty_ok = any(p.get('satisfied') for p in nonempty_patterns)
                else:
                    nonempty_ok = all(p.get('satisfied') for p in nonempty_patterns)
                stage_for = sub2stage.get(ns)
                cols = st.columns([0.6, 2.2, 2.0, 1.2, 2.6])
                with cols[0]:
                    st.markdown(f"**No.{ns}**")
                with cols[1]:
                    st.write(row.get('name',''))
                    if pats:
                        st.caption("Patterns: " + ", ".join([p.get('pattern','') for p in pats]))
                with cols[2]:
                    if stage_for:
                        st.write(f"Stage {stage_for} {stage_names.get(stage_for,'')}")
                    else:
                        st.write('—')
                    st.caption("Step OK: " + ("🟢" if row.get('ok') else "⚪"))
                with cols[3]:
                    st.write("Nonempty OK: " + ("✅" if nonempty_ok else ("❌" if nonempty_required else "")))
                with cols[4]:
                    if ns == '17':
                        if st.button("前往 Studio & Publisher", key=f"goto_pub_{ns}"):
                            st.session_state['nav_radio'] = "🎬 Studio & Publisher"
                            st.experimental_rerun()
                    elif exec_mode.get(ns) == 'auto' and stage_for:
                        if st.button(f"執行對應階段 (Stage {stage_for})", key=f"run_sub_{ns}"):
                            st.session_state['__run_result__'] = runner.run_stage(info, stage_for)
                            st.success(f"已執行 Stage {stage_for}，請稍後於日誌查看輸出。")
                    elif exec_mode.get(ns) == 'manual' and ns in ('7','13'):
                        done = has_manual_marker(info['path'], ns)
                        new_val = st.checkbox("標記完成", value=done, key=f"man_{ns}")
                        if st.button("更新標記", key=f"man_update_{ns}"):
                            set_manual_marker(info['path'], ns, new_val)
                            st.success("已更新手動標記。")
                    else:
                        st.caption("無對應自動化")
            st.divider()
            st.caption("說明：\n- 手動步驟（7,13）可在此切換完成標記。\n- 其它子步驟按鈕會導向執行對應的階段腳本。\n- 第17步 (發布) 請到 Studio & Publisher。")

        with st.expander("1–7 與 3–17 對應關係", expanded=False):
            map_rows = []
            for sid in STAGE_IDS:
                subs = stage2subs.get(sid, [])
                chinese = stage_names.get(sid,'')
                subs_text = ", ".join([f"{x} " + (by_no.get(str(x),{}).get('name','')) for x in subs])
                map_rows.append({"Stage": f"{sid} {chinese}", "Substeps": subs_text})
            st.table(pd.DataFrame(map_rows))# --- FinOps ---
elif section == "💰 FinOps Monitor":
    st.header("成本與配額控管 (FinOps Monitor)")
    st.write("此區塊保留，未變更先前行為。")

# --- Studio & Publisher ---
elif section == "🎬 Studio & Publisher":
    st.header("影音預覽與發布中樞 (Studio & Publisher)")
    st.write("此區塊保留，未變更先前行為。")

