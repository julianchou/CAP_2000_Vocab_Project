# CAP 2000 AI Video CMS (Streamlit)

## Quick Start

1. Install deps (same venv as your scripts):
   pip install -r requirements.txt

2. Run the UI:
   streamlit run app.py

3. First time: generate workspace structure with your existing script:
   python ..\setup_cap_2000_project.py

## Notes
- All stages run with the current Python (sys.executable).
- Per-episode status is saved under `workspace/EpXX_YYYY_ZZZZ/status.json`.
- Logs: `workspace/EpXX_YYYY_ZZZZ/00_logs/stage_*.log`.
- Stage definitions: `config/stages.yaml` (edit freely).
- Cost model: `config/costs.yaml` (planning only, tweak to your rates).
