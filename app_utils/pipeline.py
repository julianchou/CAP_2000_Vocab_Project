import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

from .filesystem import read_status, write_status, log_file_for_stage


def load_stage_config(root: Path, path_override: Path | None = None) -> Dict:
    """Load stages config from override path or default config/stages.yaml."""
    if path_override is not None:
        cfg_path = (root / path_override) if not Path(path_override).is_absolute() else Path(path_override)
    else:
        cfg_path = root / "config" / "stages.yaml"
    if not cfg_path.exists():
        return {"stages": []}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


class StageRunner:
    def __init__(self, root: Path, stages_path: Path | None = None, workspace_root: Path | None = None):
        self.root = root
        self.cfg = load_stage_config(root, stages_path)
        self.workspace_root = (workspace_root or (root / "workspace")).resolve()

    def _format_args(self, args: List[str], ep_info: Dict) -> List[str]:
        return [a.format(**ep_info) for a in args]

    def _script_exists(self, script_rel: str) -> bool:
        return (self.root / script_rel).exists()

    def _run_step(self, ep_info: Dict, stage_id: str, step: Dict, log_path: Path) -> Tuple[bool, str]:
        """Run a single step. Returns (ok, message)."""
        step_name = step.get("name", "step")
        step_type = step.get("type", "python")
        script_rel = step.get("script")
        args = self._format_args(step.get("args", []), ep_info)

        if step.get("skip_if_dir_has_files"):
            check_dir = (ep_info["path"] / step["skip_if_dir_has_files"]).resolve()
            if check_dir.exists() and any(check_dir.iterdir()):
                return True, f"skip {step_name}: already has files in {check_dir}"

        if step_type == "python":
            script_abs = (self.root / script_rel).resolve()
            if not script_abs.exists():
                if step.get("optional"):
                    return True, f"optional script missing: {script_rel}"
                return False, f"script not found: {script_rel}"
            cmd = [sys.executable, str(script_abs)] + args
        else:
            # Generic external command (e.g., ffmpeg)
            cmd = [script_rel] + args

        env = dict(os.environ)
        env["CAP_WORKSPACE_ROOT"] = str(self.workspace_root)

        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n==== RUN {stage_id} | {step_name} ====: {cmd}\n")
            try:
                p = subprocess.Popen(
                    cmd,
                    cwd=str(self.root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                )
                assert p.stdout is not None
                for line in p.stdout:
                    lf.write(line)
                code = p.wait()
                if code == 0:
                    lf.write(f"==== OK {stage_id} | {step_name} ===\n")
                    return True, "ok"
                else:
                    lf.write(f"==== FAIL {stage_id} | {step_name} | code={code} ===\n")
                    return False, f"exit {code}"
            except FileNotFoundError:
                lf.write(f"==== FAIL {stage_id} | {step_name} | FileNotFoundError ===\n")
                return False, "FileNotFound"
            except Exception as e:
                lf.write(f"==== FAIL {stage_id} | {step_name} | {e} ===\n")
                return False, str(e)

    def run_stage(self, ep_info: Dict, stage_id: str) -> Dict:
        status = read_status(ep_info["path"]) or {"stages": {}}
        status.setdefault("stages", {})
        log_path = log_file_for_stage(ep_info["path"], stage_id)

        status["stages"][stage_id] = "running"
        write_status(ep_info["path"], status)

        stage_def = None
        for s in self.cfg.get("stages", []):
            if str(s.get("id")) == str(stage_id):
                stage_def = s
                break
        if not stage_def:
            status["stages"][stage_id] = "error"
            write_status(ep_info["path"], status)
            Path(log_path).write_text(f"Stage {stage_id} not defined in config\n", encoding="utf-8")
            return status

        ok_all = True
        for step in stage_def.get("steps", []):
            ok, _msg = self._run_step(ep_info, stage_id, step, log_path)
            if not ok:
                ok_all = False
                break

        status["stages"][stage_id] = "done" if ok_all else "error"
        write_status(ep_info["path"], status)
        return status
