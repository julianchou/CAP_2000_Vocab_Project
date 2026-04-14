import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

from .filesystem import read_status, write_status, log_file_for_stage


def load_stage_config(root: Path) -> Dict:
    cfg_path = root / "config" / "stages.yaml"
    if not cfg_path.exists():
        return {"stages": []}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


class StageRunner:
    def __init__(self, root: Path):
        self.root = root
        self.cfg = load_stage_config(root)

    def _format_args(self, args: List[str], ep_info: Dict) -> List[str]:
        out = []
        for a in args:
            out.append(a.format(**ep_info))
        return out

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
            # Generic external command (e.g., ffmpeg), treat script as executable name
            exe = script_rel
            cmd = [exe] + args

        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n==== RUN {stage_id} | {step_name} ====: {cmd}\n")
            try:
                p = subprocess.Popen(
                    cmd,
                    cwd=str(self.root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                )
                for line in p.stdout:  # type: ignore
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

        # update running status
        status["stages"][stage_id] = "running"
        write_status(ep_info["path"], status)

        # locate stage definition
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

        # run all steps
        ok_all = True
        for step in stage_def.get("steps", []):
            ok, _msg = self._run_step(ep_info, stage_id, step, log_path)
            if not ok:
                ok_all = False
                break

        status["stages"][stage_id] = "done" if ok_all else "error"
        write_status(ep_info["path"], status)
        return status
