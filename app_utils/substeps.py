from __future__ import annotations
import os, glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import yaml

@dataclass
class OutputSpec:
    pattern: str
    nonempty: bool = False

@dataclass
class SubStep:
    no: str
    key: str
    name: str
    rule: str  # "any" or "all"
    outputs: List[OutputSpec]


def load_substeps_config(root: Path, path_override: Path | None = None) -> List[SubStep]:
    cfg = (root / path_override) if (path_override and not Path(path_override).is_absolute()) else (path_override or (root / "config" / "substeps.yaml"))
    items: List[SubStep] = []
    if cfg.exists():
        raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        for s in raw.get("substeps", []):
            # YAML 1.1 may interpret the key "no" as boolean False.
            no_val = s.get("no")
            if no_val is None and False in s:
                no_val = s.get(False)
            if no_val is None:
                no_val = s.get("No") or s.get("NO")
            outputs = [OutputSpec(**o) for o in s.get("outputs", [])]
            items.append(SubStep(
                no=str(no_val),
                key=s.get("key", f"step{(s.get('no', no_val))}"),
                name=s.get("name", f"Step {no_val}"),
                rule=s.get("rule", "any"),
                outputs=outputs,
            ))
    return items
def _match_paths(ep_path: Path, pattern: str) -> List[Path]:
    norm = pattern.replace("\\", "/")
    # glob relative to ep_path; allow wildcards
    if any(ch in norm for ch in "*?[]"):
        return [Path(p) for p in glob.glob(str(ep_path / norm))]
    p = ep_path / norm
    return [p] if p.exists() else []


def _sat(p: Path, nonempty: bool) -> bool:
    if not p.exists():
        return False
    if not nonempty:
        return True
    if p.is_file():
        try:
            return p.stat().st_size > 0
        except Exception:
            return False
    if p.is_dir():
        return any(p.iterdir())
    return True


def evaluate_substeps(ep_path: Path, mapping: List[SubStep]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for s in mapping:
        checks: List[bool] = []
        for o in s.outputs:
            matches = _match_paths(ep_path, o.pattern)
            ok = any(_sat(m, o.nonempty) for m in matches) if matches else False
            checks.append(ok)
        ok_all = any(checks) if s.rule == "any" else all(checks) if checks else False
        result[str(s.no)] = "done" if ok_all else "pending"
    return result


def evaluate_substeps_debug(ep_path: Path, mapping: List[SubStep]) -> List[Dict]:
    rows = []
    for s in mapping:
        patterns = []
        checks = []
        for o in s.outputs:
            matches = _match_paths(ep_path, o.pattern)
            sat_list = [ _sat(m, o.nonempty) for m in matches ]
            ok = any(sat_list) if matches else False
            patterns.append({
                "pattern": o.pattern,
                "nonempty": o.nonempty,
                "matched": [str(m) for m in matches[:10]],
                "matched_count": len(matches),
                "satisfied": ok,
            })
            checks.append(ok)
        ok_all = any(checks) if s.rule == "any" else all(checks) if checks else False
        rows.append({
            "no": str(s.no),
            "name": s.name,
            "rule": s.rule,
            "ok": ok_all,
            "patterns": patterns,
        })
    return rows

