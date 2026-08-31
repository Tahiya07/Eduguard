#!/usr/bin/env python
"""Master resumable experiment runner for EduGuard federated privacy research.

Usage:
  python experiments/federated/run_all_experiments.py
  python experiments/federated/run_all_experiments.py --resume
  python experiments/federated/run_all_experiments.py --status
  python experiments/federated/run_all_experiments.py --dry-run
  python experiments/federated/run_all_experiments.py --phase 3
  python experiments/federated/run_all_experiments.py --experiment fedavg_iid
  python experiments/federated/run_all_experiments.py --retry-failed
  python experiments/federated/run_all_experiments.py --new-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.federated.experiment_registry import (  # noqa: E402
    ExperimentSpec,
    build_registry,
    experiment_by_id,
    filter_registry,
)
from experiments.federated.generate_dataset_lock import verify_dataset_lock  # noqa: E402
from experiments.federated.run_integrity import (  # noqa: E402
    artifact_matches_run,
    config_hash,
    git_revision,
    load_json,
    result_is_placeholder,
)

STATE_DIR = ROOT / "experiments" / "federated" / "state"
LOGS_DIR = ROOT / "experiments" / "federated" / "logs"
RESULTS_RUNS = ROOT / "experiments" / "federated" / "results" / "runs"
STATE_FILE = STATE_DIR / "run_state.json"
MASTER_LOG = LOGS_DIR / "master_run.log"
DP_LOCK = ROOT / "artifacts" / "privacy" / "dp_bloom_validated_v1.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str:
    return git_revision(ROOT)


def _env_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(ROOT),
    }
    try:
        import torch

        snap["torch"] = torch.__version__
        snap["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["cuda_device"] = torch.cuda.get_device_name(0)
            snap["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception as exc:
        snap["torch_error"] = str(exc)
    for pkg in ("transformers", "peft", "opacus"):
        try:
            mod = __import__(pkg)
            snap[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            snap[pkg] = None
    return snap


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_state() -> Dict[str, Any]:
    if not STATE_FILE.is_file():
        return {}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _init_state(run_id: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": _utcnow(),
        "last_update": _utcnow(),
        "git_revision": _git_revision(),
        "environment": _env_snapshot(),
        "current_phase": 1,
        "current_experiment": None,
        "completed_experiments": [],
        "failed_experiments": [],
        "interrupted_experiments": [],
        "blocked_experiments": [],
        "skipped_experiments": [],
        "not_executed_experiments": [],
        "checkpoint_paths": {},
        "config_hashes": {},
        "exit_codes": {},
        "timestamps": {},
        "profile": "core",
        "max_hours_hint": None,
        "budget_stop_after_current": False,
        "status": "RUNNING",
    }


def _log_master(line: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    msg = f"[{_utcnow()}] {line}"
    print(msg)
    with MASTER_LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _dp_gate_open() -> bool:
    if not DP_LOCK.is_file():
        return False
    try:
        data = json.loads(DP_LOCK.read_text(encoding="utf-8"))
        return bool(data.get("validation_gate_passed"))
    except Exception:
        return False


def _multitask_gate_open() -> bool:
    marker = ROOT / "artifacts" / "federated" / "results" / "centralized_multitask_success.json"
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        return bool(data.get("success"))
    except Exception:
        return False


def _gate_blocked(spec: ExperimentSpec) -> Optional[str]:
    if spec.gate == "dp_validated" and not _dp_gate_open():
        return "DP validation gate not passed (artifacts/privacy/dp_bloom_validated_v1.json missing or failed)"
    if spec.gate == "multitask_centralized" and not _multitask_gate_open():
        return "Centralized multitask success criteria not met"
    return None


def _gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resource_blocked(spec: ExperimentSpec, allow_gpu: bool, laptop_mode: bool) -> Optional[str]:
    if laptop_mode and spec.resource_class in {"GPU_REQUIRED", "GPU_RECOMMENDED"}:
        return (
            f"{spec.resource_class} blocked in laptop mode "
            "(use --no-laptop-mode --allow-gpu on the GPU machine)"
        )
    if spec.resource_class == "GPU_REQUIRED" and not _gpu_available():
        return "GPU_REQUIRED experiment blocked: CUDA not available"
    if spec.resource_class == "GPU_RECOMMENDED" and not allow_gpu and not _gpu_available():
        return "GPU_RECOMMENDED experiment blocked: CUDA not available and --allow-gpu not set"
    return None


def _config_hash_for_spec(spec: ExperimentSpec) -> Optional[str]:
    if not spec.config_path:
        return None
    p = Path(spec.config_path)
    data = load_json(p)
    return config_hash(data) if data else None


def _validate_experiment_output(spec: ExperimentSpec, state: Dict[str, Any]) -> Optional[str]:
    """Return failure reason if outputs are invalid for completion."""
    if not spec.expected_outputs:
        return None
    run_id = state["run_id"]
    git_rev = state.get("git_revision")

    for rel in spec.expected_outputs:
        p = Path(rel)
        if not p.is_file():
            return f"missing output: {p}"
        if p.suffix == ".json":
            data = load_json(p)
            if data is None:
                return f"invalid JSON: {p}"
            if spec.extra.get("reject_placeholder_status") and result_is_placeholder(data):
                return f"placeholder result (status={data.get('status')})"
            if spec.extra.get("accept_only_if_passed"):
                if not data.get("validation_gate_passed"):
                    return "DP validation did not pass"
            mismatch = artifact_match_failure_reason(
                p,
                run_id=run_id,
                git_rev=git_rev,
                allow_missing_run_id=(spec.resource_class == "CPU_SMOKE"),
            )
            if mismatch:
                return f"artifact does not match active run ({mismatch}): {p}"
    return None


def _archive_state_if_new_run(old_state: Dict[str, Any]) -> None:
    if not old_state or not old_state.get("run_id"):
        return
    archive_dir = RESULTS_RUNS / old_state["run_id"]
    archive_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(archive_dir / "run_state_archived.json", old_state)


def _prereqs_met(spec: ExperimentSpec, completed: Set[str]) -> bool:
    return all(p in completed for p in spec.prerequisites)


def _outputs_valid(spec: ExperimentSpec, state: Dict[str, Any]) -> bool:
    return _validate_experiment_output(spec, state) is None


def _run_subprocess(spec: ExperimentSpec, log_path: Path, state: Dict[str, Any]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["EDUGUARD_RUN_ID"] = state["run_id"]
    env["EDUGUARD_EXPERIMENT_ID"] = spec.experiment_id
    env["EDUGUARD_GIT_REVISION"] = state.get("git_revision", "unknown")
    cfg_h = _config_hash_for_spec(spec)
    if cfg_h:
        env["EDUGUARD_CONFIG_HASH"] = cfg_h
    cmd = list(spec.command)
    _log_master(f"EXEC {spec.experiment_id}: {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# experiment={spec.experiment_id}\n# command={' '.join(cmd)}\n# started={_utcnow()}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
        logf.write(f"\n# exit_code={proc.returncode}\n# ended={_utcnow()}\n")
        return int(proc.returncode or 0)


def _status_label(exp_id: str, state: Dict[str, Any], registry: List[ExperimentSpec]) -> str:
    if exp_id in state.get("not_executed_experiments", []):
        return "NOT_EXECUTED"
    if exp_id in state.get("completed_experiments", []):
        return "COMPLETE"
    if exp_id in state.get("failed_experiments", []):
        return "FAILED"
    if exp_id in state.get("blocked_experiments", []):
        return "BLOCKED"
    if exp_id in state.get("skipped_experiments", []):
        return "SKIPPED"
    if exp_id in state.get("interrupted_experiments", []):
        return "INTERRUPTED"
    if state.get("current_experiment") == exp_id:
        return "RUNNING"
    # Check registry for experiments never started
    for spec in registry:
        if spec.experiment_id == exp_id and spec.extra.get("reject_placeholder_status"):
            for out in spec.expected_outputs:
                data = load_json(Path(out))
                if data and result_is_placeholder(data):
                    return "NOT_EXECUTED"
    return "PENDING"


def print_status(registry: List[ExperimentSpec], state: Dict[str, Any]) -> None:
    phases = sorted({s.phase for s in registry})
    print("EduGuard Federated Research Run")
    print("================================")
    print()
    for phase in phases:
        phase_specs = [s for s in registry if s.phase == phase]
        if not phase_specs:
            continue
        name = phase_specs[0].description.split("—")[0].split("(")[0].strip()[:28]
        statuses = [_status_label(s.experiment_id, state, registry) for s in phase_specs]
        if all(s == "COMPLETE" for s in statuses):
            label = "COMPLETE"
        elif any(s == "RUNNING" for s in statuses):
            label = "RUNNING"
        elif any(s == "FAILED" for s in statuses):
            label = "FAILED"
        elif any(s == "BLOCKED" for s in statuses):
            label = "BLOCKED"
        else:
            label = "PENDING"
        print(f"Phase {phase:<2} {name:<28} {label}")
    print()
    completed = state.get("completed_experiments", [])
    if completed:
        print(f"Last successful:\n    {completed[-1]}")
    if state.get("current_experiment"):
        print(f"Current:\n    {state['current_experiment']}")
    nxt = _next_pending(registry, state)
    if nxt:
        print(f"Next:\n    {nxt.experiment_id}")
    print(f"\nRun ID:\n    {state.get('run_id', 'none')}")
    print(f"DP gate: {'OPEN' if _dp_gate_open() else 'CLOSED'}")
    print(f"Profile: {state.get('profile', 'core')}")
    remaining = [
        s.experiment_id
        for s in registry
        if _status_label(s.experiment_id, state, registry) == "PENDING"
    ]
    if remaining:
        print(f"Remaining ({len(remaining)}): {', '.join(remaining[:8])}{'...' if len(remaining) > 8 else ''}")
    print(f"\nCompleted: {len(state.get('completed_experiments', []))}")
    print(f"Failed: {len(state.get('failed_experiments', []))}")
    print(f"Blocked: {len(state.get('blocked_experiments', []))}")
    print(f"Interrupted: {len(state.get('interrupted_experiments', []))}")


def _next_pending(registry: List[ExperimentSpec], state: Dict[str, Any]) -> Optional[ExperimentSpec]:
    completed = set(state.get("completed_experiments", []))
    skipped = set(state.get("skipped_experiments", []))
    blocked = set(state.get("blocked_experiments", []))
    failed = set(state.get("failed_experiments", []))
    for spec in registry:
        eid = spec.experiment_id
        if eid in completed or eid in skipped or eid in blocked or eid in failed:
            continue
        if _outputs_valid(spec, state) and eid not in state.get("interrupted_experiments", []):
            if _prereqs_met(spec, completed):
                return spec
        elif _prereqs_met(spec, completed):
            return spec
    return None


def _write_manifest(state: Dict[str, Any], registry: List[ExperimentSpec]) -> None:
    run_id = state["run_id"]
    run_dir = RESULTS_RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "git_revision": state.get("git_revision"),
        "environment": state.get("environment"),
        "started_at": state.get("started_at"),
        "last_update": state.get("last_update"),
        "experiments": [
            {
                "experiment_id": s.experiment_id,
                "phase": s.phase,
                "resource_class": s.resource_class,
                "status": _status_label(s.experiment_id, state, registry),
                "config_path": s.config_path,
                "expected_outputs": s.expected_outputs,
            }
            for s in registry
        ],
        "completed": state.get("completed_experiments", []),
        "failed": state.get("failed_experiments", []),
        "blocked": state.get("blocked_experiments", []),
    }
    _atomic_write_json(run_dir / "run_manifest.json", manifest)
    latest = RESULTS_RUNS / "latest"
    if latest.exists() or latest.is_symlink():
        try:
            if latest.is_dir() and not latest.is_symlink():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        except Exception:
            pass
    try:
        latest.symlink_to(run_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(run_dir, latest, dirs_exist_ok=True)


def run_pipeline(
    *,
    resume: bool,
    dry_run: bool,
    phase: Optional[int],
    experiment_id: Optional[str],
    retry_failed: bool,
    new_run: bool,
    allow_gpu: bool,
    laptop_mode: bool,
    profile: str = "core",
    max_hours: Optional[float] = None,
) -> int:
    py = sys.executable
    registry = filter_registry(build_registry(str(ROOT), py), profile)

    if experiment_id:
        registry = [experiment_by_id(registry, experiment_id)]

    if new_run and experiment_id:
        spec = registry[0]
        if spec.prerequisites:
            _log_master(
                "WARNING: --new-run resets run_state.json and clears completed prerequisites. "
                f"{spec.experiment_id} requires {spec.prerequisites}. "
                "Run prerequisites first with --new-run once, then continue with --resume "
                "(omit --new-run), or run the full core profile without --experiment."
            )

    state = _load_state()
    old_state = dict(state) if state else {}
    if new_run and old_state:
        _archive_state_if_new_run(old_state)
    if new_run or not state:
        run_id = _new_run_id()
        state = _init_state(run_id)
        state["profile"] = profile
        state["max_hours_hint"] = max_hours
        if not dry_run:
            _atomic_write_json(STATE_FILE, state)
    elif resume:
        if not state:
            print("No run_state.json to resume; starting new run.")
            state = _init_state(_new_run_id())
            if not dry_run:
                _atomic_write_json(STATE_FILE, state)

    dry_completed: Set[str] = set()
    t_start = time.time()
    max_seconds = float(max_hours) * 3600.0 if max_hours else None

    # Dataset lock (fail fast before expensive GPU work)
    if not laptop_mode and not dry_run:
        ok, issues = verify_dataset_lock(ROOT)
        if not ok:
            for issue in issues:
                _log_master(f"DATASET_LOCK_FAIL: {issue}")
            return 1

    if retry_failed:
        for eid in list(state.get("failed_experiments", [])):
            state["failed_experiments"].remove(eid)
        state["interrupted_experiments"] = []

    completed = set(state.get("completed_experiments", []))

    # Mark already-valid outputs as complete on resume (must match run_id)
    for spec in registry:
        if spec.experiment_id not in completed and _outputs_valid(spec, state) and _prereqs_met(spec, completed):
            if spec.experiment_id not in state.get("failed_experiments", []):
                completed.add(spec.experiment_id)
                if spec.experiment_id not in state.setdefault("completed_experiments", []):
                    state["completed_experiments"].append(spec.experiment_id)
                state.setdefault("timestamps", {})[spec.experiment_id] = _utcnow()
                state.setdefault("exit_codes", {})[spec.experiment_id] = 0
                if not dry_run:
                    _atomic_write_json(STATE_FILE, state)

    try:
        for spec in registry:
            if phase is not None and spec.phase != phase:
                continue

            eid = spec.experiment_id
            if eid in state.get("completed_experiments", []):
                continue
            if eid in state.get("skipped_experiments", []):
                continue
            if eid in state.get("blocked_experiments", []) and not retry_failed:
                continue
            if not _prereqs_met(spec, set(state.get("completed_experiments", [])) | dry_completed):
                if experiment_id:
                    missing = [p for p in spec.prerequisites if p not in state.get("completed_experiments", [])]
                    _log_master(f"SKIPPED {eid}: missing prerequisites {missing}")
                continue

            # Time budget hint: finish current, don't start new non-essential after budget
            if (
                max_seconds is not None
                and (time.time() - t_start) > max_seconds
                and spec.priority == "extended"
            ):
                _log_master(f"SKIPPED {eid}: max-hours budget reached (extended experiment)")
                state.setdefault("skipped_experiments", []).append(eid)
                continue

            gate_reason = _gate_blocked(spec)
            if gate_reason:
                if dry_run:
                    _log_master(f"DRY-RUN would BLOCK {eid}: {gate_reason}")
                    continue
                if eid not in state["blocked_experiments"]:
                    state["blocked_experiments"].append(eid)
                state["last_update"] = _utcnow()
                _atomic_write_json(STATE_FILE, state)
                _log_master(f"BLOCKED {eid}: {gate_reason}")
                if spec.blocking:
                    continue
                else:
                    state["skipped_experiments"].append(eid)
                    continue

            res_reason = _resource_blocked(spec, allow_gpu, laptop_mode)
            if res_reason:
                if dry_run:
                    _log_master(f"DRY-RUN would BLOCK {eid}: {res_reason}")
                    dry_completed.add(eid)
                    continue
                if eid not in state["blocked_experiments"]:
                    state["blocked_experiments"].append(eid)
                _log_master(f"BLOCKED {eid}: {res_reason}")
                state["last_update"] = _utcnow()
                _atomic_write_json(STATE_FILE, state)
                continue

            if dry_run:
                _log_master(f"DRY-RUN would execute {eid}: {' '.join(spec.command)}")
                dry_completed.add(eid)
                continue

            if not dry_run:
                state["current_phase"] = spec.phase
                state["current_experiment"] = eid
                state["last_update"] = _utcnow()
                _atomic_write_json(STATE_FILE, state)

            log_path = LOGS_DIR / f"{eid}.log"
            t0 = time.time()
            try:
                code = _run_subprocess(spec, log_path, state)
            except KeyboardInterrupt:
                state.setdefault("interrupted_experiments", []).append(eid)
                state["status"] = "INTERRUPTED"
                state["last_update"] = _utcnow()
                _atomic_write_json(STATE_FILE, state)
                _write_manifest(state, build_registry(str(ROOT), py))
                _log_master(f"INTERRUPTED at {eid}")
                return 130

            duration = round(time.time() - t0, 2)
            state.setdefault("exit_codes", {})[eid] = code
            state.setdefault("timestamps", {})[eid] = _utcnow()

            if code == 2:
                state.setdefault("not_executed_experiments", []).append(eid)
                _log_master(f"NOT_EXECUTED {eid} (exit=2)")
                _atomic_write_json(STATE_FILE, state)
                continue

            if code != 0:
                state.setdefault("failed_experiments", []).append(eid)
                state["status"] = "FAILED"
                _log_master(f"FAILED {eid} exit={code} duration={duration}s")
                _atomic_write_json(STATE_FILE, state)
                _write_manifest(state, build_registry(str(ROOT), py))
                if spec.blocking:
                    return code
                continue

            output_err = _validate_experiment_output(spec, state)
            if output_err:
                state.setdefault("failed_experiments", []).append(eid)
                _log_master(f"FAILED {eid}: {output_err}")
                _atomic_write_json(STATE_FILE, state)
                if spec.blocking:
                    return 1
                continue

            state.setdefault("completed_experiments", []).append(eid)
            if eid in state.get("failed_experiments", []):
                state["failed_experiments"].remove(eid)
            if eid in state.get("interrupted_experiments", []):
                state["interrupted_experiments"].remove(eid)
            _log_master(f"COMPLETE {eid} duration={duration}s")
            state["last_update"] = _utcnow()
            _atomic_write_json(STATE_FILE, state)

        if not dry_run:
            state["current_experiment"] = None
            state["status"] = "COMPLETE" if not state.get("failed_experiments") else "PARTIAL"
            state["last_update"] = _utcnow()
            _atomic_write_json(STATE_FILE, state)
            _write_manifest(state, build_registry(str(ROOT), py))
        return 0

    except Exception:
        state["status"] = "ERROR"
        state["last_update"] = _utcnow()
        _atomic_write_json(STATE_FILE, state)
        traceback.print_exc()
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="EduGuard federated research master runner")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--phase", type=int, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true", help="Attempt GPU_REQUIRED experiments when CUDA available")
    parser.add_argument(
        "--laptop-mode",
        action="store_true",
        default=True,
        help="Block GPU_REQUIRED experiments (default on resource-constrained laptop)",
    )
    parser.add_argument("--no-laptop-mode", action="store_false", dest="laptop_mode")
    parser.add_argument(
        "--profile",
        choices=("core", "extended", "all"),
        default="core",
        help="Experiment profile: core (publication-critical), extended, or all",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Scheduling hint: skip extended experiments after this many hours",
    )
    args = parser.parse_args()

    py = sys.executable
    registry = build_registry(str(ROOT), py)
    state = _load_state()

    if args.status:
        if not state:
            print("No active run. Start with: python experiments/federated/run_all_experiments.py")
            return 0
        print_status(registry, state)
        return 0

    return run_pipeline(
        resume=args.resume,
        dry_run=args.dry_run,
        phase=args.phase,
        experiment_id=args.experiment,
        retry_failed=args.retry_failed,
        new_run=args.new_run,
        allow_gpu=args.allow_gpu,
        laptop_mode=args.laptop_mode,
        profile=args.profile,
        max_hours=args.max_hours,
    )


if __name__ == "__main__":
    raise SystemExit(main())
