"""Regression tests for package-safe federated subprocess launch and round integrity."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from training.federated.communication import (
    attach_client_communication_metadata,
    require_bundle_communication,
    require_result_communication,
    require_round_communication,
)
from training.federated.simulation import (
    CLIENT_MODULE,
    SERVER_MODULE,
    _clear_round_failure,
    _write_round_failure,
    internal_module_cmd,
)
from training.federated.transport import pack_update

ROOT = Path(__file__).resolve().parents[2]


def test_internal_module_cmd_uses_dash_m():
    cmd = internal_module_cmd(sys.executable, CLIENT_MODULE, "--help")
    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", CLIENT_MODULE, "--help"]


def test_client_module_launch_from_repo_root():
    proc = subprocess.run(
        internal_module_cmd(sys.executable, CLIENT_MODULE, "--help"),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ModuleNotFoundError" not in (proc.stderr + proc.stdout)


def test_server_module_launch_from_repo_root():
    proc = subprocess.run(
        internal_module_cmd(sys.executable, SERVER_MODULE, "--help"),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ModuleNotFoundError" not in (proc.stderr + proc.stdout)


def _sample_state() -> dict:
    return {
        "layer.lora_A.default": torch.randn(4, 2),
        "layer.lora_B.default": torch.randn(2, 4),
        "layer.score.weight": torch.randn(6, 4),
    }


def test_client_bundle_communication_metadata_after_attach():
    state = _sample_state()
    bundle = pack_update(
        client_id="teacher_site_00",
        round_idx=1,
        role="teacher",
        n_samples=3,
        state=state,
    )
    bundle = attach_client_communication_metadata(bundle, state)
    require_bundle_communication(bundle)
    comm = bundle["communication"]
    assert comm["update_bytes"] > 0
    assert comm["adapter_size_bytes"] == comm["update_bytes"]
    assert comm["trainable_parameter_count"] > 0


def test_missing_communication_metadata_raises():
    bundle = pack_update(
        client_id="teacher_site_00",
        round_idx=1,
        role="teacher",
        n_samples=1,
        state=_sample_state(),
    )
    with pytest.raises(ValueError, match="missing communication block"):
        require_bundle_communication(bundle)


def test_require_round_communication_rejects_zero_upload():
    with pytest.raises(ValueError, match="upload_bytes_total must be positive"):
        require_round_communication(
            {
                "upload_bytes_total": 0,
                "download_bytes_total": 10,
                "per_client_upload_bytes": {"c0": 0},
            },
            context="round 1",
            n_clients=1,
        )


def test_synthetic_round_result_communication_non_zero():
    state = _sample_state()
    bundles = []
    for i in range(2):
        bundle = attach_client_communication_metadata(
            pack_update(
                client_id=f"teacher_site_{i:02d}",
                round_idx=1,
                role="teacher",
                n_samples=5,
                state=state,
            ),
            state,
        )
        bundles.append(bundle)

    upload_total = sum(b["communication"]["update_bytes"] for b in bundles)
    comm = {
        "upload_bytes_total": upload_total,
        "download_bytes_total": 1024 * len(bundles),
        "per_client_upload_bytes": {
            b["client_id"]: b["communication"]["update_bytes"] for b in bundles
        },
        "trainable_parameters": bundles[0]["communication"]["trainable_parameter_count"],
        "adapter_bytes": bundles[0]["communication"]["adapter_size_bytes"],
    }
    require_round_communication(comm, context="synthetic round", n_clients=2)

    result_comm = {
        "total_upload_bytes": upload_total,
        "total_download_bytes": comm["download_bytes_total"],
        "per_round_upload_bytes": [upload_total],
        "per_round_download_bytes": [comm["download_bytes_total"]],
        "trainable_parameters": comm["trainable_parameters"],
        "adapter_bytes": comm["adapter_bytes"],
    }
    require_result_communication(
        result_comm,
        configured_rounds=1,
        completed_rounds=1,
    )
    assert result_comm["total_upload_bytes"] > 0
    assert result_comm["total_download_bytes"] > 0


def test_round_failure_record_written_and_cleared(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_round_failure(run_dir, round_idx=2, client_id="teacher_site_01", error="exit 1")
    payload = json.loads((run_dir / "round_failure.json").read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["round"] == 2
    assert payload["failed_client"] == "teacher_site_01"
    _clear_round_failure(run_dir)
    assert not (run_dir / "round_failure.json").is_file()


def test_simulation_client_subprocess_uses_module_invocation():
    with patch("training.federated.simulation.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        from training.federated.simulation import _run

        cmd = internal_module_cmd(sys.executable, CLIENT_MODULE, "--help")
        _run(cmd)
        mock_run.assert_called_once()
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[1:3] == ["-m", CLIENT_MODULE]
