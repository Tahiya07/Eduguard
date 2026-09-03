"""Structured experiment registry for the master GPU research runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Priority = Literal["core", "extended"]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    phase: int
    description: str
    resource_class: str  # CPU_SMOKE | GPU_REQUIRED | GPU_RECOMMENDED
    command: List[str]
    priority: Priority = "core"
    prerequisites: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    blocking: bool = True
    gate: Optional[str] = None
    config_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def filter_registry(registry: List[ExperimentSpec], profile: str) -> List[ExperimentSpec]:
    profile = (profile or "core").lower()
    if profile == "all":
        return list(registry)
    if profile == "extended":
        return [s for s in registry if s.priority == "extended"]
    # default: core only
    return [s for s in registry if s.priority == "core"]


def _sim_cmd(py: str, r: str, **kwargs) -> List[str]:
    cmd = [py, "-m", "training.federated.simulation"]
    mapping = {
        "clients": "--clients",
        "rounds": "--rounds",
        "local_epochs": "--local-epochs",
        "algorithm": "--algorithm",
        "partition": "--partition",
        "alpha": "--alpha",
        "prox_mu": "--prox-mu",
        "results_json": "--results-json",
        "experiment_tag": "--experiment-tag",
        "global_adapter": "--global-adapter",
    }
    for key, flag in mapping.items():
        if key in kwargs and kwargs[key] is not None:
            cmd.extend([flag, str(kwargs[key])])
    if kwargs.get("aggregation_diagnostics"):
        cmd.append("--aggregation-diagnostics")
    if kwargs.get("save_best_checkpoint"):
        cmd.append("--save-best-checkpoint")
    # Simulation auto-resumes from round_checkpoint.json when present; --resume is optional.
    if kwargs.get("resume"):
        cmd.append("--resume")
    return cmd


def build_registry(repo_root: str, py: str) -> List[ExperimentSpec]:
    r = repo_root.replace("\\", "/")
    pytest_cmd = [py, "-m", "pytest", "tests/federated_training/", "-q"]

    return [
        # --- Phase 1: Environment ---
        ExperimentSpec(
            experiment_id="env_validation",
            phase=1,
            description="C1: Environment validation",
            resource_class="CPU_SMOKE",
            priority="core",
            command=[py, "-c", "import training.federated; import pandas; print('env_ok')"],
        ),
        ExperimentSpec(
            experiment_id="unit_tests_federated",
            phase=1,
            description="Federated unit tests",
            resource_class="CPU_SMOKE",
            priority="core",
            command=pytest_cmd,
            prerequisites=["env_validation"],
        ),
        ExperimentSpec(
            experiment_id="generate_environment_lock",
            phase=1,
            description="Record environment lock for reproducibility",
            resource_class="CPU_SMOKE",
            priority="core",
            command=[py, f"{r}/experiments/federated/generate_environment_lock.py"],
            prerequisites=["env_validation"],
            expected_outputs=[f"{r}/artifacts/evaluation/environment_lock.json"],
        ),
        # --- Phase 2: Baseline ---
        ExperimentSpec(
            experiment_id="baseline_runtime_import",
            phase=2,
            description="C2: EduGuard production runtime import check",
            resource_class="CPU_SMOKE",
            priority="core",
            command=[py, "-c", "from backend.service import FrameworkService; print('runtime_ok')"],
            prerequisites=["env_validation"],
        ),
        # --- Phase 3: Smoke + parity prep ---
        ExperimentSpec(
            experiment_id="fl_smoke_fedavg_iid",
            phase=3,
            description="CPU smoke FedAvg+IID (2 clients, 1 round)",
            resource_class="CPU_SMOKE",
            priority="core",
            command=_sim_cmd(
                py,
                r,
                clients=2,
                rounds=1,
                local_epochs=0.05,
                algorithm="fedavg",
                partition="iid",
                results_json=f"{r}/artifacts/federated/runs/smoke/fedavg_iid_smoke.json",
            )
            + ["--no-eval-each-round"],
            prerequisites=["unit_tests_federated"],
            expected_outputs=[f"{r}/artifacts/federated/runs/smoke/fedavg_iid_smoke.json"],
            config_path=f"{r}/experiments/federated/configs/fedavg_iid_smoke.json",
        ),
        # --- Phase 4: Core FL matrix ---
        ExperimentSpec(
            experiment_id="fedavg_iid",
            phase=4,
            description="C3: FedAvg+IID Framework reproduction",
            resource_class="GPU_REQUIRED",
            priority="core",
            command=_sim_cmd(
                py,
                r,
                clients=8,
                rounds=5,
                local_epochs=3,
                algorithm="fedavg",
                partition="iid",
                results_json=f"{r}/artifacts/federated/results/federated_lora_fedavg_iid.json",
            ),
            prerequisites=["fl_smoke_fedavg_iid"],
            expected_outputs=[f"{r}/artifacts/federated/results/federated_lora_fedavg_iid.json"],
            config_path=f"{r}/experiments/federated/configs/fedavg_iid.json",
        ),
        ExperimentSpec(
            experiment_id="framework_parity_audit",
            phase=4,
            description="Framework vs EduGuard reproducibility audit",
            resource_class="CPU_SMOKE",
            priority="core",
            command=[py, f"{r}/experiments/federated/scripts/framework_parity_audit.py"],
            prerequisites=["fedavg_iid"],
            expected_outputs=[
                f"{r}/artifacts/evaluation/framework_parity_audit.json",
                f"{r}/artifacts/evaluation/framework_parity_audit.md",
            ],
        ),
        ExperimentSpec(
            experiment_id="framework_parity_gate",
            phase=4,
            description="Framework parity gate (stop if gap > 5%)",
            resource_class="CPU_SMOKE",
            priority="core",
            command=[py, f"{r}/experiments/federated/scripts/framework_parity_gate.py"],
            prerequisites=["framework_parity_audit"],
            expected_outputs=[f"{r}/artifacts/evaluation/framework_parity_gate.json"],
            extra={"allow_parity_skip": True},
            blocking=False,
        ),
        ExperimentSpec(
            experiment_id="fedavg_iid_r20",
            phase=4,
            description="FedAvg+IID convergence diagnostic (20 rounds)",
            resource_class="GPU_REQUIRED",
            priority="core",
            command=_sim_cmd(
                py,
                r,
                clients=8,
                rounds=20,
                local_epochs=3,
                algorithm="fedavg",
                partition="iid",
                experiment_tag="fedavg_iid_r20",
                global_adapter=f"{r}/artifacts/federated/models/qwen_bloom_federated0.5B_fedavg_iid_r20",
                results_json=f"{r}/artifacts/federated/results/federated_lora_fedavg_iid_r20.json",
                save_best_checkpoint=True,
            ),
            prerequisites=["framework_parity_audit"],
            expected_outputs=[f"{r}/artifacts/federated/results/federated_lora_fedavg_iid_r20.json"],
            config_path=f"{r}/experiments/federated/configs/fedavg_iid_r20.json",
        ),
        ExperimentSpec(
            experiment_id="fedavg_iid_localepoch1",
            phase=4,
            description="FedAvg+IID client-drift diagnostic (1 local epoch)",
            resource_class="GPU_REQUIRED",
            priority="core",
            command=_sim_cmd(
                py,
                r,
                clients=8,
                rounds=5,
                local_epochs=1,
                algorithm="fedavg",
                partition="iid",
                experiment_tag="fedavg_iid_localepoch1",
                global_adapter=f"{r}/artifacts/federated/models/qwen_bloom_federated0.5B_fedavg_iid_localepoch1",
                results_json=f"{r}/artifacts/federated/results/federated_lora_fedavg_iid_localepoch1.json",
            ),
            prerequisites=["fedavg_iid_r20"],
            expected_outputs=[
                f"{r}/artifacts/federated/results/federated_lora_fedavg_iid_localepoch1.json"
            ],
            config_path=f"{r}/experiments/federated/configs/fedavg_iid_localepoch1.json",
        ),
        ExperimentSpec(
            experiment_id="fedprox_iid",
            phase=4,
            description="C4: FedProx+IID (mu=0.01)",
            resource_class="GPU_REQUIRED",
            priority="core",
            command=_sim_cmd(
                py,
                r,
                clients=8,
                rounds=5,
                local_epochs=3,
                algorithm="fedprox",
                prox_mu=0.01,
                partition="iid",
                experiment_tag="fedprox_iid",
                global_adapter=f"{r}/artifacts/federated/models/qwen_bloom_federated0.5B_fedprox_iid",
                results_json=f"{r}/artifacts/federated/results/federated_lora_fedprox_iid.json",
            ),
            prerequisites=["fedavg_iid_localepoch1"],
            expected_outputs=[f"{r}/artifacts/federated/results/federated_lora_fedprox_iid.json"],
            config_path=f"{r}/experiments/federated/configs/fedprox_iid.json",
        ),
        ExperimentSpec(
            experiment_id="fedprox_iid_r20",
            phase=4,
            description="FedProx+IID convergence diagnostic (20 rounds, mu=0.01)",
            resource_class="GPU_REQUIRED",
            priority="extended",
            command=_sim_cmd(
                py,
                r,
                clients=8,
                rounds=20,
                local_epochs=3,
                algorithm="fedprox",
                prox_mu=0.01,
                partition="iid",
                experiment_tag="fedprox_iid_r20",
                global_adapter=f"{r}/artifacts/federated/models/qwen_bloom_federated0.5B_fedprox_iid_r20",
                results_json=f"{r}/artifacts/federated/results/federated_lora_fedprox_iid_r20.json",
                save_best_checkpoint=True,
            ),
            prerequisites=["fedprox_iid"],
            expected_outputs=[f"{r}/artifacts/federated/results/federated_lora_fedprox_iid_r20.json"],
            config_path=f"{r}/experiments/federated/configs/fedprox_iid_r20.json",
            blocking=False,
        ),
        ExperimentSpec(
            experiment_id="fedavg_noniid_a05",
            phase=4,
            description="C5: FedAvg + Dirichlet alpha=0.5",
            resource_class="GPU_REQUIRED",
            priority="core",
            command=_sim_cmd(
                py,
                r,
                clients=8,
                rounds=5,
                local_epochs=3,
                algorithm="fedavg",
                partition="non_iid_label",
                alpha=0.5,
                results_json=f"{r}/artifacts/federated/results/federated_lora_fedavg_noniid_a0.5.json",
            ),
            prerequisites=["fedavg_iid"],
            expected_outputs=[f"{r}/artifacts/federated/results/federated_lora_fedavg_noniid_a0.5.json"],
            config_path=f"{r}/experiments/federated/configs/fedavg_noniid_a05.json",
        ),
        ExperimentSpec(
            experiment_id="fedprox_noniid_a05",
            phase=4,
            description="C6: FedProx + Dirichlet alpha=0.5",
            resource_class="GPU_REQUIRED",
            priority="core",
            command=_sim_cmd(
                py,
                r,
                clients=8,
                rounds=5,
                local_epochs=3,
                algorithm="fedprox",
                prox_mu=0.01,
                partition="non_iid_label",
                alpha=0.5,
                results_json=f"{r}/artifacts/federated/results/federated_lora_fedprox_noniid_a0.5.json",
            ),
            prerequisites=["fedprox_iid"],
            expected_outputs=[f"{r}/artifacts/federated/results/federated_lora_fedprox_noniid_a0.5.json"],
            config_path=f"{r}/experiments/federated/configs/fedprox_noniid_a05.json",
        ),
        ExperimentSpec(
            experiment_id="fedavg_noniid_a01",
            phase=4,
            description="C11: FedAvg + Dirichlet alpha=0.1",
            resource_class="GPU_REQUIRED",
            priority="core",
            command=_sim_cmd(
                py,
                r,
                clients=8,
                rounds=5,
                local_epochs=3,
                algorithm="fedavg",
                partition="non_iid_label",
                alpha=0.1,
                results_json=f"{r}/artifacts/federated/results/federated_lora_fedavg_noniid_a0.1.json",
            ),
            prerequisites=["fedavg_noniid_a05"],
            expected_outputs=[f"{r}/artifacts/federated/results/federated_lora_fedavg_noniid_a0.1.json"],
            config_path=f"{r}/experiments/federated/configs/fedavg_noniid_a01.json",
        ),
        ExperimentSpec(
            experiment_id="fedavg_noniid_a10",
            phase=4,
            description="C12: FedAvg + Dirichlet alpha=1.0",
            resource_class="GPU_REQUIRED",
            priority="core",
            command=_sim_cmd(
                py,
                r,
                clients=8,
                rounds=5,
                local_epochs=3,
                algorithm="fedavg",
                partition="non_iid_label",
                alpha=1.0,
                results_json=f"{r}/artifacts/federated/results/federated_lora_fedavg_noniid_a1.0.json",
            ),
            prerequisites=["fedavg_noniid_a05"],
            expected_outputs=[f"{r}/artifacts/federated/results/federated_lora_fedavg_noniid_a1.0.json"],
            config_path=f"{r}/experiments/federated/configs/fedavg_noniid_a10.json",
        ),
        # --- Phase 5: Utility ---
        ExperimentSpec(
            experiment_id="fl_baseline_diagnosis",
            phase=5,
            description="FL baseline diagnosis across targeted experiments",
            resource_class="CPU_SMOKE",
            priority="core",
            command=[py, f"{r}/experiments/federated/scripts/fl_baseline_diagnosis.py"],
            prerequisites=["fedprox_iid"],
            expected_outputs=[
                f"{r}/artifacts/evaluation/fl_baseline_diagnosis.json",
                f"{r}/artifacts/evaluation/fl_baseline_diagnosis.md",
            ],
        ),
        ExperimentSpec(
            experiment_id="utility_gap_analysis",
            phase=5,
            description="C7: Utility-gap analysis",
            resource_class="GPU_RECOMMENDED",
            priority="core",
            command=[py, f"{r}/experiments/federated/scripts/utility_gap_report.py"],
            prerequisites=["fl_baseline_diagnosis"],
            expected_outputs=[
                f"{r}/artifacts/evaluation/utility_gap_report.json",
                f"{r}/artifacts/evaluation/utility_gap_report.md",
            ],
        ),
        # --- Phase 6-8: DP ---
        ExperimentSpec(
            experiment_id="dp_validation",
            phase=6,
            description="C8: Centralized DP validation gate",
            resource_class="GPU_REQUIRED",
            priority="core",
            command=[
                py,
                "-m",
                "training.centralized.validate_dp_bloom",
                "--output",
                f"{r}/artifacts/privacy/dp_bloom_validated_v1.json",
            ],
            prerequisites=["fedavg_iid"],
            expected_outputs=[f"{r}/artifacts/privacy/dp_bloom_validated_v1.json"],
            extra={"accept_only_if_passed": True, "allow_missing_run_id": True},
        ),
        ExperimentSpec(
            experiment_id="federated_dp",
            phase=8,
            description="C9: Federated DP (blocked until C8 passes)",
            resource_class="GPU_REQUIRED",
            priority="core",
            command=[
                py,
                "-m",
                "training.federated.dp",
                "--mode",
                "federated_train",
                "--noise-multiplier",
                "1.0",
                "--target-delta",
                "1e-5",
            ],
            prerequisites=["dp_validation"],
            expected_outputs=[f"{r}/artifacts/federated/results/federated_dp_fedavg_iid.json"],
            gate="dp_validated",
            blocking=False,
            config_path=f"{r}/experiments/federated/configs/federated_dp_fedavg_iid.json",
        ),
        # --- Phase 9: SecAgg ---
        ExperimentSpec(
            experiment_id="secagg_verification",
            phase=9,
            description="C10: SecAgg simulator verification",
            resource_class="CPU_SMOKE",
            priority="core",
            command=[py, "-m", "pytest", "tests/federated_training/test_secure_aggregation.py", "-q"],
            prerequisites=["unit_tests_federated"],
        ),
        # --- Phase 11: Attacks ---
        ExperimentSpec(
            experiment_id="privacy_attacks",
            phase=11,
            description="C13: Privacy attack evaluation",
            resource_class="GPU_REQUIRED",
            priority="core",
            command=[py, f"{r}/experiments/federated/scripts/run_privacy_attacks.py"],
            prerequisites=["fedavg_iid"],
            expected_outputs=[f"{r}/artifacts/evaluation/privacy_attacks.json"],
            extra={"reject_placeholder_status": True},
        ),
        # --- Phase 14-15: Export / deployment ---
        ExperimentSpec(
            experiment_id="export_federated_artifact",
            phase=14,
            description="C14: Select best FedAvg/FedProx checkpoint, merge, recommend deploy path",
            resource_class="GPU_RECOMMENDED",
            priority="core",
            command=[py, f"{r}/experiments/federated/scripts/select_and_deploy_best_fl_checkpoint.py"],
            prerequisites=["fedavg_iid_r20", "fedprox_iid_r20"],
            expected_outputs=[
                f"{r}/artifacts/evaluation/best_fl_checkpoint_selection.json",
                f"{r}/artifacts/evaluation/deployment_recommendation.json",
            ],
            extra={"allow_missing_run_id": True},
        ),
        ExperimentSpec(
            experiment_id="paper_eval_deployable",
            phase=14,
            description="Paper tables/figures for deployable FL Bloom model",
            resource_class="GPU_RECOMMENDED",
            priority="core",
            command=[
                py,
                f"{r}/experiments/federated/scripts/evaluate_deployable_fl_model.py",
                "--out-dir",
                f"{r}/artifacts/evaluation/paper",
            ],
            prerequisites=["export_federated_artifact"],
            expected_outputs=[
                f"{r}/artifacts/evaluation/paper/paper_main_results.json",
                f"{r}/artifacts/evaluation/paper/PAPER_RESULTS.md",
            ],
            extra={"allow_missing_run_id": True},
            blocking=False,
        ),
        ExperimentSpec(
            experiment_id="deployment_regression",
            phase=15,
            description="C15: Deployment regression validation",
            resource_class="CPU_SMOKE",
            priority="core",
            command=[py, f"{r}/experiments/federated/scripts/deployment_regression.py"],
            prerequisites=["export_federated_artifact"],
            expected_outputs=[f"{r}/artifacts/evaluation/deployment_regression.json"],
        ),
        ExperimentSpec(
            experiment_id="final_research_manifest",
            phase=16,
            description="Final research manifest",
            resource_class="CPU_SMOKE",
            priority="core",
            command=[py, f"{r}/experiments/federated/scripts/build_run_manifest.py"],
            prerequisites=["fedavg_iid"],
            expected_outputs=[f"{r}/experiments/federated/results/runs/latest/run_manifest.json"],
        ),
        # --- EXTENDED (optional) ---
        ExperimentSpec(
            experiment_id="fedprox_noniid_a01",
            phase=4,
            description="E2: FedProx + Dirichlet alpha=0.1",
            resource_class="GPU_REQUIRED",
            priority="extended",
            command=_sim_cmd(
                py,
                r,
                clients=8,
                rounds=5,
                algorithm="fedprox",
                prox_mu=0.01,
                partition="non_iid_label",
                alpha=0.1,
                results_json=f"{r}/artifacts/federated/results/federated_lora_fedprox_noniid_a0.1.json",
            ),
            prerequisites=["fedavg_noniid_a01"],
            expected_outputs=[f"{r}/artifacts/federated/results/federated_lora_fedprox_noniid_a0.1.json"],
        ),
        ExperimentSpec(
            experiment_id="federated_multitask",
            phase=13,
            description="E6: Federated multitask (blocked until centralized succeeds)",
            resource_class="GPU_REQUIRED",
            priority="extended",
            command=[py, "-c", "print('federated multitask not implemented')"],
            prerequisites=["fedavg_iid"],
            gate="multitask_centralized",
            blocking=False,
        ),
    ]


def experiment_by_id(registry: List[ExperimentSpec], experiment_id: str) -> ExperimentSpec:
    for spec in registry:
        if spec.experiment_id == experiment_id:
            return spec
    raise KeyError(f"unknown experiment_id: {experiment_id}")
