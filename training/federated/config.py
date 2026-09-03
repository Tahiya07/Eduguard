"""Federated Bloom LoRA configuration aligned with train_qwen_bloom.py."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from training.paths import ROOT

TEACHER_ROLE = "teacher"
STUDENT_ROLE = "student"

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_GLOBAL_LORA = ROOT / "artifacts" / "federated" / "models" / "qwen_bloom_federated0.5B"
DEFAULT_LORA_FALLBACK = ROOT / "models" / "qwen_bloom_trained0.5B"

BLOOM_LABELS = {
    "Remember": 0,
    "Understand": 1,
    "Apply": 2,
    "Analyze": 3,
    "Evaluate": 4,
    "Create": 5,
}
BLOOM_LABEL_ORDER: List[str] = list(BLOOM_LABELS.keys())

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LORA_MODULES_TO_SAVE = ["score"]

DEFAULT_PROX_MU = 0.01


@dataclass
class FederatedLoraConfig:
    base_model: str = DEFAULT_BASE_MODEL
    global_adapter_dir: str = str(DEFAULT_GLOBAL_LORA)
    num_clients: int = 8
    rounds: int = 5
    local_epochs: float = 3.0
    learning_rate: float = 1e-4
    finetune_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    label_smoothing: float = 0.05
    use_class_weights: bool = True
    class_weight_source: str = "global"
    class_weight_max: float = 3.0
    max_grad_norm: float = 1.0
    max_length: int = 256
    batch_size: int = 2
    grad_accum: int = 8
    max_samples_per_client: int = 0
    client_fraction: float = 1.0
    clip_norm: float = 1.0
    seed: int = 42
    train_csv: str = str(ROOT / "data" / "figshare_bloom_v1_train.csv")
    test_csv: str = str(ROOT / "data" / "figshare_bloom_v1_test.csv")
    text_col: str = "question"
    label_col: str = "bloom_level"
    algorithm: str = "fedavg"
    prox_mu: float = DEFAULT_PROX_MU
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    from_scratch: bool = True
    lora_r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    modules_to_save: List[str] = field(default_factory=lambda: list(LORA_MODULES_TO_SAVE))
    target_modules: List[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))

    def lora_config_dict(self) -> Dict[str, Any]:
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "modules_to_save": list(self.modules_to_save),
            "task_type": "SEQ_CLS",
        }

    def experiment_metadata(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model,
            "algorithm": self.algorithm,
            "prox_mu": float(self.prox_mu) if self.algorithm == "fedprox" else 0.0,
            "partition": self.partition,
            "dirichlet_alpha": float(self.dirichlet_alpha) if self.partition == "non_iid_label" else None,
            "from_scratch": bool(self.from_scratch),
            "num_clients": int(self.num_clients),
            "rounds": int(self.rounds),
            "participation": "full" if self.client_fraction >= 1.0 else float(self.client_fraction),
            "seed": int(self.seed),
            "lora": self.lora_config_dict(),
            "train_csv": self.train_csv,
            "test_csv": self.test_csv,
            "clip_norm": self.clip_norm,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_peft_lora_config(config: FederatedLoraConfig | None = None):
    from peft import LoraConfig, TaskType

    cfg = config or FederatedLoraConfig()
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        modules_to_save=list(cfg.modules_to_save),
    )


def effective_prox_mu(algorithm: str, prox_mu: float) -> float:
    """FedProx mu is ignored unless the algorithm is actually FedProx."""
    if str(algorithm).lower().strip() != "fedprox":
        return 0.0
    return float(prox_mu)


def setting_tag(*, algorithm: str, partition: str, alpha: float | None = None) -> str:
    algo = algorithm.lower().strip()
    part = partition.lower().strip()
    if part == "non_iid_label":
        a = 0.5 if alpha is None else float(alpha)
        return f"{algo}_noniid_a{a:g}"
    return f"{algo}_{part}"
