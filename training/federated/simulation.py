#!/usr/bin/env python
"""Simulate federated Qwen2.5-0.5B Bloom LoRA training (FedAvg / FedProx)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from training.paths import (
    ARTIFACTS_FEDERATED,
    BUNDLES_DIR,
    ROOT,
    RUNS_DIR,
    UPDATES_DIR,
)
from training.federated.config import (
    DEFAULT_PROX_MU,
    FederatedLoraConfig,
    setting_tag,
)
from training.federated.partition import client_label_distributions, partition_csv

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.federated.run_integrity import (  # noqa: E402
    config_hash,
    dataset_hashes,
    git_revision,
)
from training.federated.execution_stats import summarize_round_bundles
from training.federated.result_report import build_federated_result_report  # noqa: E402
from training.federated.transport import load_bundle  # noqa: E402


def _run(cmd: list[str]) -> None:
    print("[sim]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _mb(num_bytes: int | float) -> float:
    return round(float(num_bytes) / (1024.0 * 1024.0), 4)


def _evaluate_global(global_dir: Path, eval_csv: Path, cfg: FederatedLoraConfig) -> dict:
    import pandas as pd

    from bloom_eval_metrics import evaluate_predictions
    from predict_bloom import BLOOM_LABELS as LABEL_LIST
    from predict_bloom import QwenBloomPredictor
    from training.federated.config import BLOOM_LABELS

    df = pd.read_csv(eval_csv).dropna(subset=[cfg.text_col, cfg.label_col])
    df = df[df[cfg.label_col].isin(BLOOM_LABELS)]
    predictor = QwenBloomPredictor(
        model_dir=str(global_dir),
        base_model=cfg.base_model,
        prefer_merged=False,
        model_size="0.5b",
    )
    label2id = {lab: i for i, lab in enumerate(LABEL_LIST)}
    y_true, y_pred, confidences = [], [], []
    for _, row in df.iterrows():
        out = predictor.predict(str(row[cfg.text_col]))
        y_true.append(label2id[str(row[cfg.label_col])])
        y_pred.append(label2id[out["prediction"]])
        confidences.append(float(out["confidence"]))
    metrics = evaluate_predictions(y_true, y_pred, confidences=confidences, bootstrap_samples=0)
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics.get("per_class"),
        "quadratic_weighted_kappa": metrics.get("quadratic_weighted_kappa"),
        "within_one_level_accuracy": metrics.get("within_one_level_accuracy"),
        "severe_error_rate": metrics.get("severe_error_rate"),
        "ece": metrics.get("ece"),
        "n_eval": len(y_true),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Federated Bloom LoRA simulation.")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=float, default=3.0)
    parser.add_argument("--max-samples-per-client", type=int, default=0)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--train-csv", default=str(ROOT / "data" / "figshare_bloom_v1_train.csv"))
    parser.add_argument("--test-csv", default=str(ROOT / "data" / "figshare_bloom_v1_test.csv"))
    parser.add_argument("--eval-csv", default=str(ROOT / "data" / "figshare_bloom_v1_val.csv"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--prox-mu", type=float, default=DEFAULT_PROX_MU)
    parser.add_argument("--partition", choices=("iid", "non_iid_label", "hash"), default="iid")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-adapter", default=None)
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-each-round", action="store_true", default=True)
    parser.add_argument("--no-eval-each-round", action="store_true")
    parser.add_argument("--from-scratch", action="store_true", default=True)
    parser.add_argument("--allow-central-seed", action="store_true")
    parser.add_argument("--init-adapter", default=None)
    args = parser.parse_args()

    t0 = time.time()
    start_time_iso = datetime.now(timezone.utc).isoformat()
    eval_each_round = bool(args.eval_each_round) and not bool(args.no_eval_each_round)
    tag = setting_tag(algorithm=args.algorithm, partition=args.partition, alpha=args.alpha)
    default_adapter = ARTIFACTS_FEDERATED / "models" / f"qwen_bloom_federated0.5B_{tag}"
    global_dir = Path(args.global_adapter) if args.global_adapter else default_adapter
    results_path = (
        Path(args.results_json)
        if args.results_json
        else ARTIFACTS_FEDERATED / "results" / f"federated_lora_{tag}.json"
    )

    locked = {
        (ROOT / "models" / "qwen_bloom_trained0.5B").resolve(),
        (ROOT / "models" / "qwen_bloom_merged0.5B").resolve(),
        (ROOT / "results" / "bloom_lora_eval_0.5B.json").resolve(),
    }
    if global_dir.resolve() in locked or results_path.resolve() in locked:
        raise SystemExit(f"Refusing to overwrite locked centralized path: {global_dir} / {results_path}")

    cfg = FederatedLoraConfig(
        base_model=args.base_model,
        num_clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        max_samples_per_client=args.max_samples_per_client,
        clip_norm=args.clip_norm,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        global_adapter_dir=str(global_dir),
        algorithm=args.algorithm,
        prox_mu=args.prox_mu,
        partition=args.partition,
        dirichlet_alpha=args.alpha,
        seed=args.seed,
        from_scratch=not args.allow_central_seed,
    )

    run_dir = RUNS_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    config_json = run_dir / "config.json"
    config_json.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    cfg_hash = config_hash(cfg.to_dict())
    round_ckpt_path = run_dir / "round_checkpoint.json"

    start_round = 1
    history: list = []
    total_upload = 0
    total_download = 0
    trainable_parameters = None
    adapter_bytes = None

    if args.resume and round_ckpt_path.is_file():
        ckpt = json.loads(round_ckpt_path.read_text(encoding="utf-8"))
        if ckpt.get("config_hash") != cfg_hash:
            raise SystemExit(
                f"Refusing resume: config_hash mismatch ({ckpt.get('config_hash')} vs {cfg_hash})"
            )
        start_round = int(ckpt.get("last_completed_round", 0)) + 1
        history = list(ckpt.get("history", []))
        comm_acc = ckpt.get("communication_accum", {})
        total_upload = int(comm_acc.get("total_upload", 0))
        total_download = int(comm_acc.get("total_download", 0))
        trainable_parameters = ckpt.get("trainable_parameters")
        adapter_bytes = ckpt.get("adapter_bytes")
        print(f"[sim] resuming from round {start_round} (checkpoint {round_ckpt_path})")

    if global_dir.exists() and not args.resume and not args.skip_train:
        print(f"[sim] fresh start: clearing existing global adapter at {global_dir}")
        shutil.rmtree(global_dir)
    global_dir.mkdir(parents=True, exist_ok=True)

    if args.init_adapter and not args.allow_central_seed:
        raise SystemExit("Refusing --init-adapter without --allow-central-seed")
    if args.init_adapter and args.allow_central_seed and not args.skip_train:
        src = Path(args.init_adapter)
        if not (src / "adapter_config.json").is_file():
            raise FileNotFoundError(f"--init-adapter not found: {src}")
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, global_dir / item.name)
        print(f"[sim] ABLATION: seeded global adapter from {src}")

    for d in (UPDATES_DIR, BUNDLES_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    parts = partition_csv(
        cfg.train_csv,
        num_clients=cfg.num_clients,
        text_col=cfg.text_col,
        label_col=cfg.label_col,
        max_per_client=cfg.max_samples_per_client,
        strategy=cfg.partition,
        dirichlet_alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
    )
    partition_meta = {
        "dataset": cfg.train_csv,
        "dataset_hashes": dataset_hashes(),
        "seed": cfg.seed,
        "partition": cfg.partition,
        "dirichlet_alpha": cfg.dirichlet_alpha if cfg.partition == "non_iid_label" else None,
        "num_clients": cfg.num_clients,
        "client_sizes": {cid: int(len(frame)) for cid, frame in parts.items()},
        "client_label_distribution": client_label_distributions(parts, cfg.label_col),
        "config_hash": cfg_hash,
    }
    (run_dir / "partition.json").write_text(json.dumps(partition_meta, indent=2), encoding="utf-8")
    label_mix = client_label_distributions(parts, cfg.label_col)
    (run_dir / "client_label_distribution.json").write_text(
        json.dumps(label_mix, indent=2), encoding="utf-8"
    )

    client_csvs: dict[str, Path] = {}
    for client_id, frame in parts.items():
        path = UPDATES_DIR / f"{client_id}.csv"
        frame[[cfg.text_col, cfg.label_col]].to_csv(path, index=False)
        client_csvs[client_id] = path
        print(f"[sim] client {client_id}: {len(frame)} rows | mix={label_mix[client_id]}")

    py = sys.executable
    client_script = ROOT / "training" / "federated" / "client.py"
    server_script = ROOT / "training" / "federated" / "server.py"
    eval_csv = Path(args.eval_csv)

    for round_idx in range(start_round, cfg.rounds + 1):
        bundle_paths: list[Path] = []
        for client_id, csv_path in client_csvs.items():
            bundle_path = BUNDLES_DIR / f"round{round_idx:02d}_{client_id}.json"
            bundle_paths.append(bundle_path)
            if not args.skip_train:
                cmd = [
                    py,
                    str(client_script),
                    "--client-id",
                    client_id,
                    "--round",
                    str(round_idx),
                    "--csv",
                    str(csv_path),
                    "--global-adapter",
                    str(global_dir),
                    "--out-bundle",
                    str(bundle_path),
                    "--local-epochs",
                    str(cfg.local_epochs),
                    "--algorithm",
                    cfg.algorithm,
                    "--prox-mu",
                    str(cfg.prox_mu),
                    "--seed",
                    str(cfg.seed),
                    "--base-model",
                    cfg.base_model,
                    "--config-json",
                    str(config_json),
                ]
                _run(cmd)

        if not bundle_paths or not all(p.is_file() for p in bundle_paths):
            raise FileNotFoundError("missing client bundles; run without --skip-train")

        bundles = [load_bundle(p) for p in bundle_paths]
        round_exec = summarize_round_bundles(bundles, round_idx)

        server_cmd = [
            py,
            str(server_script),
            "--bundles",
            *[str(p) for p in bundle_paths],
            "--global-adapter",
            str(global_dir),
            "--clip-norm",
            str(cfg.clip_norm),
            "--algorithm",
            cfg.algorithm,
            "--prox-mu",
            str(cfg.prox_mu),
            "--base-model",
            cfg.base_model,
            "--config-json",
            str(config_json),
        ]
        print("[sim]", " ".join(server_cmd))
        proc = subprocess.run(
            server_cmd,
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        if proc.stdout.strip():
            print(proc.stdout)
        comm = {}
        try:
            text = proc.stdout.strip()
            start = text.rfind("{")
            if start >= 0:
                payload = json.loads(text[start:])
                comm = payload.get("communication") or {}
        except json.JSONDecodeError:
            comm = {}

        upload = int(comm.get("upload_bytes_total") or 0)
        download = int(comm.get("download_bytes_total") or 0)
        total_upload += upload
        total_download += download
        if comm.get("trainable_parameters") is not None:
            trainable_parameters = int(comm["trainable_parameters"])
        if comm.get("adapter_bytes") is not None:
            adapter_bytes = int(comm["adapter_bytes"])

        round_record: dict = {
            "round": round_idx,
            "n_clients": len(bundle_paths),
            "upload_bytes": upload,
            "download_bytes": download,
            "upload_mb": _mb(upload),
            "download_mb": _mb(download),
            "communication_mb": _mb(upload + download),
            "execution": round_exec,
        }
        if eval_each_round and not args.skip_train:
            metrics = _evaluate_global(global_dir, eval_csv, cfg)
            round_record.update(metrics)
            print(f"[sim] round {round_idx} global eval: {metrics}")
        history.append(round_record)

        round_ckpt_path.write_text(
            json.dumps(
                {
                    "last_completed_round": round_idx,
                    "config_hash": cfg_hash,
                    "history": history,
                    "communication_accum": {
                        "total_upload": total_upload,
                        "total_download": total_download,
                    },
                    "trainable_parameters": trainable_parameters,
                    "adapter_bytes": adapter_bytes,
                    "global_adapter": str(global_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    final_metrics = {}
    if not args.skip_train:
        final_metrics = _evaluate_global(global_dir, Path(cfg.test_csv), cfg)

    elapsed = time.time() - t0
    run_id = os.environ.get("EDUGUARD_RUN_ID", "standalone")
    experiment_id = os.environ.get("EDUGUARD_EXPERIMENT_ID", tag)
    client_sizes = {cid: int(len(frame)) for cid, frame in parts.items()}
    comm_block = {
        "trainable_parameters": trainable_parameters,
        "adapter_bytes": adapter_bytes,
        "adapter_size_mb": _mb(adapter_bytes or 0),
        "total_upload_bytes": total_upload,
        "total_download_bytes": total_download,
        "per_round_upload_bytes": [r.get("upload_bytes", 0) for r in history],
        "per_round_download_bytes": [r.get("download_bytes", 0) for r in history],
    }
    report = build_federated_result_report(
        cfg=cfg,
        run_id=run_id,
        experiment_id=experiment_id,
        git_revision=git_revision(),
        config_hash=cfg_hash,
        dataset_hashes=dataset_hashes(),
        client_sample_counts=client_sizes,
        partition_strategy=cfg.partition,
        dirichlet_alpha=float(cfg.dirichlet_alpha) if cfg.partition == "non_iid_label" else None,
        setting_tag=tag,
        global_adapter=str(global_dir),
        results_json_path=str(results_path),
        round_checkpoint_path=str(round_ckpt_path),
        communication=comm_block,
        final_test_metrics=final_metrics if final_metrics else None,
        runtime_seconds=elapsed,
        start_time=start_time_iso,
        history=history,
        status="EXECUTED" if not args.skip_train else "STRUCTURE_ONLY",
    )
    # Backward-compatible aliases used by utility_gap_report / parity scripts
    report["final_test_metrics"] = final_metrics
    report["client_sizes"] = client_sizes
    report["client_label_distribution"] = label_mix
    report["simulation"] = "eduguard_federated_qwen25_0.5b_bloom_lora"
    report["privacy_disclaimer"] = (
        "Federated training keeps raw client data local during collaborative "
        "optimization but does not by itself provide formal protection against "
        "inference attacks on updates, secure aggregation, or differential privacy."
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[sim] done -> {global_dir}")
    print(f"[sim] report -> {results_path}")
    print(json.dumps({"final_test_metrics": final_metrics, "communication": report["communication"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
