#!/usr/bin/env python
"""Federated rounds over teacher-confirmed Bloom labels (deployment-side FL).

The deployed classifier is the merged FedProx r20 model. A live federated cycle
continues from it: every site trains a LoRA adapter *on top of the deployed
model* using the labels its teachers confirmed in the app
(``data/federated_local/confirmed_labels.jsonl``), and sends back only the
adapter weights. Raw questions never leave the site.

Commands (run in order; ``init``/``aggregate``/``promote`` at the coordinator,
``train`` at every site):

  init       coordinator: create the shared round-0 adapter on the deployed model
  status     site: show how many confirmed labels this machine has
  train      site: one local round -> update bundle (weights only)
  aggregate  coordinator: FedProx/FedAvg over the received bundles -> new global adapter
  promote    coordinator: merge, evaluate on the held-out Figshare split, and
             accept only if accuracy does not drop; then sites switch BLOOM_MODEL_DIR

Privacy scope: raw text stays local; bundles carry integrity hashes but are not
encrypted and carry no differential-privacy guarantee (see transport.py).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from training.paths import ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.label_store import BLOOM_LEVELS, LocalLabelStore  # noqa: E402  (no ML deps)

DEPLOYED_MODEL = ROOT / "models" / "qwen_bloom_federated0.5B_fedprox_iid_r20_best_r20_merged"
LABEL_STORE = ROOT / "data" / "federated_local" / "confirmed_labels.jsonl"
# Overridable for smoke tests so nothing is written into the repository.
SITE_DIR = Path(os.environ.get("EDUGUARD_FL_SITE_DIR", ROOT / "data" / "federated_local"))
CYCLE_DIR = Path(os.environ.get("EDUGUARD_FL_LIVE_DIR", ROOT / "artifacts" / "federated" / "live"))
HELDOUT_CSV = ROOT / "data" / "figshare_bloom_v1_val.csv"
MIN_LABELS = 30
SEED = 42


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _client_config(base_model: Path, out: Path) -> Path:
    """Local-round settings: warm start from the shared adapter, FedProx, local class weights.

    Sites have small label sets, so local epochs and batch size are reduced and class
    weights come from the site's own labels (the Figshare train CSV is not shipped to sites).
    """
    cfg = {
        "base_model": str(base_model),
        "local_epochs": 2.0,
        "batch_size": 2,
        "grad_accum": 4,
        "finetune_learning_rate": 5e-5,
        "class_weight_source": "local",
        "use_class_weights": True,
        "seed": SEED,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out


def export_site_csv(store: LocalLabelStore, out_csv: Path, min_labels: int) -> dict:
    import pandas as pd

    rows = store.rows()
    stats = store.stats()
    if len(rows) < min_labels:
        raise SystemExit(
            f"Only {len(rows)} confirmed labels on this machine (need >= {min_labels}). "
            "Teachers confirm levels in Teacher Studio -> 'Confirm level'."
        )
    present = [lvl for lvl, n in stats["per_level"].items() if n]
    if len(present) < 2:
        raise SystemExit(f"Labels cover only {present}; a local round needs at least two Bloom levels.")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"question": r["question"], "bloom_level": r["bloom_level"]} for r in rows]
    ).to_csv(out_csv, index=False)
    return stats


def cmd_status(args) -> int:
    store = LocalLabelStore(Path(args.labels), args.site_id)
    print(json.dumps(store.stats(), indent=2))
    return 0


def cmd_init(args) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification

    from peft import get_peft_model
    from training.federated.aggregation import extract_trainable_state
    from training.federated.config import BLOOM_LABELS, FederatedLoraConfig, make_peft_lora_config
    from training.federated.server import _save_global_adapter

    base = Path(args.base_model)
    out = Path(args.out)
    if (out / "adapter_config.json").is_file() and not args.force:
        raise SystemExit(f"{out} already has an adapter; pass --force to recreate round 0.")
    torch.manual_seed(SEED)  # one shared LoRA init; every site warm-starts from this exact file
    cfg = FederatedLoraConfig(base_model=str(base), seed=SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(base), num_labels=len(BLOOM_LABELS), torch_dtype=torch.float32
    )
    model = get_peft_model(model, make_peft_lora_config(cfg))
    _save_global_adapter(cfg, extract_trainable_state(model), out)
    (out / "cycle.json").write_text(
        json.dumps({"base_model": str(base), "round": 0, "created": _now()}, indent=2), encoding="utf-8"
    )
    print(f"Round-0 adapter written to {out}. Copy this folder to every site.")
    return 0


def cmd_train(args) -> int:
    global_dir = Path(args.global_adapter)
    cycle = json.loads((global_dir / "cycle.json").read_text(encoding="utf-8"))
    base = Path(cycle["base_model"]) if not args.base_model else Path(args.base_model)
    if not (global_dir / "adapter_config.json").is_file():
        raise SystemExit(f"No shared adapter in {global_dir}; run 'init' at the coordinator first.")
    store = LocalLabelStore(Path(args.labels), args.site_id)
    work = SITE_DIR / "rounds" / f"round_{args.round:02d}"
    csv_path = work / "site_labels.csv"  # stays on this machine
    stats = export_site_csv(store, csv_path, args.min_labels)
    cfg_path = _client_config(base, work / "client_config.json")
    out_bundle = Path(args.out_bundle or (work / f"{args.site_id}_round{args.round:02d}.json"))
    cmd = [
        sys.executable, "-m", "training.federated.client",
        "--client-id", args.site_id, "--round", str(args.round),
        "--csv", str(csv_path), "--global-adapter", str(global_dir),
        "--out-bundle", str(out_bundle), "--base-model", str(base),
        "--algorithm", args.algorithm, "--config-json", str(cfg_path),
    ]
    print("[local_round]", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        return rc
    (work / "round_summary.json").write_text(
        json.dumps({"site_id": args.site_id, "round": args.round, "labels": stats,
                    "bundle": str(out_bundle), "finished": _now()}, indent=2), encoding="utf-8"
    )
    print(f"\nUpdate bundle: {out_bundle}\nSend ONLY this file to the coordinator. "
          f"{csv_path} contains question text and must stay on this machine.")
    return 0


def cmd_aggregate(args) -> int:
    global_dir = Path(args.global_adapter)
    cycle_path = global_dir / "cycle.json"
    cycle = json.loads(cycle_path.read_text(encoding="utf-8"))
    bundles = sorted(Path(p) for p in args.bundles)
    if len(bundles) < args.min_sites:
        raise SystemExit(f"{len(bundles)} bundles received; need >= {args.min_sites} sites for a round.")
    cmd = [
        sys.executable, "-m", "training.federated.server",
        "--bundles", *map(str, bundles), "--global-adapter", str(global_dir),
        "--base-model", cycle["base_model"], "--algorithm", args.algorithm,
        "--comm-report", str(CYCLE_DIR / f"round_{args.round:02d}_comm.json"),
    ]
    print("[local_round]", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc == 0:
        cycle.update(round=args.round, aggregated=_now(), sites=[b.name for b in bundles])
        cycle_path.write_text(json.dumps(cycle, indent=2), encoding="utf-8")
    return rc


def _accuracy(model_dir: Path, csv: Path, limit: int) -> float:
    import pandas as pd

    from predict_bloom import QwenBloomPredictor

    df = pd.read_csv(csv).dropna(subset=["question", "bloom_level"])
    if limit:
        df = df.head(limit)
    pred = QwenBloomPredictor(model_dir=str(model_dir), model_size="0.5b")
    hits = sum(pred.predict(str(q))["prediction"] == str(y) for q, y in zip(df["question"], df["bloom_level"]))
    return hits / max(1, len(df))


def cmd_promote(args) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    global_dir = Path(args.global_adapter)
    cycle = json.loads((global_dir / "cycle.json").read_text(encoding="utf-8"))
    base = Path(cycle["base_model"])
    out = Path(args.out or (ROOT / "models" / f"qwen_bloom_federated0.5B_live_round{cycle['round']:02d}_merged"))

    model = AutoModelForSequenceClassification.from_pretrained(str(base), torch_dtype=torch.float32)
    merged = PeftModel.from_pretrained(model, str(global_dir)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(str(base)).save_pretrained(out)

    held = Path(args.heldout_csv)
    old_acc = _accuracy(base, held, args.limit)
    new_acc = _accuracy(out, held, args.limit)
    accepted = new_acc + args.tolerance >= old_acc
    report = {
        "round": cycle["round"], "sites": cycle.get("sites"), "heldout_csv": str(held),
        "n_eval": args.limit or "all", "accuracy_current": round(old_acc, 4),
        "accuracy_candidate": round(new_acc, 4), "tolerance": args.tolerance,
        "accepted": accepted, "candidate_dir": str(out), "evaluated": _now(),
    }
    CYCLE_DIR.mkdir(parents=True, exist_ok=True)
    (CYCLE_DIR / f"round_{cycle['round']:02d}_promotion.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if accepted:
        print(f"\nAccepted. On each site set:\n  $env:BLOOM_MODEL_DIR = \"{out}\"")
    else:
        print("\nRejected: accuracy dropped beyond tolerance; keep the current classifier.")
    return 0 if accepted else 3


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    default_global = str(CYCLE_DIR / "global_adapter")

    s = sub.add_parser("status")
    s.add_argument("--labels", default=str(LABEL_STORE))
    s.add_argument("--site-id", default="site_local")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("init")
    s.add_argument("--base-model", default=str(DEPLOYED_MODEL))
    s.add_argument("--out", default=default_global)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("train")
    s.add_argument("--site-id", required=True)
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--global-adapter", default=default_global)
    s.add_argument("--labels", default=str(LABEL_STORE))
    s.add_argument("--base-model", default=None, help="Override the base recorded in cycle.json")
    s.add_argument("--out-bundle", default=None)
    s.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedprox")
    s.add_argument("--min-labels", type=int, default=MIN_LABELS)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("aggregate")
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--bundles", nargs="+", required=True)
    s.add_argument("--global-adapter", default=default_global)
    s.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedprox")
    s.add_argument("--min-sites", type=int, default=2)
    s.set_defaults(fn=cmd_aggregate)

    s = sub.add_parser("promote")
    s.add_argument("--global-adapter", default=default_global)
    s.add_argument("--heldout-csv", default=str(HELDOUT_CSV))
    s.add_argument("--out", default=None)
    s.add_argument("--tolerance", type=float, default=0.01, help="Allowed accuracy drop")
    s.add_argument("--limit", type=int, default=0, help="Evaluate on first N held-out rows (0 = all)")
    s.set_defaults(fn=cmd_promote)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
