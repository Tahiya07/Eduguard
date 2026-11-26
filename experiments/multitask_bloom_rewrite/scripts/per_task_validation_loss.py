#!/usr/bin/env python
"""Per-task validation loss for every saved checkpoint of a run (forward only).

The Trainer selects checkpoints on one pooled validation loss. This splits it by
task so overfitting of a single task (e.g. the oversampled summarization set)
is visible. Loss = mean over answer tokens, the same objective as training.

Writes <results_dir>/per_task_validation_loss.json and prints a table.

Usage:
  python per_task_validation_loss.py                 # active config
  python per_task_validation_loss.py --config <cfg>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for p in (EXPERIMENT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import dataset_io
    from attention_patches import force_sdpa_repeat_kv
    from loss_masking import tokenize_with_assistant_only_loss
    from paths import ACTIVE_TRAINING_CONFIG, CONFIG_DIR

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIG_DIR / ACTIVE_TRAINING_CONFIG))
    ap.add_argument("--split", default="validation")
    a = ap.parse_args()
    cfg_path = Path(a.config) if Path(a.config).is_absolute() else REPO_ROOT / a.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / cfg["output_dir"]
    results_dir = REPO_ROOT / cfg["results_dir"]
    data_dir = REPO_ROOT / cfg["dataset_dir"]

    force_sdpa_repeat_kv()
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    rows, _ = dataset_io.load_split_rows(data_dir, a.split, validate=False)
    enc = [(r["task"], tokenize_with_assistant_only_loss(tok, r["sft_text"], r["prompt_text"], int(cfg["max_seq_length"]))) for r in rows]

    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    targets = [("base (no adapter)", None)] + [(c.name, c) for c in ckpts]
    best = out_dir / "best_adapter"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
    results = []
    for name, path in targets:
        model = PeftModel.from_pretrained(base, str(path)).eval() if path else base
        sums: dict[str, list[float]] = {}
        with torch.no_grad():
            for task, e in enc:
                ids = torch.tensor([e["input_ids"]], device="cuda")
                labels = torch.tensor([e["labels"]], device="cuda")
                first = int(labels[0].ne(-100).int().argmax())
                start = max(first - 1, 0)
                out = model(input_ids=ids, logits_to_keep=ids.shape[1] - start)
                logits = out.logits[0, :-1].float()
                tgt = labels[0, start + 1:]
                loss = torch.nn.functional.cross_entropy(logits, tgt, ignore_index=-100, reduction="sum")
                n = int(tgt.ne(-100).sum())
                s = sums.setdefault(task, [0.0, 0])
                s[0] += float(loss)
                s[1] += n
        tot = sum(v[0] for v in sums.values()) / sum(v[1] for v in sums.values())
        epoch = None
        state = (path / "trainer_state.json") if path else None
        if state and state.is_file():
            epoch = json.loads(state.read_text(encoding="utf-8")).get("epoch")
        rec = {"checkpoint": name, "epoch": epoch, "pooled": round(tot, 4),
               **{t: round(v[0] / v[1], 4) for t, v in sorted(sums.items())},
               "is_best_adapter": bool(path and best.is_dir() and (best / "adapter_model.safetensors").read_bytes() == (path / "adapter_model.safetensors").read_bytes())}
        results.append(rec)
        print(json.dumps(rec))
        if path:
            base = model.unload()  # remove adapter, keep base weights
    (results_dir / "per_task_validation_loss.json").write_text(json.dumps({"split": a.split, "results": results}, indent=2), encoding="utf-8")
    print(f"{'checkpoint':20s} {'epoch':>5s} {'pooled':>7s} {'bloom':>7s} {'qa':>7s} {'summ':>7s}")
    for r in results:
        print(f"{r['checkpoint']:20s} {str(r['epoch']):>5s} {r['pooled']:7.4f} {r.get('bloom_rewrite', 0):7.4f} {r.get('qa', 0):7.4f} {r.get('summarization', 0):7.4f}"
              + ("  <- best_adapter" if r["is_best_adapter"] else ""))


if __name__ == "__main__":
    main()
