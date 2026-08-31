"""Dataset lock — SHA256 hashes for reproducibility across machines.

Regenerate after intentional dataset changes:
  python experiments/federated/generate_dataset_lock.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.federated.run_integrity import DATASET_FILES, dataset_hashes, file_sha256

LOCK_PATH = Path(__file__).resolve().parent / "dataset_lock.json"


def dataset_stats(repo: Path | None = None) -> dict:
    repo = repo or ROOT
    stats = {}
    for rel in DATASET_FILES:
        p = repo / rel
        if not p.is_file():
            stats[rel] = {"exists": False}
            continue
        try:
            import pandas as pd

            df = pd.read_csv(p)
            label_col = "bloom_level" if "bloom_level" in df.columns else None
            stats[rel] = {
                "exists": True,
                "sha256": file_sha256(p),
                "row_count": int(len(df)),
                "label_distribution": (
                    df[label_col].value_counts().to_dict() if label_col else None
                ),
            }
        except Exception as exc:
            stats[rel] = {"exists": True, "sha256": file_sha256(p), "error": str(exc)}
    return stats


def build_lock(repo: Path | None = None) -> dict:
    repo = repo or ROOT
    hashes = dataset_hashes(repo)
    stats = dataset_stats(repo)
    return {
        "format": "eduguard_dataset_lock_v1",
        "dataset_files": list(DATASET_FILES),
        "sha256": hashes,
        "stats": stats,
    }


def verify_dataset_lock(repo: Path | None = None) -> tuple[bool, list[str]]:
    repo = repo or ROOT
    if not LOCK_PATH.is_file():
        return False, [f"missing dataset lock file: {LOCK_PATH}"]
    expected = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    issues = []
    for rel, exp_hash in (expected.get("sha256") or {}).items():
        actual = file_sha256(repo / rel)
        if actual is None:
            issues.append(f"missing dataset file: {rel}")
        elif actual != exp_hash:
            issues.append(f"dataset hash mismatch for {rel}: expected {exp_hash[:12]}... got {actual[:12]}...")
    return len(issues) == 0, issues


def main() -> int:
    lock = build_lock()
    LOCK_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"Wrote {LOCK_PATH}")
    for rel, h in lock["sha256"].items():
        print(f"  {rel}: {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
