"""I/O helpers for task-separated multitask datasets.

Persisted dataset files stay physically separated by task. A single training/evaluation
split may be combined in memory only after each task file and its manifest fingerprint
has been verified.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

TASKS = ("bloom_rewrite", "qa", "summarization")
SPLITS = ("train", "validation", "test")
MANIFEST_VERSION = "multitask_bloom_rewrite_separate_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
SUMMARIZATION_ELIGIBILITY_FILE = "summarization/summarization_eligibility.json"
_MATH_PLACEHOLDER = re.compile(r"@xmath\d+")


class SummarizationEligibilityError(ValueError):
    """A summarization row is excluded/unaudited, or the eligibility record is missing or altered."""


# EduGuard v3.1 Bloom splits (data/bloom_rewrite_eduguard_v3_1) as an alternative bloom_rewrite source.
EDUGUARD_BLOOM_SOURCE_KIND = "eduguard_bloom_v3_1"
EDUGUARD_SOURCE_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl", "dataset_manifest.json")
_EDUGUARD_PRESERVED_FIELDS = (
    "id", "source_id", "group_id", "split", "source_question", "source_bloom_level",
    "target_bloom_level", "target_rewrite",
)


class ContestedBloomLabelError(ValueError):
    """A Bloom row whose gold label is contested reached a training or evaluation path."""


def assert_no_contested_bloom_rows(rows: list[dict[str, Any]], where: str = "") -> None:
    """Fail closed if any bloom_rewrite row is flagged gold_label_contested."""
    bad = [str(r.get("id")) for r in rows if r.get("task") == "bloom_rewrite" and r.get("gold_label_contested")]
    if bad:
        raise ContestedBloomLabelError(f"{where}: {len(bad)} contested Bloom row(s), e.g. {bad[:3]}")


def verify_eduguard_bloom_rows(
    source_dir: Path,
    bloom_rows_by_split: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Check the bloom_rewrite rows against the EduGuard v3.1 source splits.

    * every contested source row is excluded, and the manifest's exclusion list equals the contested source rows;
    * every remaining source row is present, in the same split, with its IDs, labels and texts unchanged;
    * nothing else was added; sft_text is built from the preserved fields.
    """
    exclusions = manifest.get("bloom_exclusions") or {}
    declared = {str(e["id"]) for e in exclusions.get("excluded_rows", [])}
    contested: set[str] = set()
    checked = 0
    for split in SPLITS:
        source_rows = read_jsonl(source_dir / f"{split}.jsonl")
        if any(r.get("split") != split for r in source_rows):
            raise ValueError(f"EduGuard source {split}.jsonl contains rows from another split.")
        expected = {}
        for r in source_rows:
            if r.get("gold_label_contested"):
                contested.add(str(r["id"]))
            else:
                expected[str(r["id"])] = r
        rows = bloom_rows_by_split[split]
        assert_no_contested_bloom_rows(rows, f"bloom_rewrite/{split}")
        got = {str(r.get("id")): r for r in rows}
        if set(got) != set(expected):
            missing, extra = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
            raise ValueError(
                f"bloom_rewrite/{split} does not match the EduGuard source split: "
                f"missing {missing[:3]} ({len(missing)}), unexpected {extra[:3]} ({len(extra)})."
            )
        for rid, row in got.items():
            src = expected[rid]
            for field in _EDUGUARD_PRESERVED_FIELDS:
                if row.get(field) != src.get(field):
                    raise ValueError(f"bloom_rewrite/{split} row {rid}: {field} differs from the EduGuard source.")
            sft = str(row.get("sft_text") or "")
            if not sft.startswith(str(row.get("prompt_text") or "\0")) or str(row["target_rewrite"]).strip() not in sft:
                raise ValueError(f"bloom_rewrite/{split} row {rid}: sft_text/prompt_text are inconsistent.")
            checked += 1
    if declared != contested:
        raise ValueError(
            "Manifest bloom_exclusions.excluded_rows does not equal the contested rows in the EduGuard source "
            f"({len(declared)} declared, {len(contested)} contested)."
        )
    return {"kind": EDUGUARD_BLOOM_SOURCE_KIND, "rows_checked": checked, "contested_rows_excluded": len(contested)}


# Closed-book academic QA (data/qa_eduguard_v2) as an alternative qa source.
QA_EDUGUARD_SOURCE_KIND = "qa_eduguard_v2"
QA_EDUGUARD_SOURCE_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl", "reserve.jsonl", "dataset_manifest.json")
QA_CLOSED_BOOK = "closed_book"
_QA_EDUGUARD_PRESERVED_FIELDS = (
    "id", "split", "question", "reference_answer", "discipline", "discipline_canonical", "discipline_group",
    "topic", "qa_type", "difficulty", "answer_review",
)


def task_input_text(task: str, row: dict[str, Any]) -> str:
    """The model-visible input used for cross-split exact-leakage checks."""
    if task == "bloom_rewrite":
        return str(row.get("source_question") or "")
    if task == "qa":
        if row.get("qa_format") == QA_CLOSED_BOOK:
            return str(row.get("question") or "")
        return str(row.get("context") or "")
    if task == "summarization":
        return str(row.get("article") or "")
    raise ValueError(f"Unknown task {task!r}")


def verify_eduguard_qa_rows(
    source_dir: Path,
    qa_rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Check closed-book qa rows against data/qa_eduguard_v2.

    * each split holds exactly that source split's rows (reserve rows never appear);
    * IDs, split, question, reference answer and labels are unchanged; the training target equals the reference answer;
    * every held-out row has a model-assisted review verdict of "correct";
    * sft_text starts with prompt_text and ends with the reference answer.
    """
    reserve_ids = {str(r["id"]) for r in read_jsonl(source_dir / "reserve.jsonl")}
    checked = 0
    for split in SPLITS:
        expected = {str(r["id"]): r for r in read_jsonl(source_dir / f"{split}.jsonl")}
        rows = qa_rows_by_split[split]
        got = {str(r.get("id")): r for r in rows}
        if got.keys() & reserve_ids:
            raise ValueError(f"qa/{split} contains reserve (excluded) rows: {sorted(got.keys() & reserve_ids)[:3]}")
        if set(got) != set(expected):
            missing, extra = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
            raise ValueError(
                f"qa/{split} does not match the qa_eduguard_v2 source split: "
                f"missing {missing[:3]} ({len(missing)}), unexpected {extra[:3]} ({len(extra)})."
            )
        for rid, row in got.items():
            src = expected[rid]
            if row.get("qa_format") != QA_CLOSED_BOOK:
                raise ValueError(f"qa/{split} row {rid}: qa_format must be {QA_CLOSED_BOOK!r}.")
            for field in _QA_EDUGUARD_PRESERVED_FIELDS:
                if row.get(field) != src.get(field):
                    raise ValueError(f"qa/{split} row {rid}: {field} differs from the qa_eduguard_v2 source.")
            if row.get("answer") != src["reference_answer"]:
                raise ValueError(f"qa/{split} row {rid}: answer differs from the source reference_answer.")
            if split != "train" and src.get("answer_review") != "correct":
                raise ValueError(f"qa/{split} row {rid}: held-out answer is not reviewed as correct.")
            sft, prompt = str(row.get("sft_text") or ""), str(row.get("prompt_text") or "\0")
            if not sft.startswith(prompt) or not sft.endswith(src["reference_answer"].strip() + "<|im_end|>"):
                raise ValueError(f"qa/{split} row {rid}: sft_text/prompt_text are inconsistent.")
            checked += 1
    return {"kind": QA_EDUGUARD_SOURCE_KIND, "rows_checked": checked, "reserve_rows_excluded": len(reserve_ids)}


# Owner-built university lecture summarization (lecture PDF/slide/notes text -> study summary).
SUM_LECTURE_SOURCE_KIND = "eduguard_lecture_summarization"
SUM_LECTURE = "lecture"
LECTURE_SOURCE_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl")
LECTURE_REQUIRED_FIELDS = ("id", "source_text", "summary")
_LECTURE_PRESERVED_FIELDS = ("id", "source_text", "summary", "department", "course", "topic", "source_type",
                             "review_status")


def enforce_lecture_summarization_rows(
    split: str, rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Load-time guard: every row is a lecture row whose exact content hash is recorded in the manifest."""
    recorded = ((manifest.get("summarization_source") or {}).get("row_sha256") or {}).get(split)
    if not isinstance(recorded, dict):
        raise SummarizationEligibilityError(f"Manifest has no lecture row hashes for summarization/{split}.")
    seen = set()
    for row in rows:
        rid = str(row.get("id") or "")
        if row.get("summarization_format") != SUM_LECTURE:
            raise SummarizationEligibilityError(f"summarization/{split}: row {rid} is not a lecture row.")
        if recorded.get(rid) != _row_sha256(row):
            raise SummarizationEligibilityError(f"summarization/{split}: row {rid} differs from its recorded content.")
        seen.add(rid)
    if seen != set(recorded):
        raise SummarizationEligibilityError(
            f"summarization/{split}: {len(seen)} rows loaded but {len(recorded)} are recorded in the manifest."
        )
    return {"status": "lecture_source_hashes", "split": split, "rows": len(rows)}


def verify_lecture_summarization_rows(
    source_dir: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Check summarization rows against the owner-built lecture source splits.

    * each split holds exactly that source split's rows, with source fields unchanged;
    * article == source_text and abstract == summary (the training input and target);
    * rows carry the required review status when the manifest sets one;
    * sft_text starts with prompt_text and ends with the summary.
    """
    required_status = (manifest.get("summarization_source") or {}).get("required_review_status")
    checked = 0
    for split in SPLITS:
        expected = {str(r["id"]): r for r in read_jsonl(source_dir / f"{split}.jsonl")}
        got = {str(r.get("id")): r for r in rows_by_split[split]}
        if set(got) != set(expected):
            missing, extra = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
            raise ValueError(
                f"summarization/{split} does not match the lecture source split: "
                f"missing {missing[:3]} ({len(missing)}), unexpected {extra[:3]} ({len(extra)})."
            )
        for rid, row in got.items():
            src = expected[rid]
            if row.get("summarization_format") != SUM_LECTURE:
                raise ValueError(f"summarization/{split} row {rid}: summarization_format must be {SUM_LECTURE!r}.")
            for field in _LECTURE_PRESERVED_FIELDS:
                if row.get(field) != src.get(field):
                    raise ValueError(f"summarization/{split} row {rid}: {field} differs from the lecture source.")
            if row.get("article") != src["source_text"] or row.get("abstract") != src["summary"]:
                raise ValueError(f"summarization/{split} row {rid}: article/abstract differ from source_text/summary.")
            if required_status and src.get("review_status") != required_status:
                raise ValueError(f"summarization/{split} row {rid}: review_status is not {required_status!r}.")
            sft, prompt = str(row.get("sft_text") or ""), str(row.get("prompt_text") or "\0")
            if not sft.startswith(prompt) or not sft.endswith(src["summary"].strip() + "<|im_end|>"):
                raise ValueError(f"summarization/{split} row {rid}: sft_text/prompt_text are inconsistent.")
            checked += 1
    return {"kind": SUM_LECTURE_SOURCE_KIND, "rows_checked": checked}


def _row_sha256(row: dict[str, Any]) -> str:
    # Same row hash the summarization eligibility record was written with (sha256 of sorted-key JSON).
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_summarization_eligibility(data_dir: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return the verified eligibility record, or None for a legacy (unaudited) dataset directory."""
    manifest = manifest if manifest is not None else load_manifest(data_dir)
    record = manifest.get("summarization_quality_filter")
    path = data_dir / SUMMARIZATION_ELIGIBILITY_FILE
    if record is None:
        if path.is_file():
            raise SummarizationEligibilityError(
                f"{path} exists but dataset_manifest.json does not record it; rebuild the dataset."
            )
        return None
    if not path.is_file():
        raise SummarizationEligibilityError(f"Manifest requires {SUMMARIZATION_ELIGIBILITY_FILE}, which is missing.")
    if sha256_file(path) != record.get("eligibility_file_sha256"):
        raise SummarizationEligibilityError(f"{SUMMARIZATION_ELIGIBILITY_FILE} does not match the manifest SHA-256.")
    return json.loads(path.read_text(encoding="utf-8"))


def enforce_summarization_eligibility(
    data_dir: Path,
    split: str,
    rows: list[dict[str, Any]],
    *,
    allow_unaudited: bool = False,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless every summarization row is recorded as eligible (exact row hash) for this split.

    Legacy directories without an eligibility record may be read for validation/test (to reproduce earlier
    evaluations) but never for train unless the caller explicitly passes allow_unaudited=True.

    Datasets whose manifest declares a lecture summarization source are checked against the per-row hashes
    recorded in that manifest instead (full source verification runs in validate_separate_dataset).
    """
    manifest = manifest if manifest is not None else load_manifest(data_dir)
    if (manifest.get("summarization_source") or {}).get("kind") == SUM_LECTURE_SOURCE_KIND:
        return enforce_lecture_summarization_rows(split, rows, manifest)
    if any(row.get("summarization_format") == SUM_LECTURE for row in rows):
        raise SummarizationEligibilityError(
            f"summarization/{split}: lecture rows found but the manifest does not declare "
            f"summarization_source.kind={SUM_LECTURE_SOURCE_KIND!r}."
        )
    eligibility = load_summarization_eligibility(data_dir, manifest)
    if eligibility is None:
        if split == "train" and not allow_unaudited:
            raise SummarizationEligibilityError(
                f"{data_dir} has no summarization eligibility record; its summarization train rows are unaudited "
                "(e.g. @xmath placeholders in target abstracts). Use the cleaned dataset "
                "(an audited dataset such as multitask_bloom_rewrite_separate_eduguard_v3_2) "
                "or pass allow_unaudited_summarization=True deliberately."
            )
        return {"status": "unaudited_legacy", "split": split, "rows": len(rows)}
    allowed = {
        r["id"]: r for r in eligibility["rows"] if r["split"] == split and r["status"] == "eligible"
    }
    excluded = {r["id"] for r in eligibility["rows"] if r["split"] == split and r["status"] != "eligible"}
    for row in rows:
        rid = str(row.get("id") or "")
        if rid in excluded:
            raise SummarizationEligibilityError(f"summarization/{split}: row {rid} is excluded ({eligibility['policy_version']}).")
        entry = allowed.get(rid)
        if entry is None:
            raise SummarizationEligibilityError(f"summarization/{split}: row {rid} is not in the eligibility record.")
        if _row_sha256(row) != entry["row_sha256"]:
            raise SummarizationEligibilityError(f"summarization/{split}: row {rid} differs from its audited content.")
        target = str(row.get("sft_text", ""))[len(str(row.get("prompt_text", ""))):]
        if _MATH_PLACEHOLDER.search(str(row.get("abstract", ""))) or _MATH_PLACEHOLDER.search(target):
            raise SummarizationEligibilityError(f"summarization/{split}: row {rid} has an @xmath placeholder in its target.")
    if len({str(r.get("id")) for r in rows}) != len(allowed):
        raise SummarizationEligibilityError(
            f"summarization/{split}: {len(rows)} rows loaded but {len(allowed)} are recorded as eligible."
        )
    return {"status": "enforced", "split": split, "rows": len(rows), "policy_version": eligibility["policy_version"]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line in {path} at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def load_manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "dataset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset_manifest.json at {path}. Run "
            "build_eduguard_v3_2_dataset.py first."
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read dataset manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {path}")
    return manifest


def validate_separate_dataset(
    data_dir: Path,
    repo_root: Path | None = None,
    check_summarization_source: bool = True,
) -> dict[str, Any]:
    """Fail closed unless every task/split file matches the manifest fingerprints.

    check_summarization_source=False skips only the lecture-summarization source re-verification; it is for
    attach_summarization_dataset.py, which uses a base dataset solely for its bloom_rewrite/qa files and replaces
    the summarization task. Training and evaluation loads always use the default (full check)."""
    data_dir = data_dir.resolve()
    repo_root = (repo_root or REPO_ROOT).resolve()
    manifest_path = data_dir / "dataset_manifest.json"
    manifest = load_manifest(data_dir)
    merged_split_files = [
        data_dir / filename
        for filename in ("train.jsonl", "validation.jsonl", "test.jsonl")
        if (data_dir / filename).is_file()
    ]
    if merged_split_files:
        raise ValueError(
            "Task-separated layout forbids root-level merged split files: "
            + ", ".join(str(path) for path in merged_split_files)
        )
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported dataset manifest version {manifest.get('version')!r}; "
            f"expected {MANIFEST_VERSION!r}. Rebuild the task-separated datasets."
        )
    if manifest.get("storage_layout") != "one_directory_per_task":
        raise ValueError("Manifest does not declare the task-separated storage layout.")

    declared_counts = manifest.get("counts")
    hashes = manifest.get("generated_files_sha256")
    if not isinstance(declared_counts, dict) or not isinstance(hashes, dict):
        raise ValueError("Manifest must contain per-split counts and generated file hashes.")

    observed_counts: dict[str, dict[str, int]] = {}
    rows_by_task_split: dict[str, dict[str, list[dict[str, Any]]]] = {task: {} for task in TASKS}
    for split in SPLITS:
        if not isinstance(declared_counts.get(split), dict):
            raise ValueError(f"Manifest has no task counts for {split!r}.")
        observed_counts[split] = {}
        for task in TASKS:
            rel = f"{task}/{split}.jsonl"
            path = data_dir / task / f"{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing task-separated dataset file: {path}")
            expected_hash = hashes.get(rel)
            if not expected_hash:
                raise ValueError(f"Manifest has no SHA-256 for {rel}.")
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"SHA-256 mismatch for {rel}; data may be stale or edited. "
                    "Rebuild the task-separated datasets."
                )
            rows = read_jsonl(path)
            rows_by_task_split[task][split] = rows
            actual_count = len(rows)
            try:
                expected_count = int(declared_counts[split][task])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid manifest count for {split}/{task}.") from exc
            if actual_count != expected_count:
                raise ValueError(
                    f"Count mismatch for {split}/{task}: manifest={expected_count}, "
                    f"file={actual_count}."
                )
            seen_ids: set[str] = set()
            for row_number, row in enumerate(rows, 1):
                if row.get("task") != task:
                    raise ValueError(
                        f"{rel}:{row_number} has task={row.get('task')!r}, expected {task!r}."
                    )
                if row.get("split") != split:
                    raise ValueError(
                        f"{rel}:{row_number} has split={row.get('split')!r}, expected {split!r}."
                    )
                rid = str(row.get("id") or "").strip()
                if not rid:
                    raise ValueError(f"{rel}:{row_number} has an empty id.")
                if rid in seen_ids:
                    raise ValueError(f"{rel}:{row_number} duplicates id {rid!r} within {task}.")
                seen_ids.add(rid)
            observed_counts[split][task] = actual_count

    # IDs must be disjoint across splits for each task.
    for task in TASKS:
        id_sets = {
            split: {str(row.get("id") or "") for row in rows_by_task_split[task][split]}
            for split in SPLITS
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = id_sets[left] & id_sets[right]
            if overlap:
                raise ValueError(
                    f"{task} record-ID leakage between {left} and {right}: {len(overlap)} IDs."
                )

    # Guard against cross-split duplication of Bloom source groups and exact
    # task inputs before a model is allowed to train or be evaluated.
    group_sets: dict[str, set[str]] = {}
    for split in SPLITS:
        group_sets[split] = {
            str(row.get("group_id") or row.get("source_id") or row.get("source_question") or "").strip().lower()
            for row in rows_by_task_split["bloom_rewrite"][split]
        }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise ValueError(
                f"BloomShift source-group leakage between {left} and {right}: {len(overlap)} groups."
            )

    for task in TASKS:
        normalized_by_split: dict[str, set[str]] = {}
        for split in SPLITS:
            normalized_by_split[split] = {
                " ".join(task_input_text(task, row).lower().split())
                for row in rows_by_task_split[task][split]
            }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = normalized_by_split[left] & normalized_by_split[right]
            if overlap:
                raise ValueError(
                    f"{task} exact input leakage between {left} and {right}: {len(overlap)} inputs."
                )

    # Verify BloomShift source and its all-row inclusion attestation have not
    # changed since preprocessing. These are source files, not merged corpora.
    bloom_source = manifest.get("bloom_source", {})
    source_rel = bloom_source.get("path")
    source_hashes = bloom_source.get("files_sha256", {})
    if not source_rel or not isinstance(source_hashes, dict):
        raise ValueError("Manifest is missing canonical BloomShift source fingerprints.")
    source_dir = (repo_root / source_rel).resolve()
    bloom_source_check: dict[str, Any] | None = None
    if bloom_source.get("kind") == EDUGUARD_BLOOM_SOURCE_KIND:
        required_source_files = EDUGUARD_SOURCE_FILES
    elif bloom_source.get("kind") in (None, "bloomshift"):
        required_source_files = (
            "train.json", "validation.json", "test.json", "batch_review_attestation.json"
        )
    else:
        raise ValueError(f"Unknown bloom_source kind {bloom_source.get('kind')!r}.")
    for name in required_source_files:
        path = source_dir / name
        expected = source_hashes.get(name)
        if not path.is_file():
            raise FileNotFoundError(f"Canonical BloomShift source file is missing: {path}")
        if not expected or sha256_file(path) != expected:
            raise ValueError(
                f"Canonical BloomShift source fingerprint mismatch for {path}. "
                "Rebuild the task-separated datasets."
            )
    if bloom_source.get("kind") == EDUGUARD_BLOOM_SOURCE_KIND:
        bloom_source_check = verify_eduguard_bloom_rows(
            source_dir, rows_by_task_split["bloom_rewrite"], manifest
        )

    # QA source: legacy datasets carry extractive (SQuAD-style) rows and no qa_source kind; a closed-book
    # source must be declared, fingerprinted, and row-for-row verified. Mixing formats is refused.
    qa_source = manifest.get("qa_source") or {}
    qa_source_check: dict[str, Any] | None = None
    closed_book_rows = sum(
        row.get("qa_format") == QA_CLOSED_BOOK for split in SPLITS for row in rows_by_task_split["qa"][split]
    )
    if qa_source.get("kind") == QA_EDUGUARD_SOURCE_KIND:
        qa_dir = (repo_root / str(qa_source.get("path") or "")).resolve()
        qa_hashes = qa_source.get("files_sha256") or {}
        for name in QA_EDUGUARD_SOURCE_FILES:
            path = qa_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"qa_eduguard_v2 source file is missing: {path}")
            if not qa_hashes.get(name) or sha256_file(path) != qa_hashes[name]:
                raise ValueError(f"qa_eduguard_v2 source fingerprint mismatch for {path}.")
        qa_source_check = verify_eduguard_qa_rows(qa_dir, rows_by_task_split["qa"])
    elif closed_book_rows:
        raise ValueError(
            f"{closed_book_rows} closed-book qa rows found but the manifest does not declare "
            f"qa_source.kind={QA_EDUGUARD_SOURCE_KIND!r}."
        )

    # Lecture summarization source: fingerprinted source splits, verified row for row.
    sum_source = manifest.get("summarization_source") or {}
    summarization_source_check: dict[str, Any] | None = None
    if check_summarization_source and sum_source.get("kind") == SUM_LECTURE_SOURCE_KIND:
        sum_dir = (repo_root / str(sum_source.get("path") or "")).resolve()
        sum_hashes = sum_source.get("files_sha256") or {}
        for name in LECTURE_SOURCE_FILES:
            path = sum_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Lecture summarization source file is missing: {path}")
            if not sum_hashes.get(name) or sha256_file(path) != sum_hashes[name]:
                raise ValueError(f"Lecture summarization source fingerprint mismatch for {path}.")
        summarization_source_check = verify_lecture_summarization_rows(
            sum_dir, rows_by_task_split["summarization"], manifest
        )

    # Summarization eligibility: enforced for every split when the dataset carries an eligibility record;
    # legacy directories are reported as unaudited (training loads of them are refused in load_split_rows).
    summarization_eligibility = {
        split: enforce_summarization_eligibility(
            data_dir, split, rows_by_task_split["summarization"][split], allow_unaudited=True, manifest=manifest
        )
        for split in SPLITS
    }

    manifest_hash = sha256_file(manifest_path)
    return {
        "summarization_eligibility": summarization_eligibility,
        "layout": "task-separated",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest_version": manifest["version"],
        "data_dir": str(data_dir),
        "counts": observed_counts,
        "generated_files_sha256": hashes,
        "bloom_source": str(source_dir),
        "bloom_source_sha256": source_hashes,
        "bloom_annotation_provenance": bloom_source.get("heldout_annotation_provenance"),
        "bloom_source_check": bloom_source_check,
        "qa_source_check": qa_source_check,
        "summarization_source_check": summarization_source_check,
        "training_data_policy": manifest.get("training_data_policy"),
    }


def load_task_split(
    data_dir: Path,
    task: str,
    split: str,
    *,
    validate: bool = True,
    allow_unaudited_summarization: bool = False,
) -> list[dict[str, Any]]:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}.")
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")
    if validate:
        validate_separate_dataset(data_dir)
    rows = read_jsonl(data_dir / task / f"{split}.jsonl")
    if task == "summarization":
        enforce_summarization_eligibility(data_dir, split, rows, allow_unaudited=allow_unaudited_summarization)
    if task == "bloom_rewrite":
        assert_no_contested_bloom_rows(rows, f"bloom_rewrite/{split}")
    return rows


def load_split_rows(
    data_dir: Path,
    split: str,
    *,
    tasks: tuple[str, ...] | list[str] | None = None,
    validate: bool = True,
    allow_unaudited_summarization: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read task-specific split files and combine rows in memory, never on disk."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")
    if validate:
        meta = validate_separate_dataset(data_dir)
    else:
        manifest_path = data_dir / "dataset_manifest.json"
        manifest = load_manifest(data_dir)
        meta = {
            "layout": "task-separated",
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_version": manifest.get("version"),
            "counts": manifest.get("counts", {}),
            "bloom_annotation_provenance": manifest.get("bloom_source", {}).get(
                "heldout_annotation_provenance"
            ),
        }
    selected_tasks = tuple(tasks) if tasks is not None else TASKS
    unknown = set(selected_tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"Unknown task(s): {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for task in selected_tasks:
        rel = f"{task}/{split}.jsonl"
        path = data_dir / task / f"{split}.jsonl"
        task_rows = read_jsonl(path)
        if task == "summarization":
            meta["summarization_eligibility_loaded"] = enforce_summarization_eligibility(
                data_dir, split, task_rows, allow_unaudited=allow_unaudited_summarization
            )
        if task == "bloom_rewrite":
            assert_no_contested_bloom_rows(task_rows, rel)
        rows.extend(task_rows)
        counts[task] = len(task_rows)
        hashes[rel] = sha256_file(path)
    rows.sort(key=lambda row: (str(row.get("task", "")), str(row.get("id", ""))))
    meta.update(
        {
            "split": split,
            "split_counts_loaded": counts,
            "split_file_sha256": hashes,
            "loaded_tasks": list(selected_tasks),
            "combined_in_memory_only": True,
        }
    )
    return rows, meta


def filter_rows_to_tasks(
    rows: list[dict[str, Any]], tasks: list[str] | tuple[str, ...] | None
) -> list[dict[str, Any]]:
    if tasks is None:
        return list(rows)
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"Unknown task(s): {sorted(unknown)}")
    selected = set(tasks)
    return [row for row in rows if row.get("task") in selected]
