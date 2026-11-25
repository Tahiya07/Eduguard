"""Safe loaders for the EduGuard training-ready exports (standard library only).

Any script that converts these exports for training MUST load records through this module.

* Answer-dependent tasks (answer generation, solution checking, answer-based reward models):
  use ``load_answer_generation_records(split)``. It reads ``answer_generation/{split}_transformation_records.jsonl``
  and raises ``LabelOnlyRecordError`` unless every record passes ``assert_answer_eligible``.
* Converting general split files for an answer task: call ``select_answer_eligible(records)``. It filters explicitly on
  eligibility and on the usable-ID allow-list. Never rely on a missing answer field to keep label-only data out.
* Rewriting and Bloom-classification tasks may use both eligibility classes: ``load_general_records(split, kind)``.

Checks applied to every answer-generation record:
  1. ``eligibility == 'usable'``
  2. ``answer_source_available is True``
  3. ``example_id`` is in the allow-list of usable example IDs (from ``usable_examples.jsonl``)
  4. ``example_id`` is not in the deny-list of label-only example IDs (from ``label_and_generation_only_examples.jsonl``)
  5. the file's SHA-256 matches ``answer_generation/answer_generation_manifest.json`` (loader only)
A record that relabels a label-only example as "usable" still fails checks 3 and 4.
"""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../training_ready
SPLITS = ('train', 'validation', 'test')
GENERAL_KINDS = ('examples', 'transformation_records', 'bloom_classification_records')


class LabelOnlyRecordError(ValueError):
    """Raised when a record that is not answer-eligible reaches an answer-dependent path."""


def _read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def eligibility_lists(root=ROOT):
    """Return (allowed usable IDs, denied label-only IDs), read from the two example exports."""
    allowed = {r['id'] for r in _read_jsonl(os.path.join(root, 'usable_examples.jsonl')) if r['eligibility'] == 'usable'}
    denied = {r['id'] for r in _read_jsonl(os.path.join(root, 'label_and_generation_only_examples.jsonl'))}
    if allowed & denied:
        raise LabelOnlyRecordError(f'IDs listed as both usable and label-only: {sorted(allowed & denied)}')
    return allowed, denied


def _example_id(record):
    return record.get('example_id', record.get('id'))


def why_not_answer_eligible(record, allowed, denied):
    """Return the reasons a record may not be used for an answer task (empty list = eligible)."""
    reasons = []
    eid = _example_id(record)
    if record.get('eligibility') != 'usable':
        reasons.append(f"eligibility is {record.get('eligibility')!r}, not 'usable'")
    if record.get('answer_source_available') is not True:
        reasons.append('answer_source_available is not true')
    if eid in denied:
        reasons.append(f'example {eid} is label_and_generation_only')
    if eid not in allowed:
        reasons.append(f'example {eid} is not in the usable allow-list')
    return reasons


def assert_answer_eligible(records, root=ROOT, allowed=None, denied=None):
    """Raise LabelOnlyRecordError if any record is not answer-eligible; otherwise return the records unchanged."""
    if allowed is None or denied is None:
        allowed, denied = eligibility_lists(root)
    bad = []
    for r in records:
        reasons = why_not_answer_eligible(r, allowed, denied)
        if reasons:
            bad.append((r.get('record_id', _example_id(r)), reasons))
    if bad:
        raise LabelOnlyRecordError(f'{len(bad)} record(s) are not answer-eligible; first: {bad[0]}')
    return records


def select_answer_eligible(records, root=ROOT):
    """Explicitly filter records to answer-eligible ones. Returns (kept, rejected_record_ids)."""
    allowed, denied = eligibility_lists(root)
    kept, rejected = [], []
    for r in records:
        (rejected.append(r.get('record_id', _example_id(r))) if why_not_answer_eligible(r, allowed, denied) else kept.append(r))
    return assert_answer_eligible(kept, allowed=allowed, denied=denied), rejected


def load_answer_generation_records(split, root=ROOT):
    """Load the strict usable-only answer-generation records for one split, after verifying them."""
    if split not in SPLITS:
        raise ValueError(f'unknown split {split!r}')
    with open(os.path.join(root, 'answer_generation', 'answer_generation_manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    entry = manifest['splits'][split]
    path = os.path.join(root, entry['file'])
    if _sha256(path) != entry['sha256']:
        raise LabelOnlyRecordError(f'{entry["file"]} does not match the SHA-256 in answer_generation_manifest.json; rebuild the export')
    records = _read_jsonl(path)
    allowed, denied = eligibility_lists(root)
    if set(manifest['allowed_example_ids']) != allowed or set(manifest['denied_example_ids_label_and_generation_only']) != denied:
        raise LabelOnlyRecordError('answer_generation_manifest.json ID lists disagree with the example exports')
    assert_answer_eligible(records, allowed=allowed, denied=denied)
    if len(records) != entry['transformation_records']:
        raise LabelOnlyRecordError(f'{split}: {len(records)} records, manifest says {entry["transformation_records"]}')
    if any(r.get('split') != split for r in records):
        raise LabelOnlyRecordError(f'{split}: a record carries a different split label')
    return records


def load_general_records(split, kind='transformation_records', root=ROOT):
    """Load a general split file (both eligibility classes). Not for answer-dependent tasks."""
    if split not in SPLITS or kind not in GENERAL_KINDS:
        raise ValueError((split, kind))
    return _read_jsonl(os.path.join(root, 'splits', f'{split}_{kind}.jsonl'))
