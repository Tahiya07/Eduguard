"""Copy the v3.1 Bloom rewrite splits into data/bloom_rewrite_eduguard_v3_1/ (train/validation/test JSONL + manifest).

One row per (source question -> target level) transformation, using the field names of the existing
data/*/bloom_rewrite rows where they apply. Reads repaired_v3_20261125/training_ready only; refuses to overwrite.
Run from repaired_v3_20261125:  python -I scripts/export_to_data_dir.py
"""
import hashlib, json, os, sys

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(HERE))
TR = os.path.join(HERE, 'training_ready')
OUT = os.path.join(REPO, 'data', 'bloom_rewrite_eduguard_v3_1')
SPLITS = ('train', 'validation', 'test')


def jl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


if os.path.exists(OUT):
    sys.exit(f'{OUT} already exists; not overwriting')
review = json.load(open(os.path.join(HERE, 'work', 'challenge_review.json'), encoding='utf-8'))
contested = review['verdicts']['contested_gold_label']
os.makedirs(OUT)
counts, files = {}, {}
for s in SPLITS:
    rows = []
    for r in jl(os.path.join(TR, 'splits', f'{s}_transformation_records.jsonl')):
        key_src, key_tgt = f"{r['example_id']}-source", f"{r['example_id']}-{r['target_bloom_level']}"
        row = {
            'task': 'bloom_rewrite', 'split': s,
            'id': f"eduguard_v3_1_{r['record_id']}", 'source_id': r['example_id'], 'group_id': r['group_id'],
            'discipline': r['discipline'], 'topic': r['topic'],
            'source_question': r['source_question'], 'source_bloom_level': r['source_bloom_level'],
            'target_bloom_level': r['target_bloom_level'], 'target_rewrite': r['target_question'],
            'transformation_type': f"{r['source_bloom_level']}->{r['target_bloom_level']}",
            'eligibility': r['eligibility'], 'answer_source_available': r['answer_source_available'],
            'content_provenance': r['content_provenance'],
            'deliberately_false_claim': r['deliberately_false_claim'],
            'pass3_rewritten': r['pass3_rewritten'],
            'gold_label_contested': key_src in contested or key_tgt in contested,
            'review_status': 'model_assisted_not_human_verified',
            'independent_review_verified': False,
        }
        if r['deliberately_false_claim']:
            row['claim_under_evaluation'] = r['claim_under_evaluation']
        rows.append(row)
    p = os.path.join(OUT, f'{s}.jsonl')
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    counts[s] = {'rows': len(rows), 'examples': len({x['source_id'] for x in rows}), 'groups': len({x['group_id'] for x in rows}),
                 'usable_rows': sum(x['eligibility'] == 'usable' for x in rows),
                 'target_level_counts': {lv: sum(x['target_bloom_level'] == lv for x in rows)
                                         for lv in ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']}}
    files[f'{s}.jsonl'] = sha(p)

ids = {s: {x['source_id'] for x in jl(os.path.join(OUT, f'{s}.jsonl'))} for s in SPLITS}
grp = {s: {x['group_id'] for x in jl(os.path.join(OUT, f'{s}.jsonl'))} for s in SPLITS}
for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
    assert not ids[a] & ids[b] and not grp[a] & grp[b], (a, b)
ir = json.load(open(os.path.join(TR, 'integrity_report.json'), encoding='utf-8'))
manifest = {
    'version': 'bloom_rewrite_eduguard_v3_1', 'task': 'bloom_rewrite', 'date': '2026-11-25',
    'status': 'finalized candidate; not yet accepted (see README)',
    'source': {'path': 'final_corrected_dataset/repaired_v3_20261125/training_ready',
               'split_manifest_sha256': sha(os.path.join(TR, 'splits', 'split_manifest.json')),
               'master_sha256': ir['v3_master_sha256'],
               'authoritative_master_sha256': ir['authoritative_master_files_sha256']},
    'splits_pinned_to': 'final_corrected_dataset/training_ready/tests/reference_split_ids.json',
    'counts': counts, 'files_sha256': files,
    'leakage': {'example_or_group_overlap_across_splits': 0},
    'review': {'corrections': 'model-assisted (Claude); not human-verified', 'validation': 'automated',
               'contested_gold_label_rows': sum(x['gold_label_contested'] for s in SPLITS for x in jl(os.path.join(OUT, f'{s}.jsonl')))},
    'challenge_set': 'final_corrected_dataset/repaired_v3_20261125/challenge_set (built from test; evaluation only, not copied here)',
    'not_included': ['sft_text (prompt formatting is applied by the training pipeline)', 'qa and summarization tasks'],
}
with open(os.path.join(OUT, 'dataset_manifest.json'), 'w', encoding='utf-8', newline='\n') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(json.dumps({'out': OUT, 'counts': {s: c['rows'] for s, c in counts.items()}, 'contested': manifest['review']['contested_gold_label_rows']}, indent=1))
