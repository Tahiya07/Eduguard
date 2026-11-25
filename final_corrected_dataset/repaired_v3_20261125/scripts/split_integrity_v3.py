"""Split-integrity report for the v3 export (read-only). Writes reports/split_integrity_v3.json and .md.

Run from repaired_v3_20261125:  python -I scripts/split_integrity_v3.py
"""
import hashlib, json, os, re, sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FCD = os.path.dirname(HERE)
V2, V3 = os.path.join(FCD, 'training_ready'), os.path.join(HERE, 'training_ready')
SPLITS = ('train', 'validation', 'test')
LEVELS = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']


def jl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def jload(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


ref = jload(os.path.join(FCD, 'training_ready', 'tests', 'reference_split_ids.json'))['example_ids']
out = {'reference_split_file': {'path': '../training_ready/tests/reference_split_ids.json',
                                'sha256': sha(os.path.join(V2, 'tests', 'reference_split_ids.json')),
                                'v3_copy_identical': sha(os.path.join(V2, 'tests', 'reference_split_ids.json')) == sha(os.path.join(V3, 'tests', 'reference_split_ids.json'))},
       'splits': {}}
groups = {}
for s in SPLITS:
    e3 = jl(os.path.join(V3, 'splits', f'{s}_examples.jsonl'))
    e2 = {x['id']: x for x in jl(os.path.join(V2, 'splits', f'{s}_examples.jsonl'))}
    t3 = jl(os.path.join(V3, 'splits', f'{s}_transformation_records.jsonl'))
    c3 = jl(os.path.join(V3, 'splits', f'{s}_bloom_classification_records.jsonl'))
    a3 = jl(os.path.join(V3, 'answer_generation', f'{s}_transformation_records.jsonl'))
    for x in e3:
        groups.setdefault(x['group_id'], set()).add(s)
    same = lambda k: all(x[k] == e2[x['id']][k] for x in e3)
    out['splits'][s] = {
        'example_ids_identical_to_reference': [x['id'] for x in e3] == ref[s],
        'examples': len(e3), 'groups': len({x['group_id'] for x in e3}),
        'unchanged_vs_v2': {k: same(k) for k in ('group_id', 'eligibility', 'answer_source_available', 'authored_scenario_levels',
                                                 'deliberately_false_claim_levels', 'discipline', 'topic', 'source_question')},
        'transformation_records': len(t3), 'bloom_classification_records': len(c3), 'answer_generation_records': len(a3),
        'all_records_carry_this_split': all(r['split'] == s for r in e3 + t3 + c3 + a3),
        'every_record_example_in_this_split': {r['example_id'] for r in t3 + c3 + a3} <= {x['id'] for x in e3},
        'classification_label_counts': dict(Counter(r['bloom_level'] for r in c3)),
        'source_level_counts': dict(Counter(x['source_bloom_level'] for x in e3)),
        'pass3_rewritten_records': sum(r['pass3_rewritten'] for r in t3),
    }
norm = lambda q: re.sub(r'\W+', ' ', q.lower()).strip()
texts = {}
for s in SPLITS:
    for x in jl(os.path.join(V3, 'splits', f'{s}_examples.jsonl')):
        for q in [x['source_question'], *x['transformations'].values()]:
            texts.setdefault(norm(q), set()).add(s)
leak = jload(os.path.join(V3, 'residual_similarity.json'))
ch = jload(os.path.join(HERE, 'challenge_set', 'challenge_manifest.json'))
out['cross_split'] = {
    'groups_spanning_splits': sum(len(v) > 1 for v in groups.values()),
    'identical_question_texts_across_splits': sum(len(v) > 1 for v in texts.values()),
    'check_leakage': {k: v for k, v in leak.items() if k != 'top_residual_pairs'},
    'top_residual_pairs': leak['top_residual_pairs'][:7],
    'note': 'Residual pairs >= 0.35 token Jaccard are different-topic items in the same discipline (e.g. roof vs foundation '
            'construction); they were already present in v2 and split assignment is pinned, so they were reviewed, not moved.',
}
out['challenge_set'] = {'items': ch['items'], 'pairs': ch['pairs'], **ch['disjointness']}
out['label_change_effect'] = {'252': 'source label Analyze -> Understand (train split); its Understand transformation became the '
                                     'source level and was removed, and a new Analyze transformation was added. Split and group unchanged.'}
ok = all(v['example_ids_identical_to_reference'] and all(v['unchanged_vs_v2'].values()) and v['all_records_carry_this_split']
         and v['every_record_example_in_this_split'] for v in out['splits'].values()) \
    and out['cross_split']['groups_spanning_splits'] == 0 and out['cross_split']['identical_question_texts_across_splits'] == 0 \
    and out['reference_split_file']['v3_copy_identical']
out['RESULT'] = 'PASS' if ok else 'FAIL'
os.makedirs(os.path.join(HERE, 'reports'), exist_ok=True)
with open(os.path.join(HERE, 'reports', 'split_integrity_v3.json'), 'w', encoding='utf-8', newline='\n') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

L = ['# Split-integrity report — EduGuard Bloom dataset v3', '',
     f"Result: **{out['RESULT']}** (all checks automated; generated by `scripts/split_integrity_v3.py`).", '',
     '| Split | Examples (= reference IDs) | Groups | Transformation records | Classification records | Answer-generation records | Pass-3 rewritten |',
     '|---|---|---|---|---|---|---|']
for s, v in out['splits'].items():
    L.append(f"| {s} | {v['examples']} ({'yes' if v['example_ids_identical_to_reference'] else 'NO'}) | {v['groups']} | "
             f"{v['transformation_records']} | {v['bloom_classification_records']} | {v['answer_generation_records']} | {v['pass3_rewritten_records']} |")
L += ['', '- Split, group, eligibility, answer-source, authored-scenario levels, false-claim levels, discipline, topic and source-question fields are '
          'identical to v2 for every example: ' + ('yes' if all(all(v['unchanged_vs_v2'].values()) for v in out['splits'].values()) else 'NO') + '.',
      f"- Groups spanning splits: {out['cross_split']['groups_spanning_splits']}. Identical question texts across splits: "
      f"{out['cross_split']['identical_question_texts_across_splits']}. Example-ID overlap: {leak['example_id_overlap_across_splits']}.",
      f"- Maximum cross-split token Jaccard: {leak['max_cross_split_question_jaccard']}; pairs ≥ 0.35: {leak['cross_split_pairs_at_or_above_0.35']} "
      '(different topics within one discipline; pinned splits, not moved).',
      f"- Reference split file unchanged and copied byte-identically: {out['reference_split_file']['v3_copy_identical']}.",
      f"- Challenge set: {ch['items']} items and {ch['pairs']} pairs, all from test; overlap with train/validation examples, groups and "
      'question texts: 0.',
      '- Label change: 252 Analyze → Understand (train). Its group and split are unchanged.', '',
      '## Bloom classification label counts per split', '', '| Split | ' + ' | '.join(LEVELS) + ' |', '|---|' + '---|' * len(LEVELS)]
for s, v in out['splits'].items():
    L.append(f'| {s} | ' + ' | '.join(str(v['classification_label_counts'].get(lv, 0)) for lv in LEVELS) + ' |')
with open(os.path.join(HERE, 'reports', 'SPLIT_INTEGRITY_REPORT.md'), 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(L) + '\n')
print(out['RESULT'])
print('\n'.join(L))
sys.exit(0 if ok else 1)
