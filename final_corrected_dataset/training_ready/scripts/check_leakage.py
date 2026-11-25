"""Independent leakage check over the produced splits (reads splits/*.jsonl only).

Reports: example-ID overlap, exact question-text overlap, group overlap, and the most similar
cross-split question pairs (token Jaccard), written to residual_similarity.json.
Run:  python -I training_ready/scripts/check_leakage.py
"""
import json, os, re
from itertools import combinations

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP = set('the a an of to and in for by with using its their from on at as is be that this or under which whether each when stated supplied '
           'specified given following explain determine calculate analyze analyse examine assess evaluate verify identify define state design '
           'develop construct apply'.split())
def toks(s): return set(w for w in re.findall(r'[a-z0-9.]+', s.lower()) if w not in STOP and len(w) > 1)
def norm(s): return re.sub(r'\W+', ' ', s.lower()).strip()

ex = []
for s in ['train', 'validation', 'test']:
    with open(os.path.join(OUT, 'splits', f'{s}_examples.jsonl'), encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            qs = [x['source_question']] + list(x['transformations'].values())
            ex.append({'split': s, 'id': x['id'], 'group': x['group_id'], 'topic': x['topic'] or x['discipline'], 'q': qs, 't': [toks(q) for q in qs]})
ids = [e['id'] for e in ex]
id_overlap = len(ids) - len(set(ids))
group_splits = {}
for e in ex:
    group_splits.setdefault(e['group'], set()).add(e['split'])
group_overlap = [g for g, ss in group_splits.items() if len(ss) > 1]
text_split = {}
for e in ex:
    for q in e['q']:
        text_split.setdefault(norm(q), set()).add(e['split'])
text_overlap = sum(1 for ss in text_split.values() if len(ss) > 1)
pairs = []
for a, b in combinations(ex, 2):
    if a['split'] == b['split']:
        continue
    j = max(len(p & q) / max(1, len(p | q)) for p in a['t'] for q in b['t'])
    pairs.append((round(j, 3), a['id'], a['split'], b['id'], b['split'], a['topic'], b['topic']))
pairs.sort(reverse=True)
res = {'example_id_overlap_across_splits': id_overlap, 'groups_spanning_splits': len(group_overlap),
       'identical_question_texts_across_splits': text_overlap,
       'max_cross_split_question_jaccard': pairs[0][0] if pairs else 0.0,
       'cross_split_pairs_at_or_above_0.35': sum(1 for p in pairs if p[0] >= 0.35),
       'top_residual_pairs': [dict(zip(['jaccard', 'id_a', 'split_a', 'id_b', 'split_b', 'topic_a', 'topic_b'], p)) for p in pairs[:20]]}
json.dump(res, open(os.path.join(OUT, 'residual_similarity.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in res.items() if k != 'top_residual_pairs'}, indent=1))
for p in res['top_residual_pairs'][:8]:
    print(p)
