"""Shortcut baselines on the held-out challenge set (diagnostic; standard library; no model training beyond count tables).

Fits the same simple predictors as predictability_v3.py on the TRAIN split (transformation records plus source
questions) and scores every challenge item and pair. A slice that a shortcut solves easily is not a real test of
cognitive-operation recognition; a slice that defeats every shortcut is where a trained model must be checked.

Writes audits/challenge_baselines.json. Run from repaired_v3_20261125:  python -I scripts/evaluate_challenge_set.py
"""
import json, os, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'scripts'))
import predictability_v3 as P  # noqa: E402

TR = os.path.join(HERE, 'training_ready')
C = os.path.join(HERE, 'challenge_set')
LEVELS = P.LEVELS
train = P.jl(os.path.join(TR, 'splits', 'train_transformation_records.jsonl'))
train += [{'target_question': x['source_question'], 'target_bloom_level': x['source_bloom_level']}
          for x in P.jl(os.path.join(TR, 'splits', 'train_examples.jsonl'))]
items = [i for i in P.jl(os.path.join(C, 'challenge_items.jsonl')) if i['score_this_item']]   # contested gold labels excluded
pairs = [p for p in P.jl(os.path.join(C, 'challenge_pairs.jsonl')) if p['score_this_pair']]

V = lambda r: P.lead_verb(r['target_question'])[0]
preds = {
    'verb_only': P.majority(train, V),
    'first_word_only': P.majority(train, lambda r: r['target_question'].split()[0].lower()),
    'masked_verb_nb': P.nb(train, lambda r: P.mask(r['target_question'])),
    'full_text_nb': P.nb(train, lambda r: r['target_question']),
}
asq = lambda q: {'target_question': q}
slices = {'all': items}
for t in sorted({t for i in items for t in i['diagnostic_tags']}):
    slices[t] = [i for i in items if t in i['diagnostic_tags']]
slices['interrogative'] = [i for i in items if '?' in i['question']]
slices['no_tag_easy'] = [i for i in items if not set(i['diagnostic_tags']) - {'label_and_generation_only', 'authored_scenario'}]

res = {'n_train': len(train), 'slices': {}, 'pairs': {}}
for name, its in slices.items():
    res['slices'][name] = {'n': len(its)}
    for p, f in preds.items():
        res['slices'][name][p] = round(sum(f(asq(i['question'])) == i['gold_bloom_level'] for i in its) / max(1, len(its)), 3)
for p, f in preds.items():
    both = order = 0
    by = defaultdict(lambda: [0, 0])
    for pr in pairs:
        a, b = f(asq(pr['lower_question'])), f(asq(pr['higher_question']))
        ok = a == pr['lower_level'] and b == pr['higher_level']
        both += ok
        ordered = LEVELS.index(a) < LEVELS.index(b)
        order += ordered
        k = f"{pr['lower_level']}-{pr['higher_level']}"
        by[k][0] += ok; by[k][1] += 1
    res['pairs'][p] = {'both_correct': round(both / len(pairs), 3), 'order_correct': round(order / len(pairs), 3),
                       'both_correct_by_pair': {k: round(v[0] / v[1], 3) for k, v in by.items()}}
with open(os.path.join(HERE, 'audits', 'challenge_baselines.json'), 'w', encoding='utf-8', newline='\n') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"{'slice':32s} {'n':>4s} " + ' '.join(f'{p:>16s}' for p in preds))
for name, r in res['slices'].items():
    print(f"{name:32s} {r['n']:4d} " + ' '.join(f'{r[p]:16.3f}' for p in preds))
for p, r in res['pairs'].items():
    print('pairs', p, r['both_correct'], r['order_correct'], r['both_correct_by_pair'])
