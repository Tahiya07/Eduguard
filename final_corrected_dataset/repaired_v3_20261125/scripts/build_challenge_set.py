"""Held-out Bloom diagnostic challenge set, built ONLY from the v3 TEST split (never used for training or model selection).

Outputs in repaired_v3_20261125/challenge_set/:
  challenge_items.jsonl      one item per test question (58 sources + 290 transformations), with diagnostic tags
  challenge_pairs.jsonl      same-content pairs at adjacent Bloom levels (contrastive items)
  challenge_manifest.json    counts, tag definitions, SHA-256 of the files, and the train/validation disjointness proof

Tags (computed automatically from the train split and the OBE banks; they describe difficulty, not correctness):
  shared_bank_verb          leading verb is listed in two or more level banks (e.g. Determine, Identify, Establish)
  verb_multi_level_in_train leading verb appears at two or more target levels in train
  verb_shortcut_misleads    the train-majority level of the leading verb differs from the gold level
  rare_or_unseen_verb       leading verb occurs fewer than 5 times at the gold level in train
  scenario_first            the question opens with a context sentence before the instruction sentence
  masked_verb_nb_wrong      naive Bayes on the question with its leading verb removed predicts the wrong level
  false_claim_object        deliberately false claim to be judged (Evaluate); never a ground-truth answer
  authored_scenario         the question contains an authored (not source-supplied) scenario
  label_and_generation_only no answer source: valid for label prediction, not for answer scoring
content_anchor_tokens lists the source content words (and all numbers) that the question keeps, for checking that a
rewrite model preserves content.

Run from repaired_v3_20261125:  python -I scripts/build_challenge_set.py
"""
import hashlib, json, os, re, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'scripts'))
import predictability_v3 as P  # noqa: E402  (also refreshes audits/predictability_v3.json)

TR = os.path.join(HERE, 'training_ready')
OUT = os.path.join(HERE, 'challenge_set')
LEVELS = P.LEVELS
ADJ = [('Remember', 'Understand'), ('Understand', 'Apply'), ('Apply', 'Analyze'), ('Understand', 'Analyze'),
       ('Analyze', 'Evaluate'), ('Evaluate', 'Create'), ('Analyze', 'Create')]
STOP = set('the a an of to and in for by with using its their from on at as is be that this or under which whether each when '
           'how why what it are was were has have been into between than then so such these those there any all both one two '
           'three four five specified supplied stated given following'.split())


def jl(p):
    return P.jl(p)


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def content(q):
    return {w for w in re.findall(r"[a-z][a-z\-]+|\d+(?:\.\d+)?", q.lower()) if w not in STOP and (len(w) > 3 or w[0].isdigit())}


train = jl(os.path.join(TR, 'splits', 'train_transformation_records.jsonl'))
val_ex = jl(os.path.join(TR, 'splits', 'validation_examples.jsonl'))
train_ex = jl(os.path.join(TR, 'splits', 'train_examples.jsonl'))
test_ex = jl(os.path.join(TR, 'splits', 'test_examples.jsonl'))
test_tr = {(r['example_id'], r['target_bloom_level']): r for r in jl(os.path.join(TR, 'splits', 'test_transformation_records.jsonl'))}

V = lambda q: P.lead_verb(q)[0]
verb_lv = defaultdict(Counter)
for r in train:
    verb_lv[V(r['target_question'])][r['target_bloom_level']] += 1
bank_levels = defaultdict(set)
for lv, vs in P.BANK.items():
    for v in vs:
        bank_levels[v.lower()].add(lv)
masked = P.nb(train, lambda r: P.mask(r['target_question']))
default_lv = Counter(r['target_bloom_level'] for r in train).most_common(1)[0][0]

REVIEW = json.load(open(os.path.join(HERE, 'work', 'challenge_review.json'), encoding='utf-8'))
CONTESTED = REVIEW['verdicts']['contested_gold_label']
WEAK = REVIEW['verdicts']['weak_distinction_kept']
items, pairs = [], []
for x in test_ex:
    src_c = content(x['source_question'])
    qs = [('source', x['source_bloom_level'], x['source_question'], None)] + \
         [('transformation', lv, q, test_tr[(x['id'], lv)]) for lv, q in x['transformations'].items()]
    for role, lv, q, rec in qs:
        v, si = P.lead_verb(q)
        tags = []
        if len(bank_levels.get(v, ())) > 1: tags.append('shared_bank_verb')
        if len(verb_lv.get(v, ())) > 1: tags.append('verb_multi_level_in_train')
        maj = verb_lv[v].most_common(1)[0][0] if verb_lv.get(v) else default_lv
        if maj != lv: tags.append('verb_shortcut_misleads')
        if verb_lv.get(v, Counter())[lv] < 5: tags.append('rare_or_unseen_verb')
        if si > 0: tags.append('scenario_first')
        if masked({'target_question': q}) != lv: tags.append('masked_verb_nb_wrong')
        if rec is not None:
            if rec['deliberately_false_claim']: tags.append('false_claim_object')
            if rec['content_provenance'] == 'authored_scenario': tags.append('authored_scenario')
        if x['eligibility'] != 'usable': tags.append('label_and_generation_only')
        item = {'item_id': f"CH-{x['id']}-{'source' if role == 'source' else lv}", 'example_id': x['id'], 'group_id': x['group_id'],
                'split_of_origin': 'test', 'role': role, 'discipline': x['discipline'], 'topic': x['topic'],
                'question': q, 'gold_bloom_level': lv, 'leading_verb': v,
                'leading_verb_bank_levels': sorted(bank_levels.get(v, ()), key=LEVELS.index),
                'leading_verb_train_level_counts': dict(verb_lv.get(v, {})),
                'source_question': x['source_question'], 'source_bloom_level': x['source_bloom_level'],
                'content_anchor_tokens': sorted(src_c & content(q)) if role != 'source' else [],
                'eligibility': x['eligibility'], 'diagnostic_tags': tags,
                'pass3_rewritten': bool(rec and rec['pass3_rewritten'])}
        key = f"{x['id']}-{'source' if role == 'source' else lv}"
        item['review_verdict'] = 'contested_gold_label' if key in CONTESTED else 'weak_distinction_noted' if key in WEAK else 'reviewed_ok'
        item['score_this_item'] = key not in CONTESTED
        if rec is not None and rec['deliberately_false_claim']:
            item['claim_under_evaluation'] = rec['claim_under_evaluation']
        items.append(item)
    allq = {x['source_bloom_level']: x['source_question'], **x['transformations']}
    for a, b in ADJ:
        if a in allq and b in allq:
            pairs.append({'pair_id': f"CP-{x['id']}-{a}-{b}", 'example_id': x['id'], 'discipline': x['discipline'],
                          'lower_level': a, 'higher_level': b, 'lower_question': allq[a], 'higher_question': allq[b],
                          'same_leading_verb': V(allq[a]) == V(allq[b]),
                          'score_this_pair': not any(f"{x['id']}-{'source' if lv_ == x['source_bloom_level'] else lv_}" in CONTESTED for lv_ in (a, b)),
                          'task': 'Given both questions on the same content, assign each its Bloom level (or say which demands the higher level).'})

# ---------------------------------------------------------------- disjointness proof
tid = {x['id'] for x in test_ex}
other = {x['id'] for x in train_ex + val_ex}
tg = {x['group_id'] for x in test_ex}
og = {x['group_id'] for x in train_ex + val_ex}
norm = lambda s: re.sub(r'\W+', ' ', s.lower()).strip()
other_text = {norm(q) for x in train_ex + val_ex for q in [x['source_question'], *x['transformations'].values()]}
assert {i['example_id'] for i in items} <= tid and not tid & other and not tg & og
assert not any(norm(i['question']) in other_text for i in items)

os.makedirs(OUT, exist_ok=True)
for name, recs in [('challenge_items.jsonl', items), ('challenge_pairs.jsonl', pairs)]:
    with open(os.path.join(OUT, name), 'w', encoding='utf-8', newline='\n') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
tagc = Counter(t for i in items for t in i['diagnostic_tags'])
man = {
    'purpose': 'Held-out diagnostic for Bloom classification and rewriting: tests whether a model recognises the cognitive '
               'operation from the whole question rather than from a memorised verb or template. EVALUATION ONLY: never train, '
               'tune, early-stop or select checkpoints on it.',
    'source': 'v3 test split only (training_ready/splits/test_*). Built automatically; tags are heuristic and not human-verified.',
    'items': len(items), 'pairs': len(pairs),
    'items_by_role': dict(Counter(i['role'] for i in items)),
    'items_by_gold_level': {lv: sum(i['gold_bloom_level'] == lv for i in items) for lv in LEVELS},
    'items_by_discipline': dict(Counter(i['discipline'] for i in items).most_common()),
    'tag_counts': dict(tagc.most_common()),
    'pairs_by_level_pair': dict(Counter(f"{p['lower_level']}-{p['higher_level']}" for p in pairs)),
    'pairs_with_same_leading_verb': sum(p['same_leading_verb'] for p in pairs),
    'item_review': {'method': REVIEW['method'], 'verdict_counts': dict(Counter(i['review_verdict'] for i in items)),
                    'items_excluded_from_scoring': sorted(CONTESTED), 'pairs_excluded_from_scoring': sum(not p['score_this_pair'] for p in pairs),
                    'limitations': REVIEW['limitations']},
    'recommended_slices': {
        'verb_shortcut_slice': 'items tagged verb_shortcut_misleads or verb_multi_level_in_train',
        'unfamiliar_verb_slice': 'items tagged rare_or_unseen_verb',
        'scenario_first_slice': 'items tagged scenario_first',
        'adjacent_level_pairs': 'challenge_pairs.jsonl',
        'answer_scoring': 'exclude items tagged label_and_generation_only; never score against a false_claim_object claim',
    },
    'disjointness': {'example_ids_in_train_or_validation': 0, 'groups_in_train_or_validation': 0,
                     'question_texts_identical_to_train_or_validation': 0, 'test_example_ids': sorted(tid)},
    'files_sha256': {n: sha(os.path.join(OUT, n)) for n in ('challenge_items.jsonl', 'challenge_pairs.jsonl')},
}
with open(os.path.join(OUT, 'challenge_manifest.json'), 'w', encoding='utf-8', newline='\n') as f:
    json.dump(man, f, ensure_ascii=False, indent=2)
print(json.dumps({k: man[k] for k in ('items', 'pairs', 'items_by_role', 'items_by_gold_level', 'tag_counts', 'pairs_by_level_pair',
                                       'pairs_with_same_leading_verb')}, indent=1))
print('disciplines:', len(man['items_by_discipline']))
