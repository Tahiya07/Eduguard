"""Verb-only and masked-verb Bloom-label predictability (diagnostic only; standard library; read-only).

For each export (v2 baseline and v3 repaired) it fits simple predictors on the TRAIN transformation records and scores
them on VALIDATION + TEST transformation records:

  verb_only            majority level of the leading instruction verb (verb unseen in train -> majority level)
  masked_verb_nb       multinomial naive Bayes on the full question with the leading instruction verb removed
  full_text_nb         multinomial naive Bayes on the full question
  first_word_only      majority level of the first word of the question (scenario-first rows open with context)

It also reports H(level), H(level | verb), H(verb | level) on train and the share of held-out rows whose verb appears
in train at more than one level. These numbers describe the data; they were not used as rewrite targets.

The leading instruction verb is level-agnostic: the first word of the first sentence that starts with a verb from the
union of all six OBE banks (work/obe_verb_bank.json); for rows with no such sentence it is the first word.

Run from repaired_v3_20261125:  python -I scripts/predictability_v3.py
"""
import json, math, os, re, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORTS = {'v2_baseline': os.path.join(os.path.dirname(HERE), 'training_ready'), 'v3_repaired': os.path.join(HERE, 'training_ready')}
OUT = os.path.join(HERE, 'audits', 'predictability_v3.json')
LEVELS = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
BANK = json.load(open(os.path.join(HERE, 'work', 'obe_verb_bank.json'), encoding='utf-8'))
ALLV = {v for vs in BANK.values() for v in vs}
SPLIT_SENT = re.compile(r"(?:(?<=[.!])|(?<=[.!]['\"’”]))\s+(?=[A-Z])")
TOK = re.compile(r"[a-z0-9]+")


def jl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def lead_verb(q):
    sents = [s for s in SPLIT_SENT.split(q.strip()) if s.split()]
    for i, s in enumerate(sents):
        w = s.split()[0].strip(',.:;')
        if w in ALLV:
            return w.lower(), i
    return sents[0].split()[0].strip(',.:;').lower(), 0


def mask(q):
    v, i = lead_verb(q)
    sents = [s for s in SPLIT_SENT.split(q.strip()) if s.split()]
    words = sents[i].split()
    sents[i] = ' '.join(words[1:])
    return ' '.join(sents)


def toks(q):
    return TOK.findall(q.lower())


def H(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c)


def cond_H(pairs):  # H(Y | X) for list of (x, y)
    byx = defaultdict(Counter)
    for x, y in pairs:
        byx[x][y] += 1
    n = len(pairs)
    return sum(sum(c.values()) / n * H(c) for c in byx.values())


def majority(train, key):
    tab = defaultdict(Counter)
    for r in train:
        tab[key(r)][r['target_bloom_level']] += 1
    default = Counter(r['target_bloom_level'] for r in train).most_common(1)[0][0]
    return lambda r: tab[key(r)].most_common(1)[0][0] if key(r) in tab else default


def nb(train, text):
    cnt = {lv: Counter() for lv in LEVELS}
    prior = Counter(r['target_bloom_level'] for r in train)
    for r in train:
        cnt[r['target_bloom_level']].update(toks(text(r)))
    vocab = set().union(*cnt.values())
    tot = {lv: sum(c.values()) for lv, c in cnt.items()}
    n = len(train)

    def pred(r):
        ws = toks(text(r))
        best = None
        for lv in LEVELS:
            if not prior[lv]:
                continue
            s = math.log(prior[lv] / n) + sum(math.log((cnt[lv][w] + 1) / (tot[lv] + len(vocab))) for w in ws)
            if best is None or s > best[0]:
                best = (s, lv)
        return best[1]
    return pred


def score(pred, test):
    ok = Counter(); n = Counter()
    for r in test:
        n[r['target_bloom_level']] += 1
        ok[r['target_bloom_level']] += pred(r) == r['target_bloom_level']
    return {'accuracy': round(sum(ok.values()) / len(test), 4),
            'macro_recall': round(sum(ok[lv] / n[lv] for lv in n) / len(n), 4),
            'recall_by_level': {lv: round(ok[lv] / n[lv], 3) for lv in LEVELS if n[lv]}}


res = {'method': __doc__.split('\n\n')[1], 'exports': {}}
for name, root in EXPORTS.items():
    train = jl(os.path.join(root, 'splits', 'train_transformation_records.jsonl'))
    held = jl(os.path.join(root, 'splits', 'validation_transformation_records.jsonl')) + \
        jl(os.path.join(root, 'splits', 'test_transformation_records.jsonl'))
    V = lambda r: lead_verb(r['target_question'])[0]
    pairs = [(V(r), r['target_bloom_level']) for r in train]
    verb_levels = defaultdict(set)
    for v, lv in pairs:
        verb_levels[v].add(lv)
    lv_count = Counter(lv for _, lv in pairs)
    res['exports'][name] = {
        'n_train': len(train), 'n_heldout': len(held),
        'verb_only': score(majority(train, V), held),
        'masked_verb_nb': score(nb(train, lambda r: mask(r['target_question'])), held),
        'full_text_nb': score(nb(train, lambda r: r['target_question']), held),
        'first_word_only': score(majority(train, lambda r: r['target_question'].split()[0].lower()), held),
        'H_level_bits': round(H(lv_count), 3),
        'H_level_given_verb_bits': round(cond_H(pairs), 3),
        'H_verb_given_level_bits': round(cond_H([(y, x) for x, y in pairs]), 3),
        'distinct_leading_verbs_train': len(verb_levels),
        'leading_verbs_used_at_2plus_levels_train': sorted(v for v, s in verb_levels.items() if len(s) > 1),
        'heldout_rows_whose_verb_is_multi_level_in_train': round(sum(len(verb_levels.get(V(r), ())) > 1 for r in held) / len(held), 3),
        'heldout_rows_with_verb_unseen_in_train': sum(V(r) not in verb_levels for r in held),
    }
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
for name, r in res['exports'].items():
    print(name, {k: (v['accuracy'], v['macro_recall']) if isinstance(v, dict) and 'accuracy' in v else v
                 for k, v in r.items() if k not in ('leading_verbs_used_at_2plus_levels_train',)})
    print('   multi-level verbs:', r['leading_verbs_used_at_2plus_levels_train'])
