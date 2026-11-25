"""Action-verb diversity, cross-level distribution, template and lexical-shortcut audit (read-only, stdlib only).

Reads the frozen training-ready split files (never the master directly for writing; nothing is modified) and writes
CSV/JSON tables next to this script's parent folder. Run from final_corrected_dataset/:

    python -I scripts/audit_verbs_v3.py <training_ready dir> <output dir> [--original-lexicon]
"""
import sys
import csv, json, math, os, re
from collections import Counter, defaultdict
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
# usage: audit_verbs_v3.py <training_ready dir> <output dir> [--original-lexicon]
TR = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2]); os.makedirs(OUT, exist_ok=True)
EXTENDED = '--original-lexicon' not in sys.argv
LEVELS = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
SPLITS = ['train', 'validation', 'test']

# ---------------------------------------------------------------- verb lexicon -> cognitive-operation family
# Families group verbs that ask for the same cognitive operation; distinct operations stay distinct.
FAMILY = {
    'define': 'define',
    'state': 'state_recall', 'recall': 'state_recall', 'name': 'state_recall', 'expand': 'state_recall', 'write': 'state_recall',
    'list': 'list', 'enumerate': 'list',
    'identify': 'identify', 'recognize': 'identify', 'locate': 'identify', 'label': 'identify', 'specify': 'identify',
    'explain': 'explain', 'clarify': 'explain',
    'describe': 'describe', 'illustrate': 'describe', 'outline': 'describe', 'discuss': 'describe',
    'interpret': 'interpret', 'paraphrase': 'interpret', 'summarize': 'summarize', 'summarise': 'summarize',
    'classify': 'classify', 'categorize': 'classify', 'categorise': 'classify', 'assign': 'classify', 'sort': 'classify',
    'match': 'classify', 'group': 'classify',
    'calculate': 'compute', 'compute': 'compute', 'solve': 'compute', 'find': 'compute', 'evaluate_value': 'compute',
    'determine': 'compute',
    'estimate': 'predict', 'predict': 'predict', 'forecast': 'predict',
    'apply': 'apply_use', 'use': 'apply_use', 'implement': 'apply_use', 'execute': 'apply_use', 'perform': 'apply_use',
    'demonstrate': 'apply_use', 'convert': 'apply_use', 'prepare': 'apply_use', 'draw': 'apply_use', 'sketch': 'apply_use',
    'plot': 'apply_use', 'complete': 'apply_use', 'organize': 'organize', 'arrange': 'organize', 'schedule': 'organize',
    'trace': 'trace', 'simulate': 'trace', 'map': 'trace',
    'compare': 'compare', 'contrast': 'compare', 'differentiate': 'differentiate', 'distinguish': 'differentiate',
    'examine': 'examine', 'analyze': 'examine', 'analyse': 'examine', 'investigate': 'examine', 'inspect': 'examine',
    'explore': 'examine', 'decompose': 'decompose', 'separate': 'decompose', 'dissect': 'decompose', 'break': 'decompose',
    'diagnose': 'diagnose', 'debug': 'diagnose', 'troubleshoot': 'diagnose', 'detect': 'diagnose',
    'infer': 'infer', 'deduce': 'infer', 'derive': 'derive', 'prove': 'derive', 'show': 'derive',
    'verify': 'verify', 'check': 'verify', 'validate': 'verify', 'test': 'verify', 'confirm': 'verify', 'determine_whether': 'verify',
    'assess': 'assess_judge', 'evaluate': 'assess_judge', 'appraise': 'assess_judge', 'judge': 'assess_judge',
    'rate': 'assess_judge', 'weigh': 'assess_judge', 'review': 'assess_judge', 'audit': 'assess_judge',
    'justify': 'justify', 'defend': 'justify', 'argue': 'justify', 'support': 'justify', 'refute': 'justify',
    'critique': 'critique', 'criticize': 'critique', 'criticise': 'critique', 'challenge': 'critique',
    'recommend': 'decide', 'select': 'decide', 'choose': 'decide', 'decide': 'decide', 'prioritize': 'decide', 'rank': 'decide',
    'design': 'design', 'redesign': 'design',
    'develop': 'develop', 'devise': 'develop', 'plan': 'develop',
    'construct': 'construct', 'build': 'construct', 'create': 'construct', 'produce': 'construct', 'generate': 'construct',
    'assemble': 'construct', 'model': 'construct',
    'formulate': 'formulate', 'propose': 'formulate', 'hypothesize': 'formulate', 'conjecture': 'formulate',
    'compose': 'compose', 'draft': 'compose', 'rewrite': 'compose', 'revise': 'compose', 'express': 'compose',
    'synthesize': 'synthesize', 'integrate': 'synthesize', 'combine': 'synthesize',
}
# Families -> the cognitive operations named in the audit brief (coarse; a primary verb never captures the whole task).
OPERATION = {
    'define': 'retrieval', 'state_recall': 'retrieval', 'list': 'retrieval', 'identify': 'retrieval/recognition',
    'explain': 'verbal explanation', 'describe': 'verbal explanation', 'summarize': 'verbal explanation',
    'interpret': 'interpretation', 'classify': 'classification', 'compute': 'computation', 'predict': 'inference/prediction',
    'apply_use': 'application', 'organize': 'application', 'trace': 'application (procedure tracing)',
    'compare': 'comparison', 'differentiate': 'comparison', 'examine': 'decomposition/relationship analysis',
    'decompose': 'decomposition/relationship analysis', 'diagnose': 'inference (diagnosis)', 'infer': 'inference',
    'derive': 'inference (derivation)', 'verify': 'justification/verification', 'assess_judge': 'judgement',
    'justify': 'justification', 'critique': 'criticism', 'decide': 'decision-making', 'design': 'design',
    'develop': 'construction/planning', 'construct': 'construction', 'formulate': 'formulation', 'compose': 'construction (writing)',
    'synthesize': 'synthesis',
}
# Extension (pass 3): verbs introduced by the repair that the original lexicon did not know. Applied identically to the
# v2 baseline and to v3 when EXTENDED is set, so before/after numbers are measured with the same instrument.
EXT_FAMILY = {
    'indicate': 'state_recall', 'reproduce': 'state_recall', 'cite': 'state_recall', 'quote': 'state_recall',
    'retrieve': 'state_recall', 'record': 'state_recall', 'report': 'state_recall', 'tabulate': 'list',
    'characterize': 'describe', 'characterise': 'describe', 'represent': 'describe',
    'relate': 'infer', 'correlate': 'infer', 'attribute': 'infer',
    'probe': 'examine', 'scrutinize': 'examine', 'uncover': 'examine', 'structure': 'decompose', 'subdivide': 'decompose',
    'categorize': 'classify', 'substantiate': 'justify', 'critically': 'critique', 'establish': 'verify',
    'architect': 'design', 'engineer': 'design', 'adapt': 'develop', 'elaborate': 'develop', 'author': 'compose',
    'carry': 'apply_use', 'modify': 'apply_use', 'operate': 'apply_use', 'practice': 'apply_use', 'manipulate': 'apply_use',
    'run': 'apply_use', 'obtain': 'compute', 'measure': 'compute',
}
if EXTENDED:
    for _k, _v in EXT_FAMILY.items():
        FAMILY.setdefault(_k, _v)
LEX = set(FAMILY) - {'evaluate_value', 'determine_whether'}
LEADING = re.compile(r'^(given|using|for|in|with|if|when|after|based on|considering|to|from|during|under|by)\b[^,]{0,250},\s*', re.I)
SENT_SPLIT = re.compile(r'(?<=[.?!:])\s+(?=[A-Z])')
STOP = set('the a an of to and in for by with using its their from on at as is be that this or under which whether each when '
           'how why what your you it its are was were has have been into between than then so such these those there given '
           'one two three four five'.split())


def norm_verb(w):
    w = w.lower().strip(',.;:')
    return w if w in LEX else None


def instruction_verbs(q):
    """Return (primary_verb, primary_phrase, secondary_verbs, question_form, scenario_prefix)."""
    sents = SENT_SPLIT.split(q.replace('\n', ' ').strip())
    primary, phrase, secondary, scenario = None, None, [], False
    for i, s in enumerate(sents):
        body = s.strip()
        m = LEADING.match(body)
        if m:
            body = body[m.end():]
        words = body.split()
        if not words:
            continue
        v = norm_verb(words[0])
        if v:
            if primary is None:
                primary = v
                nxt = ' '.join(w.lower().strip(',.;:') for w in words[1:3])
                phrase = f'{v} {nxt}'.strip()
                if v == 'determine' and words[1:2] and words[1].lower() in ('whether', 'if'):
                    primary = 'determine_whether'
                scenario = i > 0 or bool(m)
            for mm in re.finditer(r'\b(?:and|then|,)\s+([a-z]+)\s+(?=the|a|an|how|why|whether|which|what|its|their|each|your|it|one|two|three|all|both|any|this|these|that)\b', body):
                sv = norm_verb(mm.group(1))
                if sv and sv != primary:
                    secondary.append(sv)
    form = 'imperative' if primary else ('interrogative' if q.strip().endswith('?') else 'other')
    return primary, phrase, secondary, form, scenario


def content_tokens(q):
    return {w for w in re.findall(r'[a-z0-9]+', q.lower()) if w not in STOP and len(w) > 2 and w not in LEX}


def entropy(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c) if n else 0.0


# ---------------------------------------------------------------- load frozen exports
examples = []
for s in SPLITS:
    with open(os.path.join(TR, 'splits', f'{s}_examples.jsonl'), encoding='utf-8') as f:
        examples += [json.loads(l) for l in f]
rows = []
for x in examples:
    for lv, q in x['transformations'].items():
        pv, ph, sec, form, scen = instruction_verbs(q)
        fam = FAMILY.get(pv) if pv else None
        rows.append({'example_id': x['id'], 'split': x['split'], 'eligibility': x['eligibility'], 'role': 'transformation',
                     'level': lv, 'source_level': x['source_bloom_level'], 'question': q, 'primary_verb': pv or 'NONE',
                     'primary_phrase': ph or '', 'family': fam or 'NONE', 'secondary_verbs': sec, 'form': form,
                     'scenario_prefix': scen, 'provenance': x['transformation_provenance'][lv],
                     'false_claim': lv in x['deliberately_false_claim_levels']})
src_rows = []
for x in examples:
    pv, ph, sec, form, scen = instruction_verbs(x['source_question'])
    src_rows.append({'example_id': x['id'], 'split': x['split'], 'level': x['source_bloom_level'], 'question': x['source_question'],
                     'primary_verb': pv or 'NONE', 'family': FAMILY.get(pv, 'NONE') if pv else 'NONE', 'form': form})


def w(name, header, data):
    with open(os.path.join(OUT, name), 'w', encoding='utf-8-sig', newline='') as f:
        cw = csv.writer(f); cw.writerow(header); cw.writerows(data)


results = {'scope': {'examples': len(examples), 'transformation_rows': len(rows),
                     'by_split': dict(Counter(r['split'] for r in rows)), 'by_eligibility': dict(Counter(r['eligibility'] for r in rows))}}

# ---------------------------------------------------------------- Audit 1: per-level verb / family frequency
per_level = {}
freq_rows = []
for lv in LEVELS:
    R = [r for r in rows if r['level'] == lv]
    vc, fc = Counter(r['primary_verb'] for r in R), Counter(r['family'] for r in R)
    phc = Counter(r['primary_phrase'] for r in R)
    top1 = vc.most_common(1)[0][1] / len(R); top3 = sum(c for _, c in vc.most_common(3)) / len(R)
    per_level[lv] = {
        'rows': len(R), 'unique_primary_verbs': len([v for v in vc if v != 'NONE']), 'unique_families': len([f for f in fc if f != 'NONE']),
        'top1_verb_share': round(top1, 3), 'top3_verb_share': round(top3, 3),
        'top1_family_share': round(fc.most_common(1)[0][1] / len(R), 3),
        'verb_entropy_bits': round(entropy(vc), 3), 'effective_verbs': round(2 ** entropy(vc), 2),
        'family_entropy_bits': round(entropy(fc), 3), 'effective_families': round(2 ** entropy(fc), 2),
        'verbs_used_once': sorted(v for v, c in vc.items() if c == 1),
        'no_instruction_verb_found': vc.get('NONE', 0),
        'scenario_prefixed_rows': sum(r['scenario_prefix'] for r in R),
        'rows_with_secondary_verb': sum(1 for r in R if r['secondary_verbs']),
        'top_phrases': phc.most_common(8),
    }
    for v, c in vc.most_common():
        freq_rows.append([lv, v, FAMILY.get(v, 'NONE'), c, round(c / len(R), 4)])
w('verb_frequency_by_level.csv', ['target_level', 'primary_verb', 'family', 'count', 'share_of_level'], freq_rows)
fam_rows = []
for lv in LEVELS:
    R = [r for r in rows if r['level'] == lv]
    for f_, c in Counter(r['family'] for r in R).most_common():
        fam_rows.append([lv, f_, OPERATION.get(f_, ''), c, round(c / len(R), 4)])
w('family_frequency_by_level.csv', ['target_level', 'family', 'operation', 'count', 'share_of_level'], fam_rows)
results['per_level'] = per_level

# ---------------------------------------------------------------- Audit 2: verb x level matrix
verbs = sorted({r['primary_verb'] for r in rows}, key=lambda v: -sum(1 for r in rows if r['primary_verb'] == v))
mat = {v: Counter(r['level'] for r in rows if r['primary_verb'] == v) for v in verbs}
w('verb_level_matrix.csv', ['primary_verb', 'family'] + LEVELS + ['total', 'levels_used', 'max_level_share'],
  [[v, FAMILY.get(v, 'NONE')] + [mat[v][l] for l in LEVELS] + [sum(mat[v].values()), sum(1 for l in LEVELS if mat[v][l]),
   round(max(mat[v].values()) / sum(mat[v].values()), 3)] for v in verbs])
fams = sorted({r['family'] for r in rows}, key=lambda f_: -sum(1 for r in rows if r['family'] == f_))
fmat = {f_: Counter(r['level'] for r in rows if r['family'] == f_) for f_ in fams}
w('family_level_matrix.csv', ['family', 'operation'] + LEVELS + ['total', 'levels_used'],
  [[f_, OPERATION.get(f_, '')] + [fmat[f_][l] for l in LEVELS] + [sum(fmat[f_].values()), sum(1 for l in LEVELS if fmat[f_][l])] for f_ in fams])
results['cross_level'] = {
    'verbs_single_level': sorted([v for v in verbs if sum(1 for l in LEVELS if mat[v][l]) == 1 and v != 'NONE'],
                                 key=lambda v: -sum(mat[v].values())),
    'verbs_multi_level': {v: {l: mat[v][l] for l in LEVELS if mat[v][l]} for v in verbs if sum(1 for l in LEVELS if mat[v][l]) > 1},
    'high_volume_verbs_ge90pct_one_level': {v: dict(mat[v]) for v in verbs if sum(mat[v].values()) >= 20 and max(mat[v].values()) / sum(mat[v].values()) >= 0.9},
}
# source questions (used as classification records)
results['source_questions'] = {lv: {'rows': sum(1 for r in src_rows if r['level'] == lv),
                                    'verbs': Counter(r['primary_verb'] for r in src_rows if r['level'] == lv).most_common(10)} for lv in LEVELS}

# ---------------------------------------------------------------- Audit 6: lexical shortcut analysis
def majority_predictor(train, key):
    by = defaultdict(Counter)
    for r in train: by[key(r)][r['level']] += 1
    fallback = Counter(r['level'] for r in train).most_common(1)[0][0]
    return lambda r: by[key(r)].most_common(1)[0][0] if key(r) in by else fallback


def nb_predictor(train):
    prior, tok = Counter(), defaultdict(Counter)
    vocab = set()
    for r in train:
        prior[r['level']] += 1
        for t in re.findall(r'[a-z]+', r['question'].lower()):
            tok[r['level']][t] += 1; vocab.add(t)
    tot = {l: sum(tok[l].values()) for l in LEVELS}
    def pred(r):
        ts = re.findall(r'[a-z]+', r['question'].lower())
        return max(LEVELS, key=lambda l: math.log(prior[l] + 1) + sum(math.log((tok[l][t] + 1) / (tot[l] + len(vocab))) for t in ts))
    return pred


def evaluate(pred, test):
    ok = sum(pred(r) == r['level'] for r in test)
    per = {l: round(sum(pred(r) == l for r in test if r['level'] == l) / max(1, sum(r['level'] == l for r in test)), 3) for l in LEVELS}
    return round(ok / len(test), 4), per


train_rows = [r for r in rows if r['split'] == 'train']
held = [r for r in rows if r['split'] != 'train']
first_word = lambda r: r['question'].split()[0].lower().strip(',')
first3 = lambda r: ' '.join(r['question'].lower().split()[:3])
predictors = {
    'majority_class_baseline': (lambda r: Counter(x['level'] for x in train_rows).most_common(1)[0][0]),
    'first_word_only': majority_predictor(train_rows, first_word),
    'primary_instruction_verb_only': majority_predictor(train_rows, lambda r: r['primary_verb']),
    'primary_verb_family_only': majority_predictor(train_rows, lambda r: r['family']),
    'first_three_words': majority_predictor(train_rows, first3),
    'bag_of_words_naive_bayes_full_text': nb_predictor(train_rows),
}
shortcut = {}
for name, p in predictors.items():
    acc, per = evaluate(p, held)
    shortcut[name] = {'accuracy_on_validation_plus_test': acc, 'recall_by_level': per}
# same analysis for the Bloom-classification view (source questions + transformations) using source verbs too
cls_train = train_rows + [dict(r, role='source') for r in src_rows if r['split'] == 'train']
cls_held = held + [dict(r, role='source') for r in src_rows if r['split'] != 'train']
for name, mk in [('classification_view_primary_verb_only', lambda t: majority_predictor(t, lambda r: r['primary_verb'])),
                 ('classification_view_naive_bayes_full_text', nb_predictor)]:
    acc, per = evaluate(mk(cls_train), cls_held)
    shortcut[name] = {'accuracy_on_validation_plus_test': acc, 'recall_by_level': per}
src_only_held = [dict(r) for r in src_rows if r['split'] != 'train']
acc, per = evaluate(majority_predictor(cls_train, lambda r: r['primary_verb']), src_only_held)
shortcut['source_questions_only_primary_verb'] = {'accuracy_on_validation_plus_test': acc, 'recall_by_level': per, 'n': len(src_only_held)}
lvl = Counter(r['level'] for r in rows); verb_c = Counter(r['primary_verb'] for r in rows)
joint = Counter((r['primary_verb'], r['level']) for r in rows)
H_level = entropy(lvl)
H_level_given_verb = sum(verb_c[v] / len(rows) * entropy(Counter({l: joint[(v, l)] for l in LEVELS})) for v in verb_c)
H_verb_given_level = sum(lvl[l] / len(rows) * entropy(Counter({v: joint[(v, l)] for v in verb_c})) for l in LEVELS)
shortcut['information'] = {'H_level_bits': round(H_level, 3), 'H_level_given_primary_verb_bits': round(H_level_given_verb, 3),
                           'share_of_level_uncertainty_removed_by_verb': round(1 - H_level_given_verb / H_level, 3),
                           'H_verb_given_level_bits': round(H_verb_given_level, 3),
                           'n_train': len(train_rows), 'n_heldout': len(held)}
results['lexical_shortcut'] = shortcut

# ---------------------------------------------------------------- Audit 4: templates and distinctness
tmpl = {}
STOCK = {
    'justify your judgement/conclusion/answer': r'\bjustify your (judge?ment|conclusion|answer|choice|decision|recommendation|assessment|position|evaluation)\b',
    'explain how/why the': r'^Explain (how|why) the\b',
    'examine the relationship between': r'^Examine the relationship between\b',
    'examine how': r'^Examine how\b',
    'verify whether': r'\bVerify whether\b',
    'assess whether': r'\bAssess whether\b',
    'design a / develop a': r'^(Design|Develop) an? \b',
    'define … ': r'^Define\b',
    'state the': r'^State the\b',
    'identify the': r'^Identify the\b',
    'explaining/using its … (gerund tail)': r', (explaining|showing|stating|justifying|including|using) ',
}
for lv in LEVELS:
    R = [r for r in rows if r['level'] == lv]
    o2 = Counter(' '.join(r['question'].split()[:2]) for r in R); o3 = Counter(' '.join(r['question'].split()[:3]) for r in R)
    o4 = Counter(' '.join(r['question'].split()[:4]) for r in R)
    ends = Counter(' '.join(re.sub(r'[.?]$', '', r['question'].strip()).split()[-3:]) for r in R)
    tmpl[lv] = {'distinct_opening_2grams': len(o2), 'top_opening_2grams': o2.most_common(6),
                'top_opening_3grams': o3.most_common(6), 'top_opening_4grams': o4.most_common(5),
                'share_top5_opening_3grams': round(sum(c for _, c in o3.most_common(5)) / len(R), 3),
                'top_endings_3grams': ends.most_common(5),
                'stock_patterns': {k: sum(1 for r in R if re.search(p, r['question'])) for k, p in STOCK.items() if sum(1 for r in R if re.search(p, r['question']))}}
results['templates'] = tmpl

# within-example distinctness and verb-swap detection
by_ex = defaultdict(list)
for r in rows: by_ex[r['example_id']].append(r)
src_by = {r['example_id']: r for r in src_rows}
near_pairs, swap_src, low_overlap = [], [], []
def body(q): return content_tokens(' '.join(q.split()[1:]))
for eid, R in by_ex.items():
    for a, b in combinations(R, 2):
        ta, tb = body(a['question']), body(b['question'])
        j = len(ta & tb) / max(1, len(ta | tb))
        if j >= 0.7: near_pairs.append([eid, a['level'], b['level'], round(j, 3), a['question'][:160], b['question'][:160]])
    s = src_by[eid]; ts = body(s['question'])
    for r in R:
        tr_ = body(r['question'])
        j = len(ts & tr_) / max(1, len(ts | tr_))
        if j >= 0.8: swap_src.append([eid, s['level'], r['level'], round(j, 3), s['question'][:160], r['question'][:160]])
        ct_src = content_tokens(s['question'])
        keep = len(ct_src & content_tokens(r['question'])) / max(1, len(ct_src))
        if keep < 0.15 and len(ct_src) >= 4: low_overlap.append([eid, s['level'], r['level'], round(keep, 3), s['question'][:160], r['question'][:200]])
w('near_identical_transformation_pairs.csv', ['example_id', 'level_a', 'level_b', 'body_jaccard', 'question_a', 'question_b'], sorted(near_pairs, key=lambda x: -x[3]))
w('verb_swap_of_source_candidates.csv', ['example_id', 'source_level', 'target_level', 'body_jaccard', 'source', 'transformation'], sorted(swap_src, key=lambda x: -x[3]))
w('low_source_overlap_candidates.csv', ['example_id', 'source_level', 'target_level', 'source_content_retained', 'source', 'transformation'], sorted(low_overlap, key=lambda x: x[3]))
results['distinctness'] = {'near_identical_pairs_jaccard_ge_0.7': len(near_pairs), 'verb_swap_of_source_jaccard_ge_0.8': len(swap_src),
                           'low_source_content_retention_lt_0.15': len(low_overlap)}

# ---------------------------------------------------------------- Audit 3: alignment candidate flags (heuristic; reviewed manually)
JUDGE = re.compile(r'\b(whether|justify|judge|assess|evaluate|recommend|critique|valid|appropriate|suitab|better|best|merit|defend|'
                   r'correct|consistent|adequate|sufficient|trade-?off|weigh|prioriti|select|choose|decide|which .* should|support(ed)? by)\b'
                   + (r'|\b(validat\w*|substantiat\w*|refut\w*|critically|apprais\w*|audit\w*|rate|challenge|review|sound|soundness|verify|establish)\b'
                      if EXTENDED else ''), re.I)
NEWART = re.compile(r'\b(design|develop|construct|formulate|propose|devise|compose|draft|plan|create|build|write|produce|generate|model|conjecture|rewrite|redesign|revise|express'
                    + ('|architect|engineer|synthesi[sz]e|integrate|combine|adapt|author|assemble|elaborate|establish|hypothesi[sz]e' if EXTENDED else '')
                    + r')\b', re.I)
RECALL_ONLY = {'define', 'state', 'list', 'name', 'recall', 'enumerate', 'expand'}
HIGHER = {'compare', 'contrast', 'evaluate', 'assess', 'justify', 'critique', 'analyze', 'analyse', 'examine', 'design', 'recommend'}
flags = []
for r in rows:
    q, lv, v = r['question'], r['level'], r['primary_verb']
    sec = set(r['secondary_verbs'])
    reason = None
    if lv == 'Remember' and (v in ('explain', 'describe', 'interpret', 'distinguish', 'differentiate') or re.search(r'\b(why|how does|how do|explain)\b', q, re.I)):
        reason = 'Remember row asks for explanation/comparison'
    elif lv == 'Understand' and (v in HIGHER or sec & HIGHER or re.search(r'\b(justify|evaluate|assess|design|propose|calculate)\b', q, re.I)):
        reason = 'Understand row may demand higher-level work'
    elif lv == 'Apply' and v in RECALL_ONLY | {'explain', 'describe'}:
        reason = 'Apply row opens with a recall/explanation verb'
    elif lv == 'Apply' and not re.search(r'\d|given|following|this|these|scenario|case|data|stated|specified|for an?\b' + ('|supplied' if EXTENDED else ''), q, re.I):
        reason = 'Apply row has no concrete case/data cue'
    elif lv == 'Analyze' and v in RECALL_ONLY | {'identify', 'explain', 'describe'}:
        reason = 'Analyze row opens with a recall/explanation verb'
    elif lv == 'Analyze' and re.search(r'^Examine (what|the definition|the meaning)\b', q):
        reason = 'Analyze row may ask only for meaning/definition'
    elif lv == 'Evaluate' and not JUDGE.search(q):
        reason = 'Evaluate row has no judgement/criterion cue'
    elif lv == 'Create' and not NEWART.search(q):
        reason = 'Create row has no new-artefact verb'
    elif lv == 'Create' and re.search(r'\b(standard|textbook|well-known|known) (algorithm|procedure|method|proof|derivation)\b', q, re.I):
        reason = 'Create row may reproduce a known solution'
    if reason:
        flags.append([r['example_id'], r['split'], r['eligibility'], lv, v, reason, r['question']])
w('alignment_candidate_flags.csv', ['example_id', 'split', 'eligibility', 'target_level', 'primary_verb', 'heuristic_reason', 'question'], flags)
results['alignment_flags'] = dict(Counter(f[5] for f in flags))

# ---------------------------------------------------------------- Audit 5: operation coverage per level
cov = {lv: Counter(OPERATION.get(r['family'], 'none detected') for r in rows if r['level'] == lv) for lv in LEVELS}
results['operation_coverage'] = {lv: dict(cov[lv].most_common()) for lv in LEVELS}
# also count secondary operations (rows where a second instruction verb adds an operation)
results['secondary_operations'] = {lv: dict(Counter(OPERATION.get(FAMILY.get(v), '?') for r in rows if r['level'] == lv for v in r['secondary_verbs']).most_common(8)) for lv in LEVELS}
# split-level summary for train (what the model actually sees)
results['train_only_per_level'] = {lv: {'rows': sum(1 for r in train_rows if r['level'] == lv),
                                        'top1_verb_share': round(Counter(r['primary_verb'] for r in train_rows if r['level'] == lv).most_common(1)[0][1] / sum(1 for r in train_rows if r['level'] == lv), 3),
                                        'unique_verbs': len({r['primary_verb'] for r in train_rows if r['level'] == lv} - {'NONE'})} for lv in LEVELS}
results['usable_only_per_level'] = {lv: {'rows': sum(1 for r in rows if r['level'] == lv and r['eligibility'] == 'usable'),
                                         'top1_verb_share': round(Counter(r['primary_verb'] for r in rows if r['level'] == lv and r['eligibility'] == 'usable').most_common(1)[0][1] / sum(1 for r in rows if r['level'] == lv and r['eligibility'] == 'usable'), 3)} for lv in LEVELS}
w('row_level_extraction.csv', ['example_id', 'split', 'eligibility', 'target_level', 'source_level', 'primary_verb', 'family', 'primary_phrase',
                               'secondary_verbs', 'form', 'scenario_prefix', 'provenance', 'deliberately_false_claim', 'question'],
  [[r['example_id'], r['split'], r['eligibility'], r['level'], r['source_level'], r['primary_verb'], r['family'], r['primary_phrase'],
    ';'.join(r['secondary_verbs']), r['form'], r['scenario_prefix'], r['provenance'], r['false_claim'], r['question']] for r in rows])
with open(os.path.join(OUT, 'audit_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print(json.dumps({k: results[k] for k in ('scope', 'distinctness', 'alignment_flags')}, indent=1))
for lv in LEVELS:
    p = per_level[lv]
    print(lv, {k: p[k] for k in ('rows', 'unique_primary_verbs', 'unique_families', 'top1_verb_share', 'top3_verb_share', 'effective_verbs', 'effective_families', 'no_instruction_verb_found', 'scenario_prefixed_rows')})
print(json.dumps(shortcut, indent=1))
