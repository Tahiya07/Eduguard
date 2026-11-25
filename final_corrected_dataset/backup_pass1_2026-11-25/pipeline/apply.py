import json, re, os, sys, copy
HERE = os.path.dirname(os.path.abspath(__file__))
LV = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']

def load_patch(name):
    ns = {}
    exec(open(os.path.join(HERE, 'patches', name), encoding='utf-8').read(), ns)
    return ns

recs = json.load(open(os.path.join(HERE, 'structured.json'), encoding='utf-8'))
R = {r['id']: r for r in recs}
orig_snapshot = copy.deepcopy(R)
log = []      # dicts: id, target, original, corrected, category, reason
def L(i, target, old, new, cat, reason):
    log.append(dict(id=i, target=target, original=old, corrected=new, category=cat, reason=reason))

REASONS = {
 'weak_create_calc': "Create row asked only for a complete worked calculation/derivation of the same problem, which is Apply-level execution. Rewritten to require designing or constructing a new artifact (plan, design, modified scenario, generalisation) while preserving the original concept, values, units, and constraints.",
 'weak_create_table': "Create row asked only for a table, sheet, record, or statement that organises computed or given results (Apply/Understand). Rewritten to require meaningful construction grounded in the same scenario and data.",
 'weak_create_repr': "Create row asked the learner to reproduce a standard representation, diagram, or model (Understand/Apply). Rewritten so the learner must construct something new from the same concept and data.",
 'weak_create_eval': "Create row asked for an assessment or judgement of the given case (Evaluate-level). Rewritten to require constructing a plan, design, or proposal that responds to the same case.",
 'weak_create_explain': "Create row asked for an explanation or interpretation of the given case (Understand/Analyze). Rewritten to require original construction from the same content.",
 'weak_create_trace': "Create row asked for an execution trace of a known algorithm on the given input (Apply). Rewritten to require designing a new input, variant, or artifact that uses the same content.",
}

# ---------------- 1. Notation normalisation ----------------
def bmatrix(m):
    body = m.group(1).strip()
    rows = [r.strip() for r in body.split('\\\\')]
    rows = ['[' + ', '.join(c.strip() for c in r.split('&')) + ']' for r in rows if r]
    return '[' + ', '.join(rows) + ']'
def norm_math(s):
    t = s
    t = re.sub(r'\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}', bmatrix, t, flags=re.S)
    reps = [(r'\rightleftharpoons', '⇌'), (r'\rightarrow', '→'), (r'\mid', '∣'), (r'\epsilon', 'ε'),
            (r'\lambda', 'λ'), (r'\mu', 'μ'), (r'\max', 'max'), (r'\quad', ' '), (r'\times', '×'),
            (r'\cdot', '·'), (r'\ ', ' '), (r'\,', ' ')]
    for a, b in reps: t = t.replace(a, b)
    t = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\\text\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\^\{([^}]*)\}', r'^(\1)', t)
    t = re.sub(r'_\{([^}]*)\}', r'_\1', t)
    t = t.replace('\\(', '').replace('\\)', '').replace('\\[', '').replace('\\]', '')
    t = re.sub(r'(\S)\s*\^\(-1\)', r'\1^(-1)', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'([→∣]) +', r'\1', t)
    t = re.sub(r' +∣', '∣', t)
    t = re.sub(r' ([,.;:])', r'\1', t)
    return t.strip()
COSMETIC = [('x^2sinx', 'x^2 sin x'), ('xe^xdx', 'x e^x dx'), ('xe^x', 'x e^x')]
def norm_all(s):
    t = norm_math(s)
    for a, b in COSMETIC: t = t.replace(a, b)
    return t

for r in recs:
    joined = ' '.join(r['original_lines'])
    new_lines = [norm_all(x) for x in r['original_lines']]
    if new_lines != r['original_lines']:
        L(r['id'], 'Original', joined, ' '.join(new_lines), 'notation',
          'LaTeX/flattened-equation fragments converted to consistent plain-text mathematical notation; meaning unchanged.')
        r['original_lines'] = new_lines
    for k in list(r['transformations']):
        v = r['transformations'][k]; nv = norm_all(v)
        if nv != v:
            L(r['id'], k, v, nv, 'notation', 'LaTeX/flattened-equation fragments converted to consistent plain-text mathematical notation; meaning unchanged.')
            r['transformations'][k] = nv
    if r['clo']:
        r['clo'] = norm_all(r['clo'])
    for tb in r['data_tables']:
        for row in tb:
            for j, c in enumerate(row): row[j] = norm_all(c)

# ---------------- 2. Structural fixes ----------------
sp = load_patch('p04_special.py')
r465 = R[465]
old465 = ' | '.join(r465['original_lines'])
r465['original_lines'] = [re.sub('[-]', '', x) for x in r465['original_lines']]
r465['code_lines'] = [1, 2, 3, 4]
L(465, 'Original', old465, ' | '.join(r465['original_lines']), 'structural',
  'Private-use glyph markers left by the exported code-block label were removed; the SQL query is kept verbatim and set as a code block.')
for k, r in enumerate(recs[:-1]):
    if r['section_header_after']:
        recs[k + 1]['section'] = r['section_header_after']
for i, (old, new) in sp['TITLES'].items():
    assert R[i]['title'] == old, (i, R[i]['title'])
    R[i]['title'] = new; R[i]['topic'] = new.split(' — ', 1)[1]
    L(i, 'Title', old, new, 'structural', 'Italic asterisk of "A*" was displaced to the end of the title during export; restored.')
r105 = R[105]
m = re.match(r'\|\s*Create\s*\|\s*(.*)$', r105['stray_create'])
restored = m.group(1).strip()
r105['transformations']['Create'] = restored
L(105, 'Create', r105['stray_create'], restored, 'structural',
  'The Create row was outside the transformation table (rendered as a stray "| Create | ..." paragraph). Restored as the fifth table row.')
r105['stray_create'] = None

# ---------------- 3. Bloom label corrections ----------------
for i, (old, new, reason, drop, (add_lv, add_txt)) in sp['LABELS'].items():
    r = R[i]; assert r['original_level'] == old, i
    r['original_level'] = new
    L(i, 'Original Bloom Level', old, new, 'bloom_label', reason)
    dropped = r['transformations'].pop(drop)
    L(i, drop, dropped, '(removed: level now equals the corrected source level)', 'level_coverage',
      f'After relabelling the source as {new}, a {drop} transformation would duplicate the source level; removed.')
    r['transformations'][add_lv] = add_txt
    L(i, add_lv, '(missing)', add_txt, 'level_coverage',
      f'After relabelling the source from {old} to {new}, a {add_lv} transformation was required for the five-level set; added, preserving the original content.')

# ---------------- 4. Transformation patches ----------------
for name in ['p01_create.py', 'p02_create.py', 'p03_create.py']:
    for (i, lv, txt, key) in load_patch(name)['P']:
        r = R[i]; assert lv in r['transformations'], (name, i, lv)
        old = r['transformations'][lv]
        r['transformations'][lv] = txt
        L(i, lv, old, txt, 'create_strengthened', REASONS[key])
for (i, lv, txt, cat, reason) in sp['T']:
    r = R[i]; assert lv in r['transformations'], (i, lv)
    old = r['transformations'][lv]
    r['transformations'][lv] = txt
    L(i, lv, old, txt, 'create_strengthened' if cat.startswith('weak_create') else cat, reason)

# ---------------- 5. Evaluate: explicit justification ----------------
JUST = ' Justify your judgement with explicit reference to the criteria used.'
for r in recs:
    t = r['transformations'].get('Evaluate')
    if not t or re.search(r'justif', t, re.I): continue
    nt = t.rstrip() + ('' if t.rstrip().endswith(('.', '?')) else '.') + JUST
    r['transformations']['Evaluate'] = nt
    L(r['id'], 'Evaluate', t, nt, 'evaluate_justification',
      'Evaluate row asked for a judgement or verification without requiring the learner to justify it against explicit criteria; a justification requirement was added (task content unchanged).')

# ---------------- 6. Unresolved items ----------------
unres = []
EXT = re.compile(r'\b(supplied|attached|provided|specified|following|below|given (?:data|information|values|scenario|program|code|matrix|graph|network|circuit|function|equation|table|figure|diagram|schedule))\b', re.I)
PROPOSED = re.compile(r'\b(a|the|an) (proposed|reported|calculated|candidate|attempted|supplied) ([a-z\-]+ ){0,4}?(design|solution|plan|diagram|tree|allocation|sets?|schedule|proof|answer|interpretation|classification|labels?|equations?|values?|sequence|waveform|representation|strategy|layout|model|DFA|grammar|statement|analysis|arrangement|output|result|code|query|ER diagram|rules|program|matrix|graph|curve|chart|configuration|selection|dose|rating|identification)\b', re.I)
has_inline = {405, 429, 445, 465, 477, 502, 569, 428, 442, 475, 548}
for r in recs:
    oq = ' '.join(r['original_lines'])
    hits = sorted(set(m.group(0).lower() for m in EXT.finditer(oq)))
    if r['id'] in has_inline or r['data_tables']:
        hits = [h for h in hits if h not in ('following', 'below')]
    if hits:
        unres.append(dict(id=r['id'], location='Original (and dependent transformations)',
            issue='External material referenced but not included',
            detail=f"The source question refers to material that is not present in the dataset ({', '.join(hits)}). Transformations inherit the reference. No data was invented.",
            action='Supply the referenced data/scenario in the item, or rephrase the item so it is answerable without it.'))
    for lv, t in r['transformations'].items():
        m0 = PROPOSED.search(t)
        if m0 and m0.group(2).lower() == 'supplied' and hits:  # already covered by the source-level row
            continue
        if m0 and not re.search(r'\d', t) and r['id'] not in has_inline:
            m = PROPOSED.search(t)
            unres.append(dict(id=r['id'], location=lv, issue='Transformation evaluates an unstated artefact',
                detail=f'The {lv} row refers to "{m.group(0)}" whose content is not given in the item.',
                action='Provide the artefact (e.g., the proposed answer or design) in the item, or rephrase.'))
for (i, lv, note) in sp['CHECK_NOTES']:
    if lv == 'Original':
        unres.append(dict(id=i, location='Original data table', issue='Technical data concern (SME review)', detail=note, action='Confirm whether the inadmissible heuristic is intentional; if not, revise h(A) and h(C).'))
unres.append(dict(id=445, location='Evaluate', issue='Restored content requires confirmation', detail='The proposed FIRST/FOLLOW sets were empty in the source; they were filled with the sets derived from the grammar (no other source available).', action='Confirm whether the original author intended deliberately incorrect sets for error detection.'))
for a, b, note in sp['DUPLICATES']:
    unres.append(dict(id=a, location=f'Whole example (duplicate of {b:03d})', issue='Exact/functional duplicate retained', detail=note + ' Both retained so the dataset keeps 600 examples; transformations of the pair were differentiated where they were rewritten.', action=f'Decide whether to merge {a:03d} and {b:03d} or replace one with a new item.'))

# ---------------- 7. Validation ----------------
problems = []
for r in recs:
    tr = r['transformations']
    if r['original_level'] not in LV: problems.append((r['id'], 'bad level'))
    exp = set(LV) - {r['original_level']}
    if set(tr) != exp: problems.append((r['id'], 'level set', sorted(tr)))
    if len(tr) != 5: problems.append((r['id'], 'count', len(tr)))
    for k, v in tr.items():
        if not v.strip(): problems.append((r['id'], 'empty', k))
        if '\\' in v: problems.append((r['id'], 'latex left', k, v))
        if re.sub(r'\W', '', v.lower()) == re.sub(r'\W', '', ' '.join(r['original_lines']).lower()): problems.append((r['id'], 'identical to original', k))
    if len(set(tr.values())) != len(tr): problems.append((r['id'], 'duplicate transformations'))
assert len(recs) == 600 and sorted(R) == list(range(1, 601))
print('validation problems:', problems)

# order transformations
for r in recs:
    r['transformations'] = {k: r['transformations'][k] for k in LV if k in r['transformations']}
json.dump(dict(records=recs, log=log, unresolved=unres, check_notes=sp['CHECK_NOTES'],
               duplicates=sp['DUPLICATES'], near_duplicates=sp['NEAR_DUPLICATES'], labels=list(sp['LABELS'])),
          open(os.path.join(HERE, 'corrected_full.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
from collections import Counter
print('log entries', len(log), Counter(e['category'] for e in log))
print('unresolved', len(unres), Counter(u['issue'] for u in unres))
# changed transformations count
changed = set((e['id'], e['target']) for e in log if e['target'] in LV)
print('transformations changed (any category):', len(changed))
