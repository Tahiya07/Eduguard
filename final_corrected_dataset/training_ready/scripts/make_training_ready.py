"""Build the EduGuard training-ready exports from the corrected master dataset.

Reads (never writes) the master files one directory above `training_ready/`:
  corrected_dataset.json, Correction_Log.csv, Unresolved_Questions.csv
Writes every data file in `training_ready/` (example exports, splits/, answer_generation/, audit, exclusions,
reclassifications, leakage report, integrity report). Standard library only; deterministic.
Fixes D1-D5 of RECONCILIATION_REPORT.md add per-record provenance, deliberately-false-claim markers and the strict
usable-only answer_generation/ export; they add fields only and do not change split assignment or any question text.

Run:  python -I training_ready/scripts/make_training_ready.py
"""
import csv, hashlib, json, os, random, re, sys
from collections import Counter, defaultdict
from itertools import combinations

SEED = 20261125
SPLIT_TARGETS = {'train': 0.80, 'validation': 0.10, 'test': 0.10}
LEVELS = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
NEAR_DUP_ORIGINAL_JACCARD = 0.35   # source-question similarity that forces grouping
NEAR_DUP_ANY_TEXT_JACCARD = 0.40   # any question (source or transformation) pair across examples
TOPIC_GROUPING = True              # examples sharing the same subject:subtopic title are grouped

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                      # .../training_ready
MASTER = os.path.dirname(OUT)                    # .../final_corrected_dataset
MASTER_FILES = ['corrected_dataset.json', 'Correction_Log.csv', 'Unresolved_Questions.csv',
                'Final_Summary.md', 'EduGuard_600_Corrected.pdf']

# ---------------------------------------------------------------- self-containment audit (manual verdicts)
# Every Correction_Log row whose category includes "Self-containment" was read against its source question.
# Default verdict: PRESERVES. Exceptions below.
AUDIT_FLAGS = {
    (158, 'Apply'): ('FLAG_OBJECTIVE_SHIFT',
        "Source objective is how structural modification influences pharmacological ACTIVITY; the row asks about lipophilicity and oral absorption "
        "(a pharmacokinetic/prodrug question). Different learning objective; needs author decision."),
    (393, 'Apply'): ('FLAG_SCOPE_EXPANSION',
        "Source distinguishes public goods from private goods only; the row adds a 'neither' category (club goods, common-pool resources), "
        "expanding the technical scope beyond the source objective."),
    (393, 'Create'): ('FLAG_SCOPE_EXPANSION',
        "Row requires placing goods into four categories of goods; the source objective covers only public versus private goods."),
    (319, 'Apply'): ('FLAG_CONTESTED_CONTENT',
        "Row treats the 1906 founding of the All-India Muslim League as a political consequence of the 1905 partition; historians dispute how far "
        "the partition caused it, so a single correct answer key is not secure. Needs subject-specialist review."),
}
AUDIT_NOTES = {
    (305, 'Apply'): "Objective preserved. The USD 2.15/day international poverty line is time-sensitive (World Bank lines are revised periodically); the classification task (absolute vs relative) is unaffected, but the figure should be checked before publication.",
    (120, 'Apply'): "Objective preserved. Glass vials may be sterilized/depyrogenated by dry heat or autoclaved; accept either with justification.",
    (170, 'Apply'): "Objective preserved. Composting is classed under the disposal stage, which some curricula call treatment/recovery; accept with justification.",
    (339, 'Create'): "Objective preserved; scope narrowed from a generic limit test to the chloride limit test (a standard instance).",
}
ROWS_FOR_REVIEW_EXCLUDE_EXAMPLE = True  # an example with any flagged row is held out (it would otherwise lack five valid rows)

# ---------------------------------------------------------------- helpers
STOP = set('the a an of to and in for by with using its their from on at as is be that this or under which whether each when '
           'stated supplied specified given following explain determine calculate analyze analyse examine assess evaluate verify '
           'identify define state design develop construct apply'.split())
def toks(s):
    return set(w for w in re.findall(r'[a-z0-9.]+', s.lower()) if w not in STOP and len(w) > 1)
def jacc(a, b):
    return len(a & b) / max(1, len(a | b))
def norm(s):
    return re.sub(r'\W+', ' ', s.lower()).strip()
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
def wjsonl(path, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

class DSU:
    def __init__(self, items): self.p = {i: i for i in items}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[max(a, b)] = min(a, b)

# ---------------------------------------------------------------- load master (read-only)
before = {f: sha256(os.path.join(MASTER, f)) for f in MASTER_FILES}
data = json.load(open(os.path.join(MASTER, 'corrected_dataset.json'), encoding='utf-8'))
log = list(csv.DictReader(open(os.path.join(MASTER, 'Correction_Log.csv'), encoding='utf-8-sig')))
unres = list(csv.DictReader(open(os.path.join(MASTER, 'Unresolved_Questions.csv'), encoding='utf-8-sig')))
E = {int(x['id']): x for x in data}
assert len(E) == 600

# ---------------------------------------------------------------- 6. self-containment audit file
sc_rows = [r for r in log if 'Self-containment' in r['Issue Category']]
audit = []
for r in sc_rows:
    i, lv = int(r['Example ID']), r['Target']
    verdict, note = AUDIT_FLAGS.get((i, lv), ('PRESERVES', AUDIT_NOTES.get((i, lv), 'Introduced scenario keeps the source learning objective, domain, cognitive demand, and technical scope.')))
    audit.append({'example_id': f'{i:03d}', 'target_level': lv, 'verdict': verdict,
                  'source_question': E[i]['original_question'].replace('\n', ' '), 'source_level': E[i]['original_bloom_level'],
                  'row_before_pass2': r['Original Wording'], 'row_now': r['Corrected Wording'], 'audit_note': note,
                  'correction_log_pass': r['Pass']})
audit.sort(key=lambda a: (a['verdict'] == 'PRESERVES', a['example_id'], LEVELS.index(a['target_level'])))
with open(os.path.join(OUT, 'self_containment_audit.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(audit[0]))
    w.writeheader(); w.writerows(audit)
flagged_ids = sorted({int(a['example_id']) for a in audit if a['verdict'].startswith('FLAG')})

# ---------------------------------------------------------------- content provenance (D5)
# Every transformation gets one provenance value, taken from the Correction_Log (no text is changed):
#   authored_scenario           row listed in self_containment_audit.csv: concrete scenario data was written into the row
#                               because its generic source question had none
#   authored_evaluation_object  Evaluate/Analyze row whose log entry says a specific object to judge (a claim, design, proof,
#                               value, ...) was written in ("... is now stated/given/described")
#   source_derived              no authored scenario or object is recorded for the row; it rewrites the source question's own
#                               content (wording may still have been edited for level or notation, see Correction_Log.csv)
AUTHORED = {(int(a['example_id']), a['target_level']): a for a in audit}
OBJECT_CATEGORY = re.compile(r'object/criteria specified|object specified')
OBJECT_INTRODUCED = re.compile(r'\bnow (stated|given|described|specified|included|provided)\b', re.I)
AUTHORED_OBJECT = {}
for r in log:
    k = (int(r['Example ID']), r['Target'])
    if r['Target'] in LEVELS and OBJECT_CATEGORY.search(r['Issue Category']) and OBJECT_INTRODUCED.search(r['Reason']) and k not in AUTHORED:
        AUTHORED_OBJECT[k] = r
def provenance(i, lv):
    if (i, lv) in AUTHORED:
        a = AUTHORED[(i, lv)]
        return 'authored_scenario', {'basis': 'self_containment_audit.csv', 'correction_log_pass': a['correction_log_pass'], 'audit_verdict': a['verdict']}
    if (i, lv) in AUTHORED_OBJECT:
        r = AUTHORED_OBJECT[(i, lv)]
        return 'authored_evaluation_object', {'basis': 'Correction_Log.csv: ' + r['Issue Category'], 'correction_log_pass': r['Pass']}
    return 'source_derived', {'basis': 'no authored scenario or evaluation object recorded in Correction_Log.csv'}

# ---------------------------------------------------------------- deliberately false Evaluate claims (D4)
# Evaluate rows whose Correction_Log entry documents that the stated claim/value/object is deliberately wrong. The claim is
# the object the learner must judge; it is NOT ground truth. 17 rows say "error-detection" explicitly; 45 more say the
# introduced object is "(incorrect)", "faulty", "flawed", "swapped", "infeasible" or has "a sign error".
FALSE_CLAIM_EXPLICIT = {27, 43, 64, 118, 415, 436, 442, 447, 478, 483, 499, 514, 537, 548, 562, 593, 599}
FALSE_CLAIM_OTHER = {76, 100, 103, 116, 124, 128, 132, 147, 156, 172, 178, 192, 211, 221, 226, 231, 234, 235, 241, 243, 244, 250,
                     264, 268, 271, 272, 279, 284, 285, 295, 297, 305, 309, 310, 327, 339, 347, 361, 374, 393, 401, 405, 429, 496, 506}
FALSE_CLAIM = FALSE_CLAIM_EXPLICIT | FALSE_CLAIM_OTHER
assert len(FALSE_CLAIM_EXPLICIT) == 17 and len(FALSE_CLAIM_OTHER) == 45 and not FALSE_CLAIM_EXPLICIT & FALSE_CLAIM_OTHER
FALSE_CLAIM_DETECT = re.compile(r'error[- ]detection|error[- ]spotting|\(incorrect|\bincorrect\b|is invalid|infeasible|not justified|swapped|'
                                r'is actually|\(faulty|specific flawed|\(flawed|sign error', re.I)
FALSE_CLAIM_NOT = {573: 'value was corrected so the claim is right ("to avoid an incorrect answer key")',
                   240: "'faults' are page faults, not a flaw in the claim"}
false_claim_note = {}
for r in log:
    i = int(r['Example ID'])
    if r['Target'] == 'Evaluate' and FALSE_CLAIM_DETECT.search(r['Reason']) and i not in FALSE_CLAIM_NOT:
        false_claim_note.setdefault(i, []).append(r['Reason'])
assert set(false_claim_note) == FALSE_CLAIM, (sorted(set(false_claim_note) ^ FALSE_CLAIM))  # detection and the reviewed list agree
def false_claim(i, lv):
    if lv != 'Evaluate' or i not in FALSE_CLAIM:
        return None
    return {'role': 'object_of_evaluation', 'is_ground_truth': False,
            'evidence_type': 'explicit_error_detection_note' if i in FALSE_CLAIM_EXPLICIT else 'incorrect_or_flawed_object_note',
            'correction_log_note': ' | '.join(false_claim_note[i]),
            'guidance': 'The question deliberately states a wrong or flawed claim for the learner to judge. Never treat the stated claim as a correct answer.'}

# ---------------------------------------------------------------- missing-source notes for label_and_generation_only
missing_note = defaultdict(list)
for u in unres:
    if u['Type'] == 'Genuine missing source material':
        missing_note[int(u['Example ID'])].append(u['Detail'])
def missing_phrases(i):
    m = re.search(r'\(([^)]*)\)', ' '.join(missing_note.get(i, [])))
    return [p.strip() for p in m.group(1).split(',')] if m else []

# ---------------------------------------------------------------- eligibility
REDUNDANT = {}
for u in unres:
    if u['Type'] == 'Redundant duplicate':
        REDUNDANT[int(u['Example ID'])] = int(re.search(r'duplicate of (\d+)', u['Location']).group(1))
eligibility, exclusions = {}, []
for i, x in E.items():
    tu = x['training_use']
    if tu == 'exclude_redundant_duplicate':
        eligibility[i] = 'excluded'
        exclusions.append({'example_id': f'{i:03d}', 'reason': 'redundant_duplicate', 'related_example_id': f'{REDUNDANT[i]:03d}',
                           'detail': x.get('training_use_note', '')})
    elif tu == 'usable_pending_sme_check':
        eligibility[i] = 'excluded'
        exclusions.append({'example_id': f'{i:03d}', 'reason': 'pending_sme_review', 'related_example_id': '', 'detail': x.get('training_use_note', '')})
    elif i in flagged_ids and ROWS_FOR_REVIEW_EXCLUDE_EXAMPLE:
        eligibility[i] = 'excluded'
        why = '; '.join(f"{a['target_level']}: {a['verdict']}" for a in audit if int(a['example_id']) == i and a['verdict'].startswith('FLAG'))
        exclusions.append({'example_id': f'{i:03d}', 'reason': 'self_containment_row_flagged', 'related_example_id': '', 'detail': why + ' (see self_containment_audit.csv)'})
    elif tu == 'usable':
        eligibility[i] = 'usable'
    elif tu == 'label_and_generation_only':
        eligibility[i] = 'label_and_generation_only'
    else:
        raise ValueError((i, tu))
# Row-level missing data that the master flag missed: a 'usable' example whose rows cite 'supplied' material that the
# item does not contain is moved to label_and_generation_only (export-level reclassification; master text is not edited).
SUPPLIED_FALSE_POSITIVES = {(43, 'Analyze'): "'heat supplied' is thermodynamic usage, not a reference to missing data",
                            (428, 'Evaluate'): "'the supplied traceability relationships' are present in the item's embedded table",
                            (478, 'Remember'): "'reactive power supplied by a capacitor' is physics usage, not a reference to missing data"}
reclassified, row_missing = [], defaultdict(list)
for i, x in E.items():
    if eligibility[i] != 'usable':
        continue
    for k, v in [('Source', x['original_question'])] + list(x['transformations'].items()):
        if re.search(r'\bsupplied\b', v, re.I) and (i, k) not in SUPPLIED_FALSE_POSITIVES:
            row_missing[i].append(k)
    if row_missing[i]:
        eligibility[i] = 'label_and_generation_only'
        reclassified.append({'example_id': f'{i:03d}', 'master_training_use': 'usable', 'export_eligibility': 'label_and_generation_only',
                             'rows_citing_absent_material': '; '.join(row_missing[i]),
                             'reason': "These rows refer to 'supplied' material (values, circuit, data) that the item does not contain, so they cannot be answered or checked."})
with open(os.path.join(OUT, 'eligibility_reclassifications.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['example_id', 'master_training_use', 'export_eligibility', 'rows_citing_absent_material', 'reason'])
    w.writeheader(); w.writerows(reclassified)
    for (i, k), why in SUPPLIED_FALSE_POSITIVES.items():
        w.writerow({'example_id': f'{i:03d}', 'master_training_use': 'usable', 'export_eligibility': 'usable (kept)', 'rows_citing_absent_material': k, 'reason': 'Checked, not missing data: ' + why})
with open(os.path.join(OUT, 'exclusions.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['example_id', 'reason', 'related_example_id', 'detail'])
    w.writeheader(); w.writerows(sorted(exclusions, key=lambda r: r['example_id']))

# ---------------------------------------------------------------- 7. integrity checks
problems = []
for i, x in E.items():
    if eligibility[i] == 'excluded':
        continue
    lv = x['original_bloom_level']; tr = x['transformations']
    if lv not in LEVELS: problems.append((i, 'invalid source level'))
    if set(tr) != set(LEVELS) - {lv}: problems.append((i, 'target levels are not the five other levels', sorted(tr)))
    if len(tr) != 5: problems.append((i, 'not exactly five transformations'))
    if not x['original_question'].strip(): problems.append((i, 'empty source question'))
    for k, v in tr.items():
        if not v.strip(): problems.append((i, 'empty transformation', k))
        if norm(v) == norm(x['original_question']): problems.append((i, 'transformation identical to source', k))
        if eligibility[i] == 'usable' and re.search(r'\bsupplied\b', v, re.I) and (i, k) not in SUPPLIED_FALSE_POSITIVES: problems.append((i, 'usable row still cites supplied material', k))
    if len({norm(v) for v in tr.values()}) != len(tr): problems.append((i, 'duplicate transformations within example'))
    if eligibility[i] == 'usable' and re.search(r'\bsupplied\b', x['original_question'], re.I): problems.append((i, 'usable source cites supplied material'))
eligible = sorted(i for i in E if eligibility[i] != 'excluded')

# ---------------------------------------------------------------- 8. leakage / near-duplicate detection and grouping
texts = {i: [E[i]['original_question']] + list(E[i]['transformations'].values()) for i in E}
T = {i: [toks(t) for t in texts[i]] for i in E}
def subtopic(i):
    t = E[i]['topic'] or ''
    t = t.split(':', 1)[1] if ':' in t else ''
    return norm(t)
dsu = DSU(eligible)
pairs = []
exact = defaultdict(list)
for i in eligible:
    for t in texts[i]:
        exact[norm(t)].append(i)
for key, ids in exact.items():
    ids = sorted(set(ids))
    for a, b in combinations(ids, 2):
        pairs.append((a, b, 'identical_question_text', 1.0)); dsu.union(a, b)
for a, b in combinations(eligible, 2):
    jo = jacc(T[a][0], T[b][0])
    if jo >= NEAR_DUP_ORIGINAL_JACCARD:
        pairs.append((a, b, 'similar_source_questions', round(jo, 3))); dsu.union(a, b); continue
    best = max(jacc(x, y) for x in T[a] for y in T[b])
    if best >= NEAR_DUP_ANY_TEXT_JACCARD:
        pairs.append((a, b, 'similar_question_text_across_examples', round(best, 3))); dsu.union(a, b); continue
    if TOPIC_GROUPING and subtopic(a) and subtopic(b):
        sa = {w.rstrip('s') for w in subtopic(a).split() if w not in ('and', 'of', 'the', 'in', 'a')}
        sb = {w.rstrip('s') for w in subtopic(b).split() if w not in ('and', 'of', 'the', 'in', 'a')}
        if sa and sb and (sa <= sb or sb <= sa):
            pairs.append((a, b, 'same_subtopic', round(jo, 3))); dsu.union(a, b)
# the excluded duplicates' kept partners and documented near-duplicates are covered by the rules above; record them explicitly too
KNOWN_NEAR = [(437, 521), (437, 582), (408, 481), (497, 570), (549, 599), (127, 246), (543, 595), (547, 592), (546, 600), (513, 587), (545, 596), (319, 441)]
for a, b in KNOWN_NEAR:
    if a in dsu.p and b in dsu.p:
        if dsu.find(a) != dsu.find(b): pairs.append((a, b, 'documented_near_duplicate', 0.0))
        dsu.union(a, b)
groups = defaultdict(list)
for i in eligible:
    groups[dsu.find(i)].append(i)
gid = {}
for n, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: min(kv[1])), start=1):
    for i in members: gid[i] = f'G{n:03d}'
with open(os.path.join(OUT, 'leakage_report.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['example_a', 'example_b', 'relation', 'similarity', 'group_id', 'title_a', 'title_b'])
    for a, b, rel, s in sorted(pairs):
        w.writerow([f'{a:03d}', f'{b:03d}', rel, s, gid[a], E[a]['discipline'] + (' — ' + E[a]['topic'] if E[a]['topic'] else ''),
                    E[b]['discipline'] + (' — ' + E[b]['topic'] if E[b]['topic'] else '')])

# ---------------------------------------------------------------- 9. grouped, seeded split
rng = random.Random(SEED)
glist = sorted(set(gid.values()))
rng.shuffle(glist)
members_of = defaultdict(list)
for i in eligible: members_of[gid[i]].append(i)
# groups are assigned in seeded random order so multi-example groups can land in any split
total = len(eligible)
count = Counter(); split_of_group = {}
for g in glist:
    deficits = {s: SPLIT_TARGETS[s] * total - count[s] for s in SPLIT_TARGETS}
    s = max(deficits, key=lambda k: (deficits[k], -list(SPLIT_TARGETS).index(k)))
    split_of_group[g] = s; count[s] += len(members_of[g])
split = {i: split_of_group[gid[i]] for i in eligible}
# leakage verification: no group spans splits
for g in glist:
    assert len({split[i] for i in members_of[g]}) == 1
cross = [(a, b) for a, b, *_ in pairs if split[a] != split[b]]
assert not cross, cross

# ---------------------------------------------------------------- exports
def example_record(i):
    x = E[i]
    rec = {'id': x['id'], 'discipline': x['discipline'], 'topic': x['topic'],
           'source_question': x['original_question'], 'source_bloom_level': x['original_bloom_level'],
           'transformations': x['transformations'], 'eligibility': eligibility[i],
           'answer_source_available': eligibility[i] == 'usable', 'group_id': gid.get(i), 'split': split.get(i)}
    if 'source_bloom_level_before_correction' in x: rec['source_bloom_level_before_correction'] = x['source_bloom_level_before_correction']
    if x.get('data_tables'): rec['data_tables'] = x['data_tables']
    if x.get('proposed_clo'): rec['proposed_clo'] = x['proposed_clo']
    if eligibility[i] == 'label_and_generation_only':
        rec['missing_answer_source'] = {'missing': True,
                                        'referenced_but_absent': missing_phrases(i) or [f"supplied material cited in rows: {', '.join(row_missing[i])}"],
                                        'basis': 'master_flag' if E[i]['training_use'] == 'label_and_generation_only' else 'export_reclassification',
                                        'note': 'The source question refers to data not present in the uploaded source PDF. Bloom labels and question wording are valid; '
                                                'answers cannot be derived or checked from the item. Use for Bloom classification and question-generation targets only.'}
    notes = [AUDIT_NOTES[(i, lv)] for lv in LEVELS if (i, lv) in AUDIT_NOTES]
    if notes: rec['review_notes'] = notes
    rec['transformation_provenance'] = {lv: provenance(i, lv)[0] for lv in LEVELS if lv in x['transformations']}
    rec['authored_scenario_levels'] = [lv for lv, p in rec['transformation_provenance'].items() if p == 'authored_scenario']
    rec['deliberately_false_claim_levels'] = [lv for lv in LEVELS if lv in x['transformations'] and false_claim(i, lv)]
    return rec
def transformation_record(i, lv, s):
    x = E[i]
    prov, ev = provenance(i, lv)
    rec = {'record_id': f"{x['id']}-{lv}", 'example_id': x['id'], 'group_id': gid[i], 'split': s, 'eligibility': eligibility[i],
           'answer_source_available': eligibility[i] == 'usable', 'discipline': x['discipline'], 'topic': x['topic'],
           'source_question': x['original_question'], 'source_bloom_level': x['original_bloom_level'],
           'target_bloom_level': lv, 'target_question': x['transformations'][lv],
           'content_provenance': prov, 'provenance_evidence': ev, 'deliberately_false_claim': bool(false_claim(i, lv))}
    if false_claim(i, lv): rec['claim_under_evaluation'] = false_claim(i, lv)
    return rec
usable = [example_record(i) for i in eligible if eligibility[i] == 'usable']
lgo = [example_record(i) for i in eligible if eligibility[i] == 'label_and_generation_only']
wjsonl(os.path.join(OUT, 'usable_examples.jsonl'), usable)
wjsonl(os.path.join(OUT, 'label_and_generation_only_examples.jsonl'), lgo)
os.makedirs(os.path.join(OUT, 'splits'), exist_ok=True)
usable_ids = {i for i in eligible if eligibility[i] == 'usable'}
label_only_ids = {i for i in eligible if eligibility[i] == 'label_and_generation_only'}
answer_generation = {}
manifest ={'seed': SEED, 'targets': SPLIT_TARGETS, 'grouping': {'source_question_jaccard': NEAR_DUP_ORIGINAL_JACCARD,
            'any_question_text_jaccard': NEAR_DUP_ANY_TEXT_JACCARD, 'same_subtopic': TOPIC_GROUPING, 'documented_near_duplicates': True,
            'identical_question_text': True}, 'splits': {}}
for s in SPLIT_TARGETS:
    ids = [i for i in eligible if split[i] == s]
    ex = [example_record(i) for i in ids]
    tr = []; cl = []
    for i in ids:
        x = E[i]
        cl.append({'record_id': f"{x['id']}-source", 'example_id': x['id'], 'group_id': gid[i], 'split': s, 'eligibility': eligibility[i],
                   'question': x['original_question'], 'bloom_level': x['original_bloom_level'], 'role': 'source',
                   'content_provenance': 'source_question', 'deliberately_false_claim': False})
        for lv in LEVELS:
            if lv in x['transformations']:
                tr.append(transformation_record(i, lv, s))
                cl.append({'record_id': f"{x['id']}-{lv}", 'example_id': x['id'], 'group_id': gid[i], 'split': s, 'eligibility': eligibility[i],
                           'question': x['transformations'][lv], 'bloom_level': lv, 'role': 'transformation',
                           'content_provenance': provenance(i, lv)[0], 'deliberately_false_claim': bool(false_claim(i, lv))})
    wjsonl(os.path.join(OUT, 'splits', f'{s}_examples.jsonl'), ex)
    wjsonl(os.path.join(OUT, 'splits', f'{s}_transformation_records.jsonl'), tr)
    wjsonl(os.path.join(OUT, 'splits', f'{s}_bloom_classification_records.jsonl'), cl)
    # D3: strict answer-generation export. Filtered explicitly on eligibility == 'usable' AND membership of the usable ID list;
    # it never relies on the absence of an answer field.
    ag = [r for r in tr if r['eligibility'] == 'usable' and r['answer_source_available'] is True and int(r['example_id']) in usable_ids]
    assert all(eligibility[int(r['example_id'])] == 'usable' for r in ag)
    answer_generation[s] = ag
    os.makedirs(os.path.join(OUT, 'answer_generation'), exist_ok=True)
    wjsonl(os.path.join(OUT, 'answer_generation', f'{s}_transformation_records.jsonl'), ag)
    manifest['splits'][s] = {
        'examples': len(ids), 'examples_usable': sum(eligibility[i] == 'usable' for i in ids),
        'examples_label_and_generation_only': sum(eligibility[i] == 'label_and_generation_only' for i in ids),
        'groups': len({gid[i] for i in ids}), 'transformation_records': len(tr),
        'transformation_records_usable': sum(r['eligibility'] == 'usable' for r in tr),
        'bloom_classification_records': len(cl),
        'source_level_counts': dict(Counter(E[i]['original_bloom_level'] for i in ids)),
        'example_ids': [E[i]['id'] for i in ids]}
manifest['cross_split_related_pairs'] = len(cross)
manifest['groups_total'] = len(glist)
manifest['largest_group_size'] = max(len(v) for v in members_of.values())
json.dump(manifest, open(os.path.join(OUT, 'splits', 'split_manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
ag_manifest = {
    'purpose': 'Strict usable-only export for any task that needs a correct answer (answer generation, solution checking, answer-based reward models).',
    'rule': "A record is included only if eligibility == 'usable', answer_source_available is true, and example_id is in allowed_example_ids. "
            "label_and_generation_only examples are never included in any split.",
    'loader': 'scripts/eduguard_loader.py: load_answer_generation_records() verifies the file hashes, the allow-list and the deny-list, and raises LabelOnlyRecordError otherwise.',
    'allowed_example_ids': sorted(E[i]['id'] for i in usable_ids),
    'denied_example_ids_label_and_generation_only': sorted(E[i]['id'] for i in label_only_ids),
    'splits': {s: {'transformation_records': len(answer_generation[s]),
                   'examples': len({r['example_id'] for r in answer_generation[s]}),
                   'deliberately_false_claim_records': sum(r['deliberately_false_claim'] for r in answer_generation[s]),
                   'file': f'answer_generation/{s}_transformation_records.jsonl',
                   'sha256': sha256(os.path.join(OUT, 'answer_generation', f'{s}_transformation_records.jsonl'))} for s in SPLIT_TARGETS}}
json.dump(ag_manifest, open(os.path.join(OUT, 'answer_generation', 'answer_generation_manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# ---------------------------------------------------------------- integrity report + master preservation
after = {f: sha256(os.path.join(MASTER, f)) for f in MASTER_FILES}
report = {
    'master_files_sha256': before, 'master_unchanged_during_build': before == after,
    'counts': {'master_examples': len(E), 'eligibility': dict(Counter(eligibility.values())),
               'exclusions_by_reason': dict(Counter(e['reason'] for e in exclusions)),
               'reclassified_usable_to_label_and_generation_only': len(reclassified),
               'self_containment_rows_audited': len(audit), 'self_containment_verdicts': dict(Counter(a['verdict'] for a in audit)),
               'self_containment_rows_by_pass': dict(sorted(Counter(a['correction_log_pass'] for a in audit).items())),
               'self_containment_preserves_rows_by_pass': dict(sorted(Counter(a['correction_log_pass'] for a in audit if a['verdict'] == 'PRESERVES').items())),
               'authored_scenario_rows_exported': sum(1 for (i, lv) in AUTHORED if i in eligible),
               'authored_scenario_rows_exported_by_eligibility': dict(Counter(eligibility[i] for (i, lv) in AUTHORED if i in eligible)),
               'authored_scenario_rows_exported_by_pass': dict(sorted(Counter(AUTHORED[k]['correction_log_pass'] for k in AUTHORED if k[0] in eligible).items())),
               'authored_scenario_rows_in_excluded_examples': sorted(f'{i:03d}-{lv}' for (i, lv) in AUTHORED if i not in eligible),
               'authored_evaluation_object_rows_exported': sum(1 for (i, lv) in AUTHORED_OBJECT if i in eligible),
               'deliberately_false_claim_rows_documented': len(FALSE_CLAIM),
               'deliberately_false_claim_rows_exported': sum(1 for i in FALSE_CLAIM if i in eligible),
               'deliberately_false_claim_rows_in_excluded_examples': sorted(f'{i:03d}' for i in FALSE_CLAIM if i not in eligible),
               'answer_generation_transformation_records': {s: len(answer_generation[s]) for s in SPLIT_TARGETS}},
    'integrity_problems': problems,
    'checks': {
        'every_eligible_example_has_one_source_question_with_valid_level': not any(p[1] in ('invalid source level', 'empty source question') for p in problems),
        'every_eligible_example_has_exactly_five_transformations_covering_the_other_levels': not any('five' in p[1] for p in problems),
        'no_empty_or_source_identical_or_repeated_transformations': not any(p[1] in ('empty transformation', 'transformation identical to source', 'duplicate transformations within example') for p in problems),
        'usable_examples_do_not_cite_supplied_material': not any('supplied' in p[1] for p in problems),
        'no_related_pair_split_across_train_validation_test': not cross,
        'excluded_examples_absent_from_all_training_exports': all(eligibility[int(r['id'])] != 'excluded' for r in usable + lgo),
        'answer_generation_exports_contain_only_usable_examples': all(eligibility[int(r['example_id'])] == 'usable' for s in answer_generation for r in answer_generation[s]),
        'answer_generation_exports_contain_no_label_only_examples': not any(int(r['example_id']) in label_only_ids for s in answer_generation for r in answer_generation[s]),
    },
    'leakage': {'related_pairs_found': len(pairs), 'relations': dict(Counter(p[2] for p in pairs)), 'groups': len(glist),
                'multi_example_groups': sum(1 for g in glist if len(members_of[g]) > 1)},
    'not_validated': 'These checks establish data integrity and training eligibility only. They say nothing about model performance.'}
json.dump(report, open(os.path.join(OUT, 'integrity_report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps({k: report[k] for k in ('master_unchanged_during_build', 'counts', 'checks', 'leakage')}, ensure_ascii=False, indent=1))
print('problems:', problems[:10])
print(json.dumps({s: {k: v for k, v in m.items() if k != 'example_ids'} for s, m in manifest['splits'].items()}, indent=1))
print('groups', manifest['groups_total'], 'largest', manifest['largest_group_size'])
