"""Build the EduGuard training-ready exports from the corrected master dataset.

Reads (never writes) the master files one directory above `training_ready/`:
  corrected_dataset.json, Correction_Log.csv, Unresolved_Questions.csv
Writes everything into `training_ready/`. Standard library only; deterministic.

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
    return rec
usable = [example_record(i) for i in eligible if eligibility[i] == 'usable']
lgo = [example_record(i) for i in eligible if eligibility[i] == 'label_and_generation_only']
wjsonl(os.path.join(OUT, 'usable_examples.jsonl'), usable)
wjsonl(os.path.join(OUT, 'label_and_generation_only_examples.jsonl'), lgo)
os.makedirs(os.path.join(OUT, 'splits'), exist_ok=True)
manifest = {'seed': SEED, 'targets': SPLIT_TARGETS, 'grouping': {'source_question_jaccard': NEAR_DUP_ORIGINAL_JACCARD,
            'any_question_text_jaccard': NEAR_DUP_ANY_TEXT_JACCARD, 'same_subtopic': TOPIC_GROUPING, 'documented_near_duplicates': True,
            'identical_question_text': True}, 'splits': {}}
for s in SPLIT_TARGETS:
    ids = [i for i in eligible if split[i] == s]
    ex = [example_record(i) for i in ids]
    tr = []; cl = []
    for i in ids:
        x = E[i]
        cl.append({'record_id': f"{x['id']}-source", 'example_id': x['id'], 'group_id': gid[i], 'split': s, 'eligibility': eligibility[i],
                   'question': x['original_question'], 'bloom_level': x['original_bloom_level'], 'role': 'source'})
        for lv in LEVELS:
            if lv in x['transformations']:
                tr.append({'record_id': f"{x['id']}-{lv}", 'example_id': x['id'], 'group_id': gid[i], 'split': s, 'eligibility': eligibility[i],
                           'answer_source_available': eligibility[i] == 'usable', 'discipline': x['discipline'], 'topic': x['topic'],
                           'source_question': x['original_question'], 'source_bloom_level': x['original_bloom_level'],
                           'target_bloom_level': lv, 'target_question': x['transformations'][lv]})
                cl.append({'record_id': f"{x['id']}-{lv}", 'example_id': x['id'], 'group_id': gid[i], 'split': s, 'eligibility': eligibility[i],
                           'question': x['transformations'][lv], 'bloom_level': lv, 'role': 'transformation'})
    wjsonl(os.path.join(OUT, 'splits', f'{s}_examples.jsonl'), ex)
    wjsonl(os.path.join(OUT, 'splits', f'{s}_transformation_records.jsonl'), tr)
    wjsonl(os.path.join(OUT, 'splits', f'{s}_bloom_classification_records.jsonl'), cl)
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

# ---------------------------------------------------------------- integrity report + master preservation
after = {f: sha256(os.path.join(MASTER, f)) for f in MASTER_FILES}
report = {
    'master_files_sha256': before, 'master_unchanged_during_build': before == after,
    'counts': {'master_examples': len(E), 'eligibility': dict(Counter(eligibility.values())),
               'exclusions_by_reason': dict(Counter(e['reason'] for e in exclusions)),
               'reclassified_usable_to_label_and_generation_only': len(reclassified),
               'self_containment_rows_audited': len(audit), 'self_containment_verdicts': dict(Counter(a['verdict'] for a in audit))},
    'integrity_problems': problems,
    'checks': {
        'every_eligible_example_has_one_source_question_with_valid_level': not any(p[1] in ('invalid source level', 'empty source question') for p in problems),
        'every_eligible_example_has_exactly_five_transformations_covering_the_other_levels': not any('five' in p[1] for p in problems),
        'no_empty_or_source_identical_or_repeated_transformations': not any(p[1] in ('empty transformation', 'transformation identical to source', 'duplicate transformations within example') for p in problems),
        'usable_examples_do_not_cite_supplied_material': not any('supplied' in p[1] for p in problems),
        'no_related_pair_split_across_train_validation_test': not cross,
        'excluded_examples_absent_from_all_training_exports': all(eligibility[int(r['id'])] != 'excluded' for r in usable + lgo),
    },
    'leakage': {'related_pairs_found': len(pairs), 'relations': dict(Counter(p[2] for p in pairs)), 'groups': len(glist),
                'multi_example_groups': sum(1 for g in glist if len(members_of[g]) > 1)},
    'not_validated': 'These checks establish data integrity and training eligibility only. They say nothing about model performance.'}
json.dump(report, open(os.path.join(OUT, 'integrity_report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps({k: report[k] for k in ('master_unchanged_during_build', 'counts', 'checks', 'leakage')}, ensure_ascii=False, indent=1))
print('problems:', problems[:10])
print(json.dumps({s: {k: v for k, v in m.items() if k != 'example_ids'} for s, m in manifest['splits'].items()}, indent=1))
print('groups', manifest['groups_total'], 'largest', manifest['largest_group_size'])
