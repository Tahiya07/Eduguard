"""Build the v3 training-ready export from the repaired master (corrected_dataset_v3.json).

Frame: the v2 export in ../training_ready is read (never written). Every split assignment, group ID, eligibility class,
provenance marker, false-claim marker, data table and missing-answer-source note is carried over by example ID; only
question text (and the 252 source label) comes from the v3 master. Output: repaired_v3_20261125/training_ready/.

Run from repaired_v3_20261125:   python -I scripts/make_training_ready_v3.py
"""
import copy, hashlib, json, os, shutil, sys

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # repaired_v3_20261125
FCD = os.path.dirname(HERE)                                                 # final_corrected_dataset
V2 = os.path.join(FCD, 'training_ready')
OUT = os.path.join(HERE, 'training_ready')
SPLITS = ('train', 'validation', 'test')
LEVELS = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
MASTER_FILES = ['corrected_dataset.json', 'Correction_Log.csv', 'Unresolved_Questions.csv', 'Final_Summary.md', 'EduGuard_600_Corrected.pdf']


def jl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def jload(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def wjl(p, recs):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def wjson(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def v2_files():
    out = {}
    for d, _, fs in os.walk(V2):
        if '__pycache__' in d:
            continue
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, V2).replace('\\', '/')] = sha(p)
    return out


before_master = {f: sha(os.path.join(FCD, f)) for f in MASTER_FILES}
before_v2 = v2_files()
M3 = {x['id']: x for x in jload(os.path.join(HERE, 'corrected_dataset_v3.json'))}

# Regenerate the data folders from scratch; tests/ and hand-written scripts (verifier, test suite) are left in place.
for sub in ('splits', 'answer_generation'):
    if os.path.exists(os.path.join(OUT, sub)):
        shutil.rmtree(os.path.join(OUT, sub))

NEW_PROV_EVIDENCE = {'basis': 'pass 3: new transformation after source-label correction; no authored scenario or evaluation object'}


def order(d):
    return {lv: d[lv] for lv in LEVELS if lv in d}


def rebuild_example(x):
    m = M3[x['id']]
    y = copy.deepcopy(x)
    assert m['original_question'] == x['source_question'], x['id']
    new_src = m['original_bloom_level']
    y['source_bloom_level'] = new_src
    if new_src != x['source_bloom_level']:
        y['source_bloom_level_before_correction'] = x['source_bloom_level']
    y['transformations'] = order(m['transformations'])
    prov = {}
    for lv in y['transformations']:
        prov[lv] = x['transformation_provenance'].get(lv, 'source_derived')
    y['transformation_provenance'] = prov
    y['authored_scenario_levels'] = [lv for lv, p in prov.items() if p == 'authored_scenario']
    y['deliberately_false_claim_levels'] = [lv for lv in x['deliberately_false_claim_levels'] if lv in prov]
    y['pass3_rewritten_levels'] = [lv for lv, t in y['transformations'].items() if x['transformations'].get(lv) != t]
    # keep key order stable: put new keys at the end
    return y


ex = {}
for s in SPLITS:
    ex[s] = [rebuild_example(x) for x in jl(os.path.join(V2, 'splits', f'{s}_examples.jsonl'))]
byid = {x['id']: x for s in SPLITS for x in ex[s]}
v2ex = {x['id']: x for s in SPLITS for x in jl(os.path.join(V2, 'splits', f'{s}_examples.jsonl'))}


def rebuild_transformations(s):
    old = {(r['example_id'], r['target_bloom_level']): r for r in jl(os.path.join(V2, 'splits', f'{s}_transformation_records.jsonl'))}
    out = []
    for x in ex[s]:
        for lv, text in x['transformations'].items():
            r = old.get((x['id'], lv))
            if r is None:   # 252-Analyze: new level after the source-label change
                tmpl = next(v for (i, l), v in old.items() if i == x['id'])
                r = copy.deepcopy(tmpl)
                r['record_id'] = f"{x['id']}-{lv}"
                r['target_bloom_level'] = lv
                r['content_provenance'] = 'source_derived'
                r['provenance_evidence'] = dict(NEW_PROV_EVIDENCE)
                r['deliberately_false_claim'] = False
                r.pop('claim_under_evaluation', None)
            else:
                r = copy.deepcopy(r)
            r['source_bloom_level'] = x['source_bloom_level']
            r['target_question'] = text
            assert r['content_provenance'] == x['transformation_provenance'][lv]
            r['pass3_rewritten'] = lv in x['pass3_rewritten_levels']
            out.append(r)
    return out


def rebuild_classification(s, trs):
    old = jl(os.path.join(V2, 'splits', f'{s}_bloom_classification_records.jsonl'))
    src = {r['example_id']: r for r in old if r['role'] == 'source'}
    out = []
    for x in ex[s]:
        r = copy.deepcopy(src[x['id']])
        r['bloom_level'] = x['source_bloom_level']
        r['pass3_rewritten'] = False
        out.append(r)
        for t in (t for t in trs if t['example_id'] == x['id']):
            out.append({'record_id': t['record_id'], 'example_id': t['example_id'], 'group_id': t['group_id'], 'split': s,
                        'eligibility': t['eligibility'], 'question': t['target_question'], 'bloom_level': t['target_bloom_level'],
                        'role': 'transformation', 'content_provenance': t['content_provenance'],
                        'deliberately_false_claim': t['deliberately_false_claim'], 'pass3_rewritten': t['pass3_rewritten']})
    return out


tr, cl, ag = {}, {}, {}
for s in SPLITS:
    tr[s] = rebuild_transformations(s)
    cl[s] = rebuild_classification(s, tr[s])
    ag[s] = [r for r in tr[s] if r['eligibility'] == 'usable']
    wjl(os.path.join(OUT, 'splits', f'{s}_examples.jsonl'), ex[s])
    wjl(os.path.join(OUT, 'splits', f'{s}_transformation_records.jsonl'), tr[s])
    wjl(os.path.join(OUT, 'splits', f'{s}_bloom_classification_records.jsonl'), cl[s])
    wjl(os.path.join(OUT, 'answer_generation', f'{s}_transformation_records.jsonl'), ag[s])

order_ids = [x['id'] for x in jl(os.path.join(V2, 'usable_examples.jsonl'))]
wjl(os.path.join(OUT, 'usable_examples.jsonl'), [byid[i] for i in order_ids])
order_ids = [x['id'] for x in jl(os.path.join(V2, 'label_and_generation_only_examples.jsonl'))]
wjl(os.path.join(OUT, 'label_and_generation_only_examples.jsonl'), [byid[i] for i in order_ids])

# ---------------------------------------------------------------- unchanged registers carried forward
for f in ['exclusions.csv', 'eligibility_reclassifications.csv', 'self_containment_audit.csv']:
    shutil.copy2(os.path.join(V2, f), os.path.join(OUT, f))
os.makedirs(os.path.join(OUT, 'tests'), exist_ok=True)
shutil.copy2(os.path.join(V2, 'tests', 'reference_split_ids.json'), os.path.join(OUT, 'tests', 'reference_split_ids.json'))

# ---------------------------------------------------------------- schema: add pass-3 fields, repair mojibake
def fix(o):
    if isinstance(o, str):
        try:
            return o.encode('cp1252').decode('utf-8') if 'â€' in o else o
        except (UnicodeEncodeError, UnicodeDecodeError):
            return o
    if isinstance(o, list):
        return [fix(v) for v in o]
    if isinstance(o, dict):
        return {k: fix(v) for k, v in o.items()}
    return o


schema = fix(jload(os.path.join(V2, 'schema.json')))
schema['$id'] = 'eduguard-bloom-transformation-training-ready-v3'
schema['title'] = 'EduGuard Bloom transformation dataset v3 (pass-3 repaired) — training-ready record schemas'
D = schema['$defs']
D['example']['properties']['pass3_rewritten_levels'] = {'type': 'array', 'items': {'$ref': '#/$defs/bloomLevel'},
    'description': 'Target levels whose question text differs from v2 after passes 3 and 3.1 (model-assisted, automated checks; not human-verified).'}
D['example']['required'].append('pass3_rewritten_levels')
D['example']['properties']['source_bloom_level_before_correction']['description'] = \
    'Present only when the source label was corrected (068, 090, 152, 182, 226, 241; 252 in pass 3).'
for k in ('transformationRecord', 'bloomClassificationRecord'):
    D[k]['properties']['pass3_rewritten'] = {'type': 'boolean', 'description': 'true when this question text was rewritten in pass 3.'}
    D[k]['required'].append('pass3_rewritten')
wjson(os.path.join(OUT, 'schema.json'), schema)

# ---------------------------------------------------------------- manifests
man = jload(os.path.join(V2, 'splits', 'split_manifest.json'))
man['version'] = 'v3.1 (pass-3 repair + pass-3.1 form revision, 2026-11-25)'
man['derived_from'] = {'file': '../../training_ready/splits/split_manifest.json', 'sha256': before_v2['splits/split_manifest.json'],
                       'note': 'Split, group and eligibility assignments are copied unchanged by example ID.'}
for s in SPLITS:
    e = man['splits'][s]
    cnt = {}
    for x in ex[s]:
        cnt[x['source_bloom_level']] = cnt.get(x['source_bloom_level'], 0) + 1
    e['source_level_counts'] = cnt
    assert e['example_ids'] == [x['id'] for x in ex[s]]
    assert e['transformation_records'] == len(tr[s]) and e['bloom_classification_records'] == len(cl[s])
    assert e['transformation_records_usable'] == len(ag[s])
    e['pass3_rewritten_transformation_records'] = sum(r['pass3_rewritten'] for r in tr[s])
    e['files_sha256'] = {f'{s}_{k}.jsonl': sha(os.path.join(OUT, 'splits', f'{s}_{k}.jsonl'))
                         for k in ('examples', 'transformation_records', 'bloom_classification_records')}
wjson(os.path.join(OUT, 'splits', 'split_manifest.json'), man)

agm = jload(os.path.join(V2, 'answer_generation', 'answer_generation_manifest.json'))
for s in SPLITS:
    p = os.path.join(OUT, 'answer_generation', f'{s}_transformation_records.jsonl')
    agm['splits'][s]['sha256'] = sha(p)
    assert agm['splits'][s]['transformation_records'] == len(ag[s])
agm['version'] = 'v3.1 (pass-3 repair + pass-3.1 form revision, 2026-11-25)'
wjson(os.path.join(OUT, 'answer_generation', 'answer_generation_manifest.json'), agm)

# ---------------------------------------------------------------- scripts
os.makedirs(os.path.join(OUT, 'scripts'), exist_ok=True)
shutil.copy2(os.path.join(V2, 'scripts', 'eduguard_loader.py'), os.path.join(OUT, 'scripts', 'eduguard_loader.py'))
shutil.copy2(os.path.join(V2, 'scripts', 'check_leakage.py'), os.path.join(OUT, 'scripts', 'check_leakage.py'))

# ---------------------------------------------------------------- integrity report
after_master = {f: sha(os.path.join(FCD, f)) for f in MASTER_FILES}
after_v2 = v2_files()
rep = jload(os.path.join(V2, 'integrity_report.json'))
allt = [r for s in SPLITS for r in tr[s]]
rep3 = {
    'version': 'v3.1 (pass-3 repair + pass-3.1 form revision, 2026-11-25)',
    'authoritative_master_files_sha256': after_master,
    'authoritative_master_unchanged_during_build': before_master == after_master == rep['master_files_sha256'],
    'v2_export_unchanged_during_build': before_v2 == after_v2,
    'v2_export_files_sha256': after_v2,
    'v3_master_sha256': {'corrected_dataset_v3.json': sha(os.path.join(HERE, 'corrected_dataset_v3.json')),
                         'Correction_Log_v3.csv': sha(os.path.join(HERE, 'Correction_Log_v3.csv'))},
    'counts': dict(rep['counts']),
    'pass3': {
        'transformation_records_total': len(allt),
        'transformation_records_rewritten': sum(r['pass3_rewritten'] for r in allt),
        'examples_with_any_rewrite': sum(1 for s in SPLITS for x in ex[s] if x['pass3_rewritten_levels']),
        'source_label_changes': {i: {'before': v2ex[i]['source_bloom_level'], 'after': byid[i]['source_bloom_level']}
                                 for i in byid if v2ex[i]['source_bloom_level'] != byid[i]['source_bloom_level']},
        'review_method': 'model-assisted (Claude) rewrite of every eligible example (pass 3) and revision under the relaxed form rule '
                         '(pass 3.1: no per-level verb bank; imperative or interrogative); automated format, structure, '
                         'number-preservation and false-claim-value checks; NOT human-verified',
    },
}
wjson(os.path.join(OUT, 'integrity_report.json'), rep3)
print(json.dumps({k: v for k, v in rep3.items() if k not in ('v2_export_files_sha256', 'counts')}, ensure_ascii=False, indent=1))
print({s: (len(ex[s]), len(tr[s]), len(cl[s]), len(ag[s])) for s in SPLITS})
if not (rep3['authoritative_master_unchanged_during_build'] and rep3['v2_export_unchanged_during_build']):
    sys.exit('integrity failure')
