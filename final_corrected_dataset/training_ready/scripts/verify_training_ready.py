"""Independent post-build verification. Reads the master and the exports; never writes. Exits 1 on any failure.

Run from the folder that holds corrected_dataset.json:
    python -I training_ready/scripts/verify_training_ready.py
"""
import csv, hashlib, json, os, sys
from collections import Counter

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.dirname(OUT)
SPLITS = ('train', 'validation', 'test')
def jl(p):
    with open(p, encoding='utf-8') as f: return [json.loads(l) for l in f if l.strip()]
def jload(p):
    with open(p, encoding='utf-8') as f: return json.load(f)
def rcsv(p):
    with open(p, encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def sha(p):
    with open(p, 'rb') as f: return hashlib.sha256(f.read()).hexdigest()

fails = []
def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))
    if not ok: fails.append(name)

rep = jload(os.path.join(OUT, 'integrity_report.json'))
for f, h in rep['master_files_sha256'].items():
    now = sha(os.path.join(MASTER, f))
    check(f'master unchanged: {f}', now == h, now)
M = {x['id']: x for x in jload(os.path.join(MASTER, 'corrected_dataset.json'))}
U = jl(os.path.join(OUT, 'usable_examples.jsonl')); L = jl(os.path.join(OUT, 'label_and_generation_only_examples.jsonl'))
uid, lid = {x['id'] for x in U}, {x['id'] for x in L}
ex = {s: jl(os.path.join(OUT, 'splits', f'{s}_examples.jsonl')) for s in SPLITS}
tr = {s: jl(os.path.join(OUT, 'splits', f'{s}_transformation_records.jsonl')) for s in SPLITS}
ag = {s: jl(os.path.join(OUT, 'answer_generation', f'{s}_transformation_records.jsonl')) for s in SPLITS}
excl = rcsv(os.path.join(OUT, 'exclusions.csv')); by = Counter(r['reason'] for r in excl)
check('600 examples accounted for', (uid | lid | {r['example_id'] for r in excl}) == set(M) and not (uid | lid) & {r['example_id'] for r in excl}, f'{len(M)}')
check('341 usable / 245 label-only', (len(uid), len(lid)) == (341, 245) and not uid & lid, f'{len(uid)}/{len(lid)}')
check('14 exclusions: 8 duplicate, 3 SME, 3 flagged', dict(by) == {'redundant_duplicate': 8, 'pending_sme_review': 3, 'self_containment_row_flagged': 3}, str(dict(by)))
check('general transformation records 2345/295/290', [len(tr[s]) for s in SPLITS] == [2345, 295, 290], str([len(tr[s]) for s in SPLITS]))
check('answer-generation records 1350/180/175', [len(ag[s]) for s in SPLITS] == [1350, 180, 175], str([len(ag[s]) for s in SPLITS]))
check('answer-generation records are usable only', all(r['eligibility'] == 'usable' and r['example_id'] in uid and r['example_id'] not in lid for s in SPLITS for r in ag[s]))
ref = jload(os.path.join(OUT, 'tests', 'reference_split_ids.json'))['example_ids']
man = jload(os.path.join(OUT, 'splits', 'split_manifest.json'))
check('split ID lists identical to the pre-fix build', all(man['splits'][s]['example_ids'] == ref[s] == [x['id'] for x in ex[s]] for s in SPLITS),
      str({s: len(ref[s]) for s in SPLITS}))
check('question text identical to master', all(x['source_question'] == M[x['id']]['original_question'] and x['transformations'] == M[x['id']]['transformations'] for x in U + L))
allt = [r for s in SPLITS for r in tr[s]]
fc = [r for r in allt if r['deliberately_false_claim']]
check('false-claim markers: 59 exported Evaluate rows, none treated as ground truth',
      len(fc) == 59 and all(r['target_bloom_level'] == 'Evaluate' and r['claim_under_evaluation']['is_ground_truth'] is False for r in fc),
      f"{len(fc)} exported; explicit error-detection {sum(r['claim_under_evaluation']['evidence_type'] == 'explicit_error_detection_note' for r in fc)}")
au = rcsv(os.path.join(OUT, 'self_containment_audit.csv'))
want = {(a['example_id'], a['target_level']) for a in au if a['example_id'] in uid | lid}
got = {(r['example_id'], r['target_bloom_level']) for r in allt if r['content_provenance'] == 'authored_scenario'}
check('authored-scenario markers match the audit (144 exported rows)', got == want and len(got) == 144, f'{len(got)}')
print('provenance counts:', dict(Counter(r['content_provenance'] for r in allt)))
print('false claims by split (all / usable):', {s: (sum(r['deliberately_false_claim'] for r in tr[s]), sum(r['deliberately_false_claim'] for r in ag[s])) for s in SPLITS})
print('answer-generation file sha256:', {s: e['sha256'] for s, e in jload(os.path.join(OUT, 'answer_generation', 'answer_generation_manifest.json'))['splits'].items()})
print('RESULT:', 'all checks passed' if not fails else f'{len(fails)} failed: {fails}')
sys.exit(1 if fails else 0)
