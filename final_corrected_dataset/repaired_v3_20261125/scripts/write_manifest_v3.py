"""Write MANIFEST_v3.json and SHA256SUMS_v3.txt for every deliverable in repaired_v3_20261125 (read-only otherwise).

Run last, from repaired_v3_20261125:  python -I scripts/write_manifest_v3.py
"""
import hashlib, json, os, sys

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FCD = os.path.dirname(HERE)
SKIP_DIRS = {'__pycache__'}
SKIP_FILES = {'MANIFEST_v3.json', 'SHA256SUMS_v3.txt'}


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


files = {}
for d, ds, fs in os.walk(HERE):
    ds[:] = sorted(x for x in ds if x not in SKIP_DIRS)
    for f in sorted(fs):
        if f in SKIP_FILES or (os.path.basename(d) == 'work' and f.endswith('_dump.txt')):
            continue
        p = os.path.join(d, f)
        files[os.path.relpath(p, HERE).replace('\\', '/')] = {'sha256': sha(p), 'bytes': os.path.getsize(p)}

ir = json.load(open(os.path.join(HERE, 'training_ready', 'integrity_report.json'), encoding='utf-8'))
for f, h in ir['authoritative_master_files_sha256'].items():
    assert sha(os.path.join(FCD, f)) == h, f'authoritative master changed: {f}'

role = lambda p: ('repaired master' if p in ('corrected_dataset_v3.json', 'Correction_Log_v3.csv') else
                  'exclusion register' if p.startswith('EXCLUSION') else
                  'training-ready export' if p.startswith('training_ready/') else
                  'challenge set (evaluation only)' if p.startswith('challenge_set/') else
                  'report' if p.startswith('reports/') else 'diagnostic audit' if p.startswith('audits/') else
                  'build script' if p.startswith('scripts/') else 'repair working files (patches, rules, verb banks)')
man = {
    'dataset': 'EduGuard Bloom transformation dataset',
    'version': 'v3.1 (pass-3 repair + pass-3.1 form revision)', 'date': '2026-11-25',
    'derived_from': {'authoritative_master': ir['authoritative_master_files_sha256'],
                     'v2_export_split_manifest_sha256': ir['v2_export_files_sha256']['splits/split_manifest.json'],
                     'reference_split_ids_sha256': ir['v2_export_files_sha256']['tests/reference_split_ids.json']},
    'review_status': {'corrections': 'model-assisted (Claude); not human-verified',
                      'validation': 'automated (scripts and unit tests)',
                      'human_verified_corrections': 0},
    'counts': {'examples_master': 600, 'examples_exported': 586, 'usable': 341, 'label_and_generation_only': 245, 'excluded': 14,
               'transformation_records': 2930, 'pass3_rewritten_records': ir['pass3']['transformation_records_rewritten'],
               'source_label_changes': ir['pass3']['source_label_changes']},
    'not_done': ['no model training', 'no GPU use', 'no model-configuration change', 'no change to the authoritative master or the v2 export',
                 'no git commit or push'],
    'files': {p: {**v, 'role': role(p)} for p, v in files.items()},
}
with open(os.path.join(HERE, 'MANIFEST_v3.json'), 'w', encoding='utf-8', newline='\n') as f:
    json.dump(man, f, ensure_ascii=False, indent=2)
with open(os.path.join(HERE, 'SHA256SUMS_v3.txt'), 'w', encoding='utf-8', newline='\n') as f:
    for p, v in files.items():
        f.write(f"{v['sha256']}  {p}\n")
print(len(files), 'files listed')
