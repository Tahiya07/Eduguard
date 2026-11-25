"""Print eligible examples for review: python -I dump_batch.py <batch_no> [batch_size]. Read-only."""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
LV = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
ex = {}
for s in ['train', 'validation', 'test']:
    for l in open(os.path.join(ROOT, 'training_ready', 'splits', f'{s}_examples.jsonl'), encoding='utf-8'):
        x = json.loads(l); ex[x['id']] = x
ids = sorted(ex)
BANK = json.load(open(os.path.join(os.path.dirname(HERE), 'work', 'obe_verb_bank.json'), encoding='utf-8'))
import re
WH = {'What', 'Why', 'How', 'Which', 'Who', 'When', 'Where', 'Is', 'Are', 'Does', 'Do', 'Can', 'Should'}
def ok(lv, q):
    # OBE: no question marks or wh-openings; the instruction sentence must start with a bank verb of this level.
    # Declarative scenario sentences may precede it (user directive 2026-11-25).
    if '?' in q:
        return False
    sents = re.split(r"(?:(?<=[.!])|(?<=[.!]['\"’”]))\s+(?=[A-Z])", q.strip())
    if sents[0].split()[0] in WH:
        return False
    return any(s_.split()[0].strip(',.:;') in BANK[lv] for s_ in sents if s_.split())
b, n = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 25
for i in ids[(b - 1) * n: b * n]:
    x = ex[i]
    tag = 'U' if x['eligibility'] == 'usable' else 'L'
    print(f"=== {i} [{x['split']}/{tag}] {x['discipline']} | {x['topic']}")
    print(f"  SRC({x['source_bloom_level']}): {x['source_question']}")
    for lv in LV:
        if lv in x['transformations']:
            p = {'authored_scenario': 'A', 'authored_evaluation_object': 'O', 'source_derived': '-'}[x['transformation_provenance'][lv]]
            f = 'F' if lv in x['deliberately_false_claim_levels'] else ' '
            bad = '' if ok(lv, x['transformations'][lv]) else '!'
            print(f"  {bad}{lv[:3]}[{p}{f}]: {x['transformations'][lv]}")
print(f'# batch {b}: ids {ids[(b-1)*n]}..{ids[min(b*n, len(ids))-1]} of {len(ids)}; batches total {-(-len(ids)//n)}')
