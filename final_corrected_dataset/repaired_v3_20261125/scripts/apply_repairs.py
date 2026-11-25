"""Apply pass-3 patch files to a copy of the master and validate them (standard library only).

Reads (never writes) ../../corrected_dataset.json and ../../Correction_Log.csv.
Writes, inside repaired_v3_20261125/:
  corrected_dataset_v3.json        repaired master (all 600 examples; excluded examples unchanged)
  Correction_Log_v3.csv            full correction history: passes 1-2 unchanged + pass-3 rows
  work/apply_report.json           coverage, validation results, change counts
Exit code 1 if any validation fails.
Run: python -I scripts/apply_repairs.py   (from repaired_v3_20261125/)
"""
import csv, glob, hashlib, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); V3 = os.path.dirname(HERE); ROOT = os.path.dirname(V3)
LV = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
BANK = json.load(open(os.path.join(V3, 'work', 'obe_verb_bank.json'), encoding='utf-8'))
WH = {'What', 'Why', 'How', 'Which', 'Who', 'When', 'Where', 'Is', 'Are', 'Does', 'Do', 'Can', 'Should'}
SPLIT_SENT = re.compile(r"(?:(?<=[.!])|(?<=[.!]['\"’”]))\s+(?=[A-Z])")
NUM = re.compile(r'(?<![A-Za-z_])\d+(?:[.,]\d+)*')
# Reviewed exceptions: numbers that are method constants or quantities derived directly from stated values,
# not new data. Each was checked individually during pass 3.
ACCEPTED_NUMBERS = {
    ('285', 'Create'): ({'2'}, "'2-to-1 multiplexer' is a device name in a Create design task, not data"),
    ('330', 'Analyze'): ({'2'}, '2^k is the shift-multiplication identity'),
    ('446', 'Analyze'): ({'0'}, 'pole at s = 0 is read directly from G(s)=K/(s(s+2)(s+4))'),
    ('472', 'Analyze'): ({'1'}, 'exponent 1/2 and the -1 follow from the stated two-year period'),
    ('512', 'Analyze'): ({'16', '8'}, '2n = 8 states and 2^4 = 16 possible states follow from the stated 4 bits'),
    ('551', 'Analyze'): ({'4'}, "Simpson's one-third rule weights 1, 4, 1 are method constants"),
}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def obe_ok(level, q):
    """Pass 3.1 format check (user directive, 2026-11-25): the per-level verb-bank rule is REMOVED.
    Imperative and interrogative forms are both allowed, and any verb may be used at any level when the cognitive
    demand justifies it. Only basic well-formedness is checked automatically: a non-empty task
    of at least two words that is not visibly truncated. Bloom alignment is judged by review, not by verb.
    Returns (ok, form) where form is 'interrogative', 'imperative' or 'scenario+task'."""
    t = q.strip()
    if len(t.split()) < 2:
        return False, 'too short'
    if t.endswith((',', '-', '(')) or t.split()[-1] in ('and', 'or', 'the', 'a', 'of'):
        return False, 'truncated: ' + t[-20:]
    sents = [x for x in re.split(r'(?<=[.!?])\s+(?=[A-Z])', t) if x.split()]
    if '?' in t:
        return True, 'interrogative'
    return True, ('scenario+task' if len(sents) > 1 and sents[0].split()[0] in ('A', 'An', 'The', 'In') else 'imperative')


def parse_patches():
    entries, reviewed, problems = [], set(), []
    for path in sorted(glob.glob(os.path.join(V3, 'work', 'patches', 'b*.txt'))) + sorted(glob.glob(os.path.join(V3, 'work', 'patches', 's*.txt'))) + sorted(glob.glob(os.path.join(V3, 'work', 'patches', 'r*.txt'))):
        lines = open(path, encoding='utf-8').read().splitlines()
        if not lines or not lines[0].startswith('#reviewed '):
            problems.append(f'{path}: missing #reviewed header'); continue
        reviewed |= {x.strip() for x in lines[0][len('#reviewed '):].split(',') if x.strip() and x.strip() != 'none'}
        i = 1
        while i < len(lines):
            if not lines[i].strip():
                i += 1; continue
            if not lines[i].startswith('@@ '):
                problems.append(f'{os.path.basename(path)}:{i+1}: expected @@ header'); i += 1; continue
            parts = lines[i][3:].split('|', 3)
            if len(parts) != 4 or i + 1 >= len(lines):
                problems.append(f'{os.path.basename(path)}:{i+1}: malformed header'); i += 2; continue
            eid, level, cat, reason = (p.strip() for p in parts)
            entries.append({'file': os.path.basename(path), 'line': i + 1, 'id': eid, 'level': level,
                            'category': cat, 'reason': reason, 'text': lines[i + 1].strip()})
            i += 2
    return entries, reviewed, problems


def main():
    master_path = os.path.join(ROOT, 'corrected_dataset.json')
    log_path = os.path.join(ROOT, 'Correction_Log.csv')
    master_hash_before = {p: sha(os.path.join(ROOT, p)) for p in ('corrected_dataset.json', 'Correction_Log.csv')}
    master = json.load(open(master_path, encoding='utf-8'))
    E = {x['id']: x for x in master}
    eligible = {}
    for s in ('train', 'validation', 'test'):
        for l in open(os.path.join(ROOT, 'training_ready', 'splits', f'{s}_examples.jsonl'), encoding='utf-8'):
            x = json.loads(l); eligible[x['id']] = x
    entries, reviewed, problems = parse_patches()

    # coverage
    missing_review = sorted(set(eligible) - reviewed)
    extra_review = sorted(reviewed - set(eligible))
    if missing_review: problems.append(f'eligible examples not reviewed: {missing_review}')
    if extra_review: problems.append(f'reviewed ids that are not eligible: {extra_review}')

    # duplicate patch targets
    # sweep files (s*.txt) supersede first-pass rewrites and pass-3.1 revisions (r*.txt) supersede both;
    # duplicates within one series are errors
    for series in ('b', 's', 'r'):
        seen = Counter((e['id'], e['level']) for e in entries if e['file'].startswith(series))
        dups = [k for k, c in seen.items() if c > 1]
        if dups: problems.append(f'duplicate patch targets within {series}-series: {dups}')
    last = {}
    for e in entries: last[(e['id'], e['level'])] = e
    superseded = [e for e in entries if last[(e['id'], e['level'])] is not e]
    entries = [e for e in entries if last[(e['id'], e['level'])] is e]

    repaired = json.loads(json.dumps(master))
    R = {x['id']: x for x in repaired}
    log_rows, label_changes = [], []
    for e in entries:
        x = R.get(e['id'])
        if x is None or e['id'] not in eligible:
            problems.append(f"{e['file']}:{e['line']}: id {e['id']} not an eligible example"); continue
        if e['level'] == 'SOURCE_LABEL':
            new = e['category']
            if new not in LV: problems.append(f"{e['file']}:{e['line']}: bad label {new}"); continue
            old = x['original_bloom_level']
            x['source_bloom_level_before_correction'] = old
            x['original_bloom_level'] = new
            label_changes.append({'id': e['id'], 'from': old, 'to': new, 'reason': e['reason']})
            log_rows.append({'Example ID': e['id'], 'Target': 'Source label', 'Original Wording': old, 'Corrected Wording': new,
                             'Issue Category': 'Source label correction', 'Reason': 'Pass 3: ' + e['reason'], 'Pass': '3'})
            continue
        if e['level'] not in LV:
            problems.append(f"{e['file']}:{e['line']}: bad level {e['level']}"); continue
        if e['category'] == 'DROP':
            old = x['transformations'].pop(e['level'], None)
            if old is None: problems.append(f"{e['file']}:{e['line']}: DROP of absent row {e['id']}-{e['level']}")
            log_rows.append({'Example ID': e['id'], 'Target': e['level'], 'Original Wording': old or '', 'Corrected Wording': '',
                             'Issue Category': 'Row removed after source relabel', 'Reason': 'Pass 3: ' + e['reason'], 'Pass': '3'})
            continue
        old = x['transformations'].get(e['level'])
        if old is None and e['category'] != 'new_row_after_relabel':
            problems.append(f"{e['file']}:{e['line']}: {e['id']}-{e['level']} does not exist"); continue
        old = E[e['id']]['transformations'].get(e['level']) if e['id'] in E else old
        if old is not None and old.strip() == e['text']:
            if e['file'].startswith('r'):   # pass-3.1 revision that restores the pre-pass-3 wording
                x['transformations'][e['level']] = e['text']
                log_rows.append({'Example ID': e['id'], 'Target': e['level'], 'Original Wording': old, 'Corrected Wording': e['text'],
                                 'Issue Category': 'Pass 3.1 – reverted to pre-pass-3 wording', 'Reason': 'Pass 3.1: ' + e['reason'], 'Pass': '3.1'})
                continue
            problems.append(f"{e['file']}:{e['line']}: patch identical to existing text"); continue
        x['transformations'][e['level']] = e['text']
        log_rows.append({'Example ID': e['id'], 'Target': e['level'], 'Original Wording': old or '', 'Corrected Wording': e['text'],
                         'Issue Category': ('Pass 3.1 – ' if e['file'].startswith('r') else 'Pass 3 – ') + e['category'].replace('_', ' '),
                         'Reason': ('Pass 3.1: ' if e['file'].startswith('r') else 'Pass 3: ') + e['reason'],
                         'Pass': '3.1' if e['file'].startswith('r') else '3'})

    # structural + OBE + content-preservation validation on every eligible example
    obe_fail, struct_fail, number_flags, false_claim_flags, accepted_number_exceptions = [], [], [], [], []
    verb_use = defaultdict(Counter)
    patched = {(e['id'], e['level']) for e in entries if e['level'] in LV and e['category'] != 'DROP'}
    for eid in sorted(eligible):
        x = R[eid]; ex_v2 = eligible[eid]
        lv = x['original_bloom_level']; tr = x['transformations']
        if set(tr) != set(LV) - {lv} or len(tr) != 5 or any(not v.strip() for v in tr.values()):
            struct_fail.append((eid, sorted(tr)))
        if len({re.sub(r'\W+', ' ', v.lower()) for v in tr.values()}) != 5:
            struct_fail.append((eid, 'duplicate transformations'))
        pool = ' '.join([E[eid]['original_question']] + list(E[eid]['transformations'].values()))
        pool_nums = set(NUM.findall(pool))
        for level, q in tr.items():
            ok, verb = obe_ok(level, q)
            if not ok: obe_fail.append((eid, level, verb, q[:120]))
            else: verb_use[level][verb] += 1
            if (eid, level) in patched:
                new_nums = set(NUM.findall(q)) - pool_nums
                accepted = ACCEPTED_NUMBERS.get((eid, level))
                if accepted and new_nums <= accepted[0]:
                    accepted_number_exceptions.append((eid, level, sorted(new_nums), accepted[1]))
                elif new_nums:
                    number_flags.append((eid, level, sorted(new_nums), q[:140]))
        if 'Evaluate' in ex_v2['deliberately_false_claim_levels'] and (eid, 'Evaluate') in patched:
            lost = set(NUM.findall(E[eid]['transformations']['Evaluate'])) - set(NUM.findall(tr['Evaluate']))
            if lost: false_claim_flags.append((eid, sorted(lost)))

    if obe_fail: problems.append(f'{len(obe_fail)} rows fail the format check')
    if struct_fail: problems.append(f'{len(struct_fail)} structural failures')
    if number_flags: problems.append(f'{len(number_flags)} patched rows introduce numbers absent from the example')
    if false_claim_flags: problems.append(f'{len(false_claim_flags)} deliberately false Evaluate rows lost a stated value')

    master_hash_after = {p: sha(os.path.join(ROOT, p)) for p in master_hash_before}
    if master_hash_before != master_hash_after: problems.append('MASTER CHANGED DURING APPLY')

    report = {'patch_entries': len(entries), 'superseded_first_pass_entries': len(superseded), 'reviewed_examples': len(reviewed & set(eligible)), 'eligible_examples': len(eligible),
              'rows_rewritten': sum(1 for e in entries if e['level'] in LV and e['category'] not in ('DROP',)),
              'rows_dropped': sum(1 for e in entries if e['category'] == 'DROP'),
              'source_label_changes': label_changes, 'categories': dict(Counter(e['category'] for e in entries).most_common()),
              'rows_rewritten_by_level': dict(Counter(e['level'] for e in entries if e['level'] in LV and e['category'] != 'DROP')),
              'obe_failures': obe_fail, 'structural_failures': struct_fail, 'number_flags': number_flags,
              'false_claim_value_loss': false_claim_flags, 'accepted_number_exceptions': accepted_number_exceptions, 'question_forms_by_level': {k: dict(v.most_common()) for k, v in verb_use.items()},
              'master_sha256': master_hash_after, 'problems': problems}
    json.dump(report, open(os.path.join(V3, 'work', 'apply_report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if '--write' in sys.argv and not problems:
        json.dump(repaired, open(os.path.join(V3, 'corrected_dataset_v3.json'), 'w', encoding='utf-8', newline='\n'), ensure_ascii=False, indent=2)
        orig_log = list(csv.DictReader(open(log_path, encoding='utf-8-sig')))
        with open(os.path.join(V3, 'Correction_Log_v3.csv'), 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(orig_log[0].keys()) + ['Review Method'])
            w.writeheader()
            for r in orig_log:
                w.writerow(dict(r, **{'Review Method': 'passes 1-2: model-assisted (Claude) with script checks; not human-verified'}))
            for r in log_rows:
                method = ('pass 3.1: model-assisted (Claude) revision under the relaxed format rule (no per-level verb bank; imperative or interrogative); automated format, structure, and number-preservation checks; not human-verified'
                          if r['Pass'] == '3.1' else
                          'pass 3: model-assisted (Claude) rewrite with automated OBE, structure, and number-preservation checks; not human-verified')
                w.writerow(dict(r, **{'Review Method': method}))
    print(json.dumps({k: v for k, v in report.items() if k in ('patch_entries', 'reviewed_examples', 'rows_rewritten', 'rows_dropped', 'source_label_changes', 'rows_rewritten_by_level', 'problems')}, ensure_ascii=False, indent=1))
    for k in ('obe_failures', 'structural_failures', 'number_flags', 'false_claim_value_loss'):
        for item in report[k][:40]:
            print(k, item)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
