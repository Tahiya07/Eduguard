"""Automated checks for the EduGuard training-ready exports (standard library only).

Run from the folder that holds corrected_dataset.json:
    python -I -m unittest discover -s training_ready/tests -v
"""
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)                 # .../training_ready
MASTER = os.path.dirname(ROOT)                # .../final_corrected_dataset
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import eduguard_loader as L  # noqa: E402

SPLITS = ('train', 'validation', 'test')
LEVELS = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
EXPECTED_ANSWER_RECORDS = {'train': 1350, 'validation': 180, 'test': 175}
EXPECTED_GENERAL_RECORDS = {'train': 2345, 'validation': 295, 'test': 290}
DUPLICATES = {'156': '027', '262': '187', '502': '421', '509': '472', '517': '458', '530': '417', '578': '512', '597': '457'}
SME = {'429', '445', '477'}
FLAGGED = {'158', '319', '393'}
FALSE_CLAIM_EXPLICIT = {'027', '043', '064', '118', '415', '436', '442', '447', '478', '483', '499', '514', '537', '548', '562', '593', '599'}
FALSE_CLAIM_EXCLUDED = {'156', '393', '429'}


def jl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def rcsv(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def jload(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ------------------------------------------------------------ minimal JSON Schema validator (subset used by schema.json)
def validate(inst, sch, root, path='$'):
    if '$ref' in sch:
        return validate(inst, root['$defs'][sch['$ref'].split('/')[-1]], root, path)
    errs = []
    t = sch.get('type')
    types = {'object': dict, 'array': list, 'string': str, 'boolean': bool}
    if t and not (isinstance(inst, types[t]) and not (t != 'boolean' and isinstance(inst, bool))):
        return [f'{path}: expected {t}']
    if 'const' in sch and inst != sch['const']: errs.append(f'{path}: != const')
    if 'enum' in sch and inst not in sch['enum']: errs.append(f'{path}: {inst!r} not in enum')
    if isinstance(inst, str):
        if 'pattern' in sch and not re.search(sch['pattern'], inst): errs.append(f'{path}: pattern')
        if len(inst) < sch.get('minLength', 0): errs.append(f'{path}: minLength')
    if isinstance(inst, list):
        if len(inst) > sch.get('maxItems', 10 ** 9): errs.append(f'{path}: maxItems')
        for n, v in enumerate(inst):
            if 'items' in sch: errs += validate(v, sch['items'], root, f'{path}[{n}]')
    if isinstance(inst, dict):
        for k in sch.get('required', []):
            if k not in inst: errs.append(f'{path}: missing {k}')
        if not sch.get('minProperties', 0) <= len(inst) <= sch.get('maxProperties', 10 ** 9): errs.append(f'{path}: property count')
        props = sch.get('properties', {})
        for k, v in inst.items():
            if 'propertyNames' in sch: errs += validate(k, sch['propertyNames'], root, f'{path}.<{k}>')
            if k in props:
                errs += validate(v, props[k], root, f'{path}.{k}')
            elif sch.get('additionalProperties') is False:
                errs.append(f'{path}: unexpected {k}')
            elif isinstance(sch.get('additionalProperties'), dict):
                errs += validate(v, sch['additionalProperties'], root, f'{path}.{k}')
    return errs


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = {x['id']: x for x in jload(os.path.join(MASTER, 'corrected_dataset.json'))}
        cls.usable = jl(os.path.join(ROOT, 'usable_examples.jsonl'))
        cls.label = jl(os.path.join(ROOT, 'label_and_generation_only_examples.jsonl'))
        cls.ex = {s: jl(os.path.join(ROOT, 'splits', f'{s}_examples.jsonl')) for s in SPLITS}
        cls.tr = {s: jl(os.path.join(ROOT, 'splits', f'{s}_transformation_records.jsonl')) for s in SPLITS}
        cls.cl = {s: jl(os.path.join(ROOT, 'splits', f'{s}_bloom_classification_records.jsonl')) for s in SPLITS}
        cls.ag = {s: jl(os.path.join(ROOT, 'answer_generation', f'{s}_transformation_records.jsonl')) for s in SPLITS}
        cls.uid = {x['id'] for x in cls.usable}
        cls.lid = {x['id'] for x in cls.label}
        cls.excl = rcsv(os.path.join(ROOT, 'exclusions.csv'))
        cls.audit = rcsv(os.path.join(ROOT, 'self_containment_audit.csv'))
        cls.log = rcsv(os.path.join(MASTER, 'Correction_Log.csv'))
        cls.report = jload(os.path.join(ROOT, 'integrity_report.json'))


class TestMasterAndCounts(Base):
    def test_master_hashes_unchanged(self):
        for f, h in self.report['master_files_sha256'].items():
            self.assertEqual(sha256(os.path.join(MASTER, f)), h, f)
        self.assertTrue(self.report['master_unchanged_during_build'])

    def test_all_600_accounted_for(self):
        exported = self.uid | self.lid
        excluded = {r['example_id'] for r in self.excl}
        self.assertEqual(len(self.master), 600)
        self.assertFalse(exported & excluded)
        self.assertEqual(exported | excluded, set(self.master))
        self.assertEqual({x['id'] for s in SPLITS for x in self.ex[s]}, exported)

    def test_category_counts(self):
        self.assertEqual((len(self.uid), len(self.lid)), (341, 245))
        self.assertFalse(self.uid & self.lid)
        by = {}
        for r in self.excl: by.setdefault(r['reason'], set()).add(r['example_id'])
        self.assertEqual(by['redundant_duplicate'], set(DUPLICATES))
        self.assertEqual(by['pending_sme_review'], SME)
        self.assertEqual(by['self_containment_row_flagged'], FLAGGED)
        self.assertEqual(len(self.excl), 14)

    def test_duplicates_excluded_partners_eligible(self):
        for dup, partner in DUPLICATES.items():
            self.assertNotIn(dup, self.uid | self.lid)
            self.assertIn(partner, self.uid | self.lid)
            self.assertEqual(self.master[dup]['training_use'], 'exclude_redundant_duplicate')

    def test_five_transformations_and_text_unchanged(self):
        for x in self.usable + self.label:
            m = self.master[x['id']]
            self.assertEqual(set(x['transformations']), set(LEVELS) - {x['source_bloom_level']})
            self.assertEqual(x['transformations'], m['transformations'])
            self.assertEqual(x['source_question'], m['original_question'])
            self.assertEqual(x['source_bloom_level'], m['original_bloom_level'])
        self.assertEqual(sum(len(x['transformations']) for x in self.usable), 1705)

    def test_general_record_counts(self):
        for s in SPLITS:
            self.assertEqual(len(self.tr[s]), EXPECTED_GENERAL_RECORDS[s], s)
            self.assertEqual(len(self.cl[s]), 6 * len(self.ex[s]), s)
            for r in self.tr[s]:
                self.assertEqual(r['target_question'], self.master[r['example_id']]['transformations'][r['target_bloom_level']])

    def test_split_ids_identical_to_reference(self):
        ref = jload(os.path.join(TESTS, 'reference_split_ids.json'))
        man = jload(os.path.join(ROOT, 'splits', 'split_manifest.json'))
        for s in SPLITS:
            self.assertEqual(man['splits'][s]['example_ids'], ref['example_ids'][s], s)
            self.assertEqual([x['id'] for x in self.ex[s]], ref['example_ids'][s], s)
            self.assertTrue(all(x['split'] == s for x in self.ex[s] + self.tr[s] + self.cl[s] + self.ag[s]))


class TestAnswerGenerationSafeguard(Base):
    def test_counts_and_only_usable(self):
        for s in SPLITS:
            recs = L.load_answer_generation_records(s)
            self.assertEqual(len(recs), EXPECTED_ANSWER_RECORDS[s], s)
            self.assertTrue(all(r['eligibility'] == 'usable' and r['example_id'] in self.uid and r['example_id'] not in self.lid for r in recs))
            self.assertEqual(recs, [r for r in self.tr[s] if r['eligibility'] == 'usable'])

    def test_explicit_filter_on_general_split(self):
        for s in SPLITS:
            kept, rejected = L.select_answer_eligible(self.tr[s])
            self.assertEqual(len(kept), EXPECTED_ANSWER_RECORDS[s])
            self.assertEqual(len(rejected), EXPECTED_GENERAL_RECORDS[s] - EXPECTED_ANSWER_RECORDS[s])
            self.assertFalse({r['example_id'] for r in kept} & self.lid)

    def test_rejects_general_split_passed_as_answer_data(self):
        with self.assertRaises(L.LabelOnlyRecordError):
            L.assert_answer_eligible(self.tr['train'])

    def test_rejects_injected_label_only_record(self):
        label_rec = next(r for r in self.tr['train'] if r['eligibility'] == 'label_and_generation_only')
        with self.assertRaises(L.LabelOnlyRecordError):
            L.assert_answer_eligible(self.ag['train'] + [label_rec])

    def test_rejects_label_only_record_relabelled_as_usable(self):
        rec = copy.deepcopy(next(r for r in self.tr['validation'] if r['eligibility'] == 'label_and_generation_only'))
        rec['eligibility'] = 'usable'; rec['answer_source_available'] = True
        with self.assertRaises(L.LabelOnlyRecordError):
            L.assert_answer_eligible([rec])
        kept, rejected = L.select_answer_eligible([rec])
        self.assertEqual((kept, rejected), ([], [rec['record_id']]))

    def test_rejects_usable_record_with_answer_flag_removed(self):
        rec = copy.deepcopy(self.ag['test'][0]); rec['answer_source_available'] = False
        with self.assertRaises(L.LabelOnlyRecordError):
            L.assert_answer_eligible([rec])

    def test_rejects_label_only_example_records(self):
        with self.assertRaises(L.LabelOnlyRecordError):
            L.assert_answer_eligible(self.label[:1])

    def test_loader_rejects_tampered_answer_file(self):
        tmp = tempfile.mkdtemp()
        try:
            for f in ['usable_examples.jsonl', 'label_and_generation_only_examples.jsonl']:
                shutil.copy(os.path.join(ROOT, f), tmp)
            shutil.copytree(os.path.join(ROOT, 'answer_generation'), os.path.join(tmp, 'answer_generation'))
            label_rec = next(r for r in self.tr['train'] if r['eligibility'] == 'label_and_generation_only')
            path = os.path.join(tmp, 'answer_generation', 'train_transformation_records.jsonl')
            with open(path, 'a', encoding='utf-8', newline='\n') as f:
                f.write(json.dumps(label_rec, ensure_ascii=False) + '\n')
            with self.assertRaises(L.LabelOnlyRecordError):
                L.load_answer_generation_records('train', root=tmp)
            # also with the manifest hash refreshed, so only the eligibility checks can catch it
            mp = os.path.join(tmp, 'answer_generation', 'answer_generation_manifest.json')
            man = jload(mp)
            man['splits']['train']['sha256'] = sha256(path); man['splits']['train']['transformation_records'] += 1
            with open(mp, 'w', encoding='utf-8') as f:
                json.dump(man, f)
            with self.assertRaises(L.LabelOnlyRecordError):
                L.load_answer_generation_records('train', root=tmp)
        finally:
            shutil.rmtree(tmp)

    def test_manifest_lists_match_exports(self):
        man = jload(os.path.join(ROOT, 'answer_generation', 'answer_generation_manifest.json'))
        self.assertEqual(set(man['allowed_example_ids']), self.uid)
        self.assertEqual(set(man['denied_example_ids_label_and_generation_only']), self.lid)


class TestFalseClaims(Base):
    def test_markers(self):
        recs = [r for s in SPLITS for r in self.tr[s]]
        flagged = {r['example_id'] for r in recs if r['deliberately_false_claim']}
        self.assertEqual(len(flagged), 59)
        self.assertTrue(FALSE_CLAIM_EXPLICIT <= flagged)
        self.assertEqual(self.report['counts']['deliberately_false_claim_rows_documented'], 62)
        self.assertEqual(set(self.report['counts']['deliberately_false_claim_rows_in_excluded_examples']), FALSE_CLAIM_EXCLUDED)
        for r in recs:
            if r['deliberately_false_claim']:
                c = r['claim_under_evaluation']
                self.assertEqual(r['target_bloom_level'], 'Evaluate')
                self.assertEqual((c['role'], c['is_ground_truth']), ('object_of_evaluation', False))
                self.assertEqual(c['evidence_type'] == 'explicit_error_detection_note', r['example_id'] in FALSE_CLAIM_EXPLICIT)
                self.assertTrue(c['correction_log_note'])
                self.assertEqual(r['target_question'], self.master[r['example_id']]['transformations']['Evaluate'])
            else:
                self.assertNotIn('claim_under_evaluation', r)

    def test_markers_match_correction_log(self):
        logged = {r['Example ID'].zfill(3) for r in self.log if r['Target'] == 'Evaluate'
                  and re.search(r'error[- ]detection|error[- ]spotting', r['Reason'], re.I)}
        self.assertEqual(logged, FALSE_CLAIM_EXPLICIT)

    def test_consistent_across_record_types(self):
        ex = {x['id']: x for x in self.usable + self.label}
        for s in SPLITS:
            for r in self.tr[s]:
                self.assertEqual(r['deliberately_false_claim'], r['target_bloom_level'] in ex[r['example_id']]['deliberately_false_claim_levels'])
            for r in self.cl[s]:
                if r['role'] == 'transformation':
                    self.assertEqual(r['deliberately_false_claim'], r['bloom_level'] in ex[r['example_id']]['deliberately_false_claim_levels'])
                else:
                    self.assertFalse(r['deliberately_false_claim'])

    def test_answer_generation_keeps_markers(self):
        n = sum(r['deliberately_false_claim'] for s in SPLITS for r in self.ag[s])
        self.assertEqual(n, sum(1 for s in SPLITS for r in self.tr[s] if r['deliberately_false_claim'] and r['eligibility'] == 'usable'))


class TestProvenance(Base):
    def test_authored_scenarios(self):
        audit_keys = {(a['example_id'], a['target_level']) for a in self.audit}
        self.assertEqual(len(audit_keys), 152)
        self.assertEqual(sum(1 for a in self.audit if a['verdict'] == 'PRESERVES'), 148)
        recs = [r for s in SPLITS for r in self.tr[s]]
        marked = {(r['example_id'], r['target_bloom_level']) for r in recs if r['content_provenance'] == 'authored_scenario'}
        self.assertEqual(len(marked), 144)
        self.assertEqual(marked, {k for k in audit_keys if k[0] in self.uid | self.lid})
        excluded_preserves = sorted(k for k in audit_keys - marked
                                    if next(a for a in self.audit if (a['example_id'], a['target_level']) == k)['verdict'] == 'PRESERVES')
        self.assertEqual(excluded_preserves, [('156', 'Analyze'), ('156', 'Apply'), ('158', 'Analyze'), ('158', 'Create')])
        passes = {}
        for a in self.audit: passes[a['correction_log_pass']] = passes.get(a['correction_log_pass'], 0) + 1
        self.assertEqual(passes, {'2': 119, '1+2': 31, '1': 2})
        for r in recs:
            if r['content_provenance'] == 'authored_scenario':
                self.assertEqual(r['provenance_evidence']['basis'], 'self_containment_audit.csv')
                self.assertEqual(r['provenance_evidence']['audit_verdict'], 'PRESERVES')

    def test_every_record_has_provenance_and_types_agree(self):
        ex = {x['id']: x for x in self.usable + self.label}
        for x in ex.values():
            self.assertEqual(set(x['transformation_provenance']), set(x['transformations']))
            self.assertEqual(x['authored_scenario_levels'], [lv for lv, p in x['transformation_provenance'].items() if p == 'authored_scenario'])
        for s in SPLITS:
            for r in self.tr[s]:
                self.assertEqual(r['content_provenance'], ex[r['example_id']]['transformation_provenance'][r['target_bloom_level']])
            for r in self.cl[s]:
                want = 'source_question' if r['role'] == 'source' else ex[r['example_id']]['transformation_provenance'][r['bloom_level']]
                self.assertEqual(r['content_provenance'], want)


class TestSchema(Base):
    def test_all_records_conform(self):
        schema = jload(os.path.join(ROOT, 'schema.json'))
        D = schema['$defs']
        groups = [(self.usable + self.label + [x for s in SPLITS for x in self.ex[s]], D['example']),
                  ([r for s in SPLITS for r in self.tr[s] + self.ag[s]], D['transformationRecord']),
                  ([r for s in SPLITS for r in self.cl[s]], D['bloomClassificationRecord'])]
        n = 0
        for recs, sch in groups:
            for r in recs:
                errs = validate(r, sch, schema)
                self.assertFalse(errs, (r.get('record_id', r.get('id')), errs[:3]))
                n += 1
        self.assertEqual(n, 586 + 586 + 2930 + 1705 + 3516)


if __name__ == '__main__':
    unittest.main()
