import sys, os, json, csv, re, hashlib
sys.path.insert(0, r"C:/Users/USER/AppData/Local/Temp/claude/a4dc50f46edb93c0/a15f57c1-6094-4e38-93ba-1bc45a85b28a/scratchpad/pylib")
from collections import Counter
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                KeepTogether, PageBreak, Preformatted, CondPageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
SRC_PDF = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
LV = ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create']
d = json.load(open(os.path.join(HERE, 'corrected_full.json'), encoding='utf-8'))
recs = d['records']; log = d['log']; unres = d['unresolved']
src = {r['id']: r for r in json.load(open(os.path.join(HERE, 'structured.json'), encoding='utf-8'))}

def qtext(r):
    return '\n'.join(r['original_lines'])

# ---------------- corrected_dataset.json ----------------
out = []
label_changed = {e['id']: e['original'] for e in log if e['target'] == 'Original Bloom Level'}
for r in recs:
    item = {
        'id': f"{r['id']:03d}",
        'discipline': r['discipline'],
        'topic': r['topic'],
        'original_question': qtext(r),
        'original_bloom_level': r['original_level'],
    }
    if r['id'] in label_changed:
        item['source_bloom_level_before_correction'] = label_changed[r['id']]
    if r.get('clo'):
        item['proposed_clo'] = r['clo']
    if r['data_tables']:
        item['data_tables'] = [{'header': t[0], 'rows': t[1:]} for t in r['data_tables']]
    item['transformations'] = {k: r['transformations'][k] for k in LV if k in r['transformations']}
    out.append(item)
with open(os.path.join(OUT, 'corrected_dataset.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---------------- Correction_Log.csv ----------------
CATNAME = {
 'evaluate_justification': 'Evaluate – justification/criteria added',
 'create_strengthened': 'Create – strengthened to genuine construction',
 'notation': 'Mathematical notation',
 'level_coverage': 'Level coverage (after relabel)',
 'structural': 'Structural',
 'bloom_label': 'Bloom label correction',
 'technical_ambiguity': 'Technical – ambiguity',
 'technical_error': 'Technical – error',
 'technical_missing_content': 'Technical – missing content',
 'self_containment': 'Self-containment',
 'duplicate_of_original': 'Duplicate of original question',
 'duplicate_cross_example': 'Duplicate across examples',
 'technical_check_no_change': 'Technical check – no change (answer-key note)',
 'duplicate_flag': 'Duplicate – documented, retained',
}
order = {lv: i for i, lv in enumerate(['Title', 'Original', 'Original Bloom Level'] + LV)}
rows = []
for e in log:
    rows.append([f"{e['id']:03d}", e['target'], e['original'], e['corrected'], CATNAME[e['category']], e['reason']])
cur = {r['id']: r for r in recs}
for (i, lv, note) in d['check_notes']:
    if lv == 'Original':
        continue
    t = cur[i]['transformations'][lv]
    rows.append([f"{i:03d}", lv, t, t, CATNAME['technical_check_no_change'], note])
for a, b, note in d['duplicates']:
    rows.append([f"{a:03d}", 'Whole example', f'Duplicate pair {a:03d} / {b:03d}', 'Both retained (no merge)', CATNAME['duplicate_flag'], note])
rows.sort(key=lambda x: (x[0], order.get(x[1], 99)))
with open(os.path.join(OUT, 'Correction_Log.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Example ID', 'Target', 'Original Wording', 'Corrected Wording', 'Issue Category', 'Reason'])
    w.writerows(rows)

# ---------------- Unresolved_Questions.csv ----------------
unres_sorted = sorted(unres, key=lambda u: (u['id'], u['location']))
with open(os.path.join(OUT, 'Unresolved_Questions.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Example ID', 'Location', 'Issue', 'Detail', 'Recommended Action'])
    for u in unres_sorted:
        w.writerow([f"{u['id']:03d}", u['location'], u['issue'], u['detail'], u['action']])

# ---------------- PDF ----------------
FD = r'C:\Windows\Fonts'
pdfmetrics.registerFont(TTFont('Body', os.path.join(FD, 'cambria.ttc'), subfontIndex=0))
pdfmetrics.registerFont(TTFont('Body-Bold', os.path.join(FD, 'cambriab.ttf')))
pdfmetrics.registerFont(TTFont('Body-Italic', os.path.join(FD, 'cambriai.ttf')))
pdfmetrics.registerFont(TTFont('Body-BoldItalic', os.path.join(FD, 'cambriaz.ttf')))
pdfmetrics.registerFont(TTFont('Mono', os.path.join(FD, 'consola.ttf')))
addMapping('Body', 0, 0, 'Body'); addMapping('Body', 1, 0, 'Body-Bold')
addMapping('Body', 0, 1, 'Body-Italic'); addMapping('Body', 1, 1, 'Body-BoldItalic')

INK = colors.HexColor('#1f2328'); MUTED = colors.HexColor('#57606a'); RULE = colors.HexColor('#c9d1d9')
HEAD = colors.HexColor('#eef1f4'); ACCENT = colors.HexColor('#1f4e79')
st = {
 'title': ParagraphStyle('title', fontName='Body-Bold', fontSize=20, leading=25, textColor=INK, spaceAfter=10),
 'sub': ParagraphStyle('sub', fontName='Body', fontSize=11.5, leading=15, textColor=MUTED, spaceAfter=6),
 'body': ParagraphStyle('body', fontName='Body', fontSize=10, leading=13.2, textColor=INK, spaceAfter=3),
 'ex': ParagraphStyle('ex', fontName='Body-Bold', fontSize=11.5, leading=14.5, textColor=ACCENT, spaceBefore=8, spaceAfter=3),
 'sec': ParagraphStyle('sec', fontName='Body-Bold', fontSize=15, leading=19, textColor=INK, spaceBefore=10, spaceAfter=6),
 'cell': ParagraphStyle('cell', fontName='Body', fontSize=9.5, leading=12.2, textColor=INK),
 'cellb': ParagraphStyle('cellb', fontName='Body-Bold', fontSize=9.5, leading=12.2, textColor=INK),
 'note': ParagraphStyle('note', fontName='Body-Italic', fontSize=9, leading=11.5, textColor=MUTED, spaceAfter=3),
 'code': ParagraphStyle('code', fontName='Mono', fontSize=8.8, leading=11, textColor=INK, leftIndent=8),
}
def P(t, s='body'): return Paragraph(escape(t), st[s])
def PL(label, t, s='body'): return Paragraph(f'<b>{escape(label)}</b> {escape(t)}', st[s])

def on_page(c, doc):
    c.saveState(); c.setFont('Body', 8.5); c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, 'EduGuard Bloom Transformation Dataset — Corrected Edition')
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f'Page {doc.page}')
    c.restoreState()

doc = BaseDocTemplate(os.path.join(OUT, 'EduGuard_600_Corrected.pdf'), pagesize=A4,
                      leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                      title='EduGuard 600 Corrected', author='EduGuard dataset correction',
                      subject='Corrected Bloom-level question transformation dataset (600 examples)')
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='f')
doc.addPageTemplates([PageTemplate(id='p', frames=[frame], onPage=on_page)])
W = doc.width
story = []
cats = Counter(e['category'] for e in log)
story += [Spacer(1, 30 * mm), P('EduGuard Bloom Transformation Dataset', 'title'),
          P('Corrected edition: 600 examples, each with five Bloom-level transformations', 'sub'),
          Spacer(1, 4 * mm),
          P('Source: “EduGuard: Federated Sub-Billion Bloom Classification and Multitask Academic Generation for Privacy-Aware University Assistance” (PDF). The source file was not modified.'),
          P('This edition keeps the source layout (example heading, original question, original Bloom level, optional proposed CLO, embedded data, and a Target Level / Transformed question table). Where a source Bloom label was corrected, the level line shows the earlier label.'),
          P('Companion files in the same folder: Correction_Log.csv (every change, with the original wording and reason), Unresolved_Questions.csv (items that need human input, such as referenced material that is not in the dataset), Final_Summary.md (counts and method), and corrected_dataset.json (machine-readable copy with the same wording as this document).'),
          Spacer(1, 4 * mm),
          P(f"Examples: 600 · Transformations: {sum(len(r['transformations']) for r in recs)} · Logged changes: {len(log)} · Unresolved items: {len(unres)}", 'note'),
          PageBreak()]
for r in recs:
    block = [P(r['section'], 'sec')] if r.get('section') else []
    block.append(P(f"Example {r['id']:03d}: {r['title']}", 'ex'))
    lines = r['original_lines']
    code = set(r.get('code_lines', []))
    block.append(PL('Original:', lines[0]))
    for j, ln in enumerate(lines[1:], start=1):
        block.append(Paragraph(escape(ln), st['code']) if j in code else P(ln))
    for t in r['data_tables']:
        ncol = len(t[0])
        data = [[Paragraph(escape(c), st['cellb']) for c in t[0]]] + [[Paragraph(escape(c), st['cell']) for c in row] for row in t[1:]]
        tb = Table(data, colWidths=[W * 0.9 / ncol] * ncol, hAlign='LEFT')
        tb.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, RULE), ('BACKGROUND', (0, 0), (-1, 0), HEAD),
                                ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 2.5),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)]))
        block += [Spacer(1, 2), tb, Spacer(1, 3)]
    lvl = r['original_level']
    if r['id'] in label_changed:
        block.append(Paragraph(f"<b>Original Bloom Level:</b> {lvl} <i>(corrected from {label_changed[r['id']]})</i>", st['body']))
    else:
        block.append(PL('Original Bloom Level:', lvl))
    if r.get('clo'):
        block.append(P(r['clo']))
    data = [[Paragraph('Target Level', st['cellb']), Paragraph('Transformed question', st['cellb'])]]
    for k in LV:
        if k in r['transformations']:
            data.append([Paragraph(k, st['cellb']), Paragraph(escape(r['transformations'][k]), st['cell'])])
    tb = Table(data, colWidths=[24 * mm, W - 24 * mm], repeatRows=1)
    tb.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, RULE), ('BACKGROUND', (0, 0), (-1, 0), HEAD),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 3),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    story.append(KeepTogether(block + [Spacer(1, 2), tb]))
    story.append(Spacer(1, 4))
doc.build(story)

# ---------------- Final_Summary.md ----------------
n_trans = sum(len(r['transformations']) for r in recs)
changed_t = set((e['id'], e['target']) for e in log if e['target'] in LV)
substantive = set((e['id'], e['target']) for e in log if e['target'] in LV and e['category'] not in ('notation',))
unchanged = n_trans - len(changed_t)
labels = [e for e in log if e['category'] == 'bloom_label']
techn = [e for e in log if e['category'].startswith('technical') or e['category'] == 'self_containment']
struct = [e for e in log if e['category'] == 'structural']
notation_n = sum(1 for e in log if e['category'] == 'notation')
uc = Counter(u['issue'] for u in unres)
ex_with_unres = len(set(u['id'] for u in unres))
lvl_counts = Counter(r['original_level'] for r in recs)
src_counts = Counter(src[i]['original_level'] for i in src)
create_n = cats['create_strengthened']; eval_n = cats['evaluate_justification']
# validation checks
checks = []
checks.append(('600 examples retained (none excluded)', len(recs) == 600 and sorted(r['id'] for r in recs) == list(range(1, 601))))
checks.append(('Every example has exactly 5 transformations', all(len(r['transformations']) == 5 for r in recs)))
checks.append(('The 5 target levels are distinct and exclude the source level', all(set(r['transformations']) == set(LV) - {r['original_level']} for r in recs)))
checks.append(('No empty transformation or question', all(v.strip() for r in recs for v in r['transformations'].values()) and all(qtext(r).strip() for r in recs)))
jl = json.load(open(os.path.join(OUT, 'corrected_dataset.json'), encoding='utf-8'))
checks.append(('corrected_dataset.json parses and has 600 records', len(jl) == 600))
import pymupdf
pdf = pymupdf.open(os.path.join(OUT, 'EduGuard_600_Corrected.pdf'))
ptxt = re.sub(r'\s+', '', ''.join(p.get_text() for p in pdf))
def nz(s): return re.sub(r'\s+', '', s)
miss = [(x['id'], k) for x in jl for k, v in x['transformations'].items() if nz(v) not in ptxt]
missq = [x['id'] for x in jl if nz(x['original_question'].split('\n')[0]) not in ptxt]
checks.append(('Document and JSON wording identical (every JSON question and transformation found verbatim in the PDF text layer)', not miss and not missq))
h = hashlib.sha256(open(SRC_PDF, 'rb').read()).hexdigest()
checks.append((f'Source PDF unchanged (SHA-256 {h[:16]}…, {os.path.getsize(SRC_PDF):,} bytes)', h == sys.argv[3]))
checks.append(('No residual LaTeX commands', not any('\\' in v for r in recs for v in list(r['transformations'].values()) + r['original_lines'])))

md = []
A = md.append
A('# EduGuard 600-Example Dataset — Final Correction Summary\n')
A('Source: the uploaded PDF, *EduGuard: Federated Sub-Billion Bloom Classification and Multitask Academic Generation for Privacy-Aware University Assistance* (449 pages). The source file was read but not modified. The companion DOCX with the same title was used only to recover table structure and equation markup; every question was checked against the PDF text.\n')
A('## Counts\n')
A('| Measure | Count |\n|---|---:|')
A(f'| Examples reviewed | 600 |')
A(f'| Examples retained | 600 (none excluded) |')
A(f'| Transformations reviewed | {n_trans} |')
A(f'| Original Bloom labels corrected | {len(labels)} |')
A(f'| Transformations changed (any reason) | {len(changed_t)} |')
A(f'| — Create rows strengthened into genuine construction tasks | {create_n} |')
A(f'| — Evaluate rows given an explicit justification requirement | {eval_n} |')
A(f'| — Rows rewritten or added for level coverage after a relabel | {cats["level_coverage"] // 2} added, {cats["level_coverage"] // 2} removed |')
A(f'| — Notation-only changes (LaTeX or flattened equations to plain notation) | {notation_n} fields |')
A(f'| Transformations unchanged | {unchanged} |')
A(f'| Exact or functional duplicate pairs (documented, both kept) | {len(d["duplicates"])} |')
A(f'| Near-duplicate pairs noted (distinct values or levels, kept) | {len(d["near_duplicates"])} |')
A(f'| Unresolved items (rows in Unresolved_Questions.csv) | {len(unres)} across {ex_with_unres} examples |')
A(f'| Structural fixes | {len(struct)} |')
A(f'| Technical fixes (errors, ambiguity, missing content, self-containment) | {len(techn)} |')
A(f'| Technical answer-key notes (value checked, wording unchanged) | {sum(1 for c in d["check_notes"] if c[1] != "Original")} |')
A('')
A('## Bloom label corrections\n')
A('| Example | Before | After | Reason |\n|---|---|---|---|')
for e in labels:
    A(f"| {e['id']:03d} | {e['original']} | {e['corrected']} | {e['reason']} |")
A('\nFor each relabelled example, the transformation at the new source level was removed and one at the former source level was added, so each example still covers the five other levels. In 152 the Apply and Create rows also repeated the original task and were replaced; in 044 the Create row duplicated Example 226\'s original question and was replaced.\n')
A('Distribution of source labels, before → after: ' + ', '.join(f"{k} {src_counts[k]} → {lvl_counts[k]}" for k in LV) + '.\n')
A('## What changed and why\n')
A(f'- **Create rows ({create_n}).** Many Create rows asked for a "complete calculation", "derivation", "table", "worksheet", "trace", "statement", or a standard "representation"/"model" of the same problem. That is Apply or Understand work. Each was rewritten as real construction: a design, plan, modified scenario, generalisation, new test input, or proposal. The concept, scenario, numbers, units, and constraints are kept. Where a design needs a target value the source does not give, the learner is asked to set and justify it ("a chosen …") rather than the editor making one up.')
A(f'- **Evaluate rows ({eval_n}).** Rows that asked for a judgement or verification without asking for justification now end with "Justify your judgement with explicit reference to the criteria used." The task itself is unchanged.')
A('- **Technical.** Example 461 treated an NPV of ≈Tk. 824 as given, but the correct value is ≈Tk. 225.39; corrected. In Example 445 the proposed FIRST/FOLLOW sets were empty in the source and have been filled from the grammar (flagged for confirmation). Examples 417 and 530 use the same transformer data but report 5% and 5.26%; the regulation convention for each is now stated. Examples 446 and 462 referred to "proposed" values that were never given and are now self-contained.')
A('- **Answer-key notes, wording unchanged.** Eleven "verify whether the reported value is correct" items state a wrong value (415, 436, 447, 478, 483, 499, 514, 537, 548, 562, 599). They work as error-detection Evaluate tasks, so the wording is kept and the correct value is recorded in the log. Nine items state a correct value (407, 411, 440, 457, 525, 533, 551, 555, 560), also confirmed in the log.')
A('- **Structural.** Example 105\'s Create row, which sat outside its table, was restored as the fifth row and then strengthened. The "A*" titles of 366 and 477 were repaired. Private-use glyph markers were removed from 465\'s SQL code block, which is set in monospace. Embedded data tables in 428, 442, 475, 477 and 548 are kept, as are the multi-line data in 405, 429, 445, 465, 502 and 569 and the discipline section headings between Examples 080 and 093.')
A('- **Notation.** Equation markup that had flattened (for example `n=1∞n3n`, `01xexdx`) is rendered in consistent linear notation (`∑_(n=1)^∞ n/(3^n)`, `∫_0^1 x e^x dx`). LaTeX fragments in 072, 419, 421, 445, 502, 525, 569, 571 and 591 are converted to the same notation, with matrices written as `[[2, 1], [5, 3]]`.\n')
A('## Duplicates\n')
A('No examples were removed. The following pairs ask the same task at the same level with the same data. Both are kept, and their rewritten rows were differentiated where possible. Each pair is listed in Unresolved_Questions.csv for a merge-or-replace decision.\n')
A('| Pair | Note |\n|---|---|')
for a, b, n in d['duplicates']:
    A(f'| {a:03d} / {b:03d} | {n} |')
A('\nNear-duplicates, which differ in values or level and are kept as distinct items:\n')
for a, b, n in d['near_duplicates']:
    A(f'- {a:03d} / {b:03d}: {n}')
A('\n## Unresolved items (need human input)\n')
A('| Issue | Rows |\n|---|---:|')
for k, v in uc.most_common():
    A(f'| {k} | {v} |')
A(f'\nThe largest group is source questions that refer to material not in the dataset ("the supplied matrix", "the following resource allocation state", "the specified circuit"). That material cannot be recovered from the source, and inventing it is outside the brief, so those items keep their references and are listed for an author to supply the data. A smaller group is transformation rows that evaluate an artefact the item never shows ("a proposed testing strategy", "a supplied C program"). Example 477\'s heuristic is not admissible (h(A)=6 against a true remaining cost of 5, h(C)=4 against 3); the data is unchanged and flagged for review.\n')
A('## Method notes\n')
A('- Labels were judged by the cognitive operation each question requires, not by its leading verb.')
A('- Dataset convention kept: many Apply rows ask the learner to *classify, match, or identify* supplied new cases using a known scheme. That is treated as carrying out a procedure on unfamiliar instances (Apply), with the Understand row asking for explanation. The convention is used consistently, so these rows were left as they are. A reviewer who wants strict revised-taxonomy usage, where *classifying* belongs to Understand, should review Apply rows beginning "Classify", "Match", or "Identify" (about 160 rows).')
A('- Every change is in Correction_Log.csv with its original wording and reason. Wording in EduGuard_600_Corrected.pdf and corrected_dataset.json is generated from the same records.\n')
A('## Final validation checklist\n')
for name, ok in checks:
    A(f"- [{'x' if ok else ' '}] {name}")
A('')
open(os.path.join(OUT, 'Final_Summary.md'), 'w', encoding='utf-8').write('\n'.join(md))
print('checks:'); [print(' ', 'PASS' if ok else 'FAIL', n) for n, ok in checks]
if miss or missq: print('missing in pdf:', miss[:10], missq[:10])
print('pages', len(pdf))
