import json, os
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION

root=Path('artifacts/bloom_dataset_audit')
src=root/'extracted_dataset/source_extraction.jsonl'
fix=root/'corrected_dataset/v2_autonomous/corrected_question_records.jsonl'
outdir=root/'corrected_dataset/working_600'
outdir.mkdir(parents=True,exist_ok=True)
out=outdir/'EduGuard_600_Examples_Corrected_Working.docx'
source=[json.loads(x) for x in src.read_text(encoding='utf-8').splitlines() if x.strip()]
corrected=[json.loads(x) for x in fix.read_text(encoding='utf-8').splitlines() if x.strip()]
byid={x['example_id']:x for x in corrected}
assert len(source)==600 and len(corrected)==600 and set(byid)==set(range(1,601))

doc=Document()
sec=doc.sections[0]
sec.top_margin=Inches(.65); sec.bottom_margin=Inches(.65); sec.left_margin=Inches(.8); sec.right_margin=Inches(.8)
styles=doc.styles
styles['Normal'].font.name='Aptos'; styles['Normal'].font.size=Pt(9)
styles['Heading 1'].font.name='Aptos Display'; styles['Heading 1'].font.size=Pt(16)
styles['Heading 2'].font.name='Aptos Display'; styles['Heading 2'].font.size=Pt(12)

title=doc.add_paragraph(); title.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=title.add_run('EduGuard Bloom-Level Dataset\n600-Example Corrected Working Copy'); r.bold=True; r.font.size=Pt(20); r.font.color.rgb=RGBColor(31,78,121)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.add_run('Prepared from the preserved source DOCX extraction').italic=True

doc.add_heading('Document status',level=1)
for t in [
 'This file contains all 600 source example IDs in their original order. The source DOCX has not been modified.',
 'Verified corrections already recorded in the correction log are applied. Other text is retained from the source extraction; inclusion here does not mean every item has completed independent semantic validation.',
 'Items without completed re-audit remain marked pending in the accompanying working records. This is a working corrected copy, not a claim that all 600 examples are fully validated.',
 'No examples were added, renumbered, or intentionally omitted.'
]: doc.add_paragraph(t,style='List Bullet')
doc.add_paragraph(f'Count: {len(source)} examples; {sum(len(x.get("transformations",[])) for x in source)} source transformation slots.')
doc.add_page_break()

for i,s in enumerate(source):
    c=byid[s['example_id']]
    orig=c.get('original_record',s)
    changed=(c.get('original_question')!=orig.get('original_question') or c.get('original_bloom')!=orig.get('original_bloom'))
    tx=c.get('transformations',[])
    doc.add_heading(f"Example {s['example_id']:03d}: {c.get('discipline_course_topic') or c.get('heading','')}",level=1)
    p=doc.add_paragraph(); p.add_run('Discipline/course/topic: ').bold=True; p.add_run(str(c.get('discipline_course_topic') or 'Not specified'))
    if changed:
        p=doc.add_paragraph(); p.add_run('Original source question (preserved): ').bold=True; p.add_run(str(orig.get('original_question','')))
        p=doc.add_paragraph(); p.add_run('Original source Bloom label: ').bold=True; p.add_run(str(orig.get('original_bloom','Not specified')))
        p=doc.add_paragraph(); p.add_run('Corrected question: ').bold=True; p.add_run(str(c.get('original_question','')))
        p=doc.add_paragraph(); p.add_run('Corrected Bloom label: ').bold=True; p.add_run(str(c.get('original_bloom','Not specified')))
    else:
        p=doc.add_paragraph(); p.add_run('Original question: ').bold=True; p.add_run(str(c.get('original_question','')))
        p=doc.add_paragraph(); p.add_run('Original Bloom level: ').bold=True; p.add_run(str(c.get('original_bloom','Not specified')))
    p=doc.add_paragraph(); p.add_run('Working audit status: ').bold=True; p.add_run(str(c.get('audit_status','PENDING_REAUDIT')))
    p=doc.add_paragraph(); p.add_run('Correction status: ').bold=True; p.add_run(str(c.get('correction_status','UNCHANGED')))
    if len(tx)!=5:
        p=doc.add_paragraph(); rr=p.add_run(f'FLAG: {len(tx)} transformations present; expected exactly 5.'); rr.bold=True; rr.font.color.rgb=RGBColor(192,0,0)
    for tr in tx:
        p=doc.add_paragraph(style='List Bullet')
        p.add_run(f"{tr.get('target_bloom','Unlabeled')}: ").bold=True
        p.add_run(str(tr.get('transformed_question','[MISSING QUESTION TEXT]')))
    for tab in c.get('embedded_tables',[]):
        if tab:
            try:
                rows=tab.get('rows',tab) if isinstance(tab,dict) else tab
                if isinstance(rows,list) and rows:
                    table=doc.add_table(rows=0,cols=max(len(r) for r in rows if isinstance(r,list)))
                    table.style='Table Grid'
                    for row in rows:
                        if isinstance(row,list):
                            cells=table.add_row().cells
                            for j,val in enumerate(row): cells[j].text=str(val)
            except Exception: pass
    if i<len(source)-1: doc.add_page_break()

# Appendix: correction trail
doc.add_page_break(); doc.add_heading('Appendix: verified correction trail',level=1)
logpath=root/'corrected_dataset/v2_autonomous/correction_log.csv'
if logpath.exists():
    import csv
    with logpath.open(encoding='utf-8-sig',newline='') as f: logs=list(csv.DictReader(f))
    doc.add_paragraph(f'{len(logs)} logged correction actions are reflected in this working copy where applicable. Full machine-readable detail is retained in correction_log.csv.')
    for row in logs:
        p=doc.add_paragraph(style='List Bullet')
        p.add_run(f"Example {row.get('example_id')} — {row.get('target_level')}: ").bold=True
        p.add_run(row.get('issue',''))
else: doc.add_paragraph('No correction log was available.')

doc.save(out)
print(json.dumps({'output':str(out.resolve()),'examples':len(source),'transformations':sum(len(x.get('transformations',[])) for x in source),'logged_correction_actions':len(logs) if logpath.exists() else 0,'size_bytes':out.stat().st_size},ensure_ascii=False))
