import csv,json
from pathlib import Path
p=Path('artifacts/bloom_dataset_audit/phase3_semantic_validation')
src=[json.loads(s) for s in Path('artifacts/bloom_dataset_audit/extracted_dataset/source_extraction.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
by={x['example_id']:x for x in src}
notes={
136:{
'Remember':('Remember','Retrieves names of relational-algebra operations.','Same student/course retrieval and named relations.','Yes','Operation names are established facts.','PASS',''),
'Understand':('Understand','Explains the roles of selection, projection, and join in the retrieval.','Same query objective and relation set.','Yes','No specific tuples needed for the conceptual explanation.','PASS',''),
'Apply':('Apply','Executes relational-algebra operations on concrete relations.','Same retrieval objective and three named relations.','No','Student, Enrollment, and Course schemas and tuples plus the specified course are absent.','REVIEW','The referenced relations and course identifier are missing.'),
'Analyze':('Analyze','Examines schema attributes and links to derive an operation sequence.','Same entities and retrieval goal.','No','Schemas/attributes and relationship definitions are not supplied.','REVIEW','Analysis depends on absent relation schemas.'),
'Evaluate':('Evaluate','Judges correctness of an expression against the intended retrieval.','Same target result and relations.','No','No candidate expression, relation schemas, or explicit correctness basis is given.','REVIEW','Expression and relation definitions are absent; "verify" alone does not supply an evaluand.')},
137:{
'Remember':('Remember','Retrieves a definition and identifies roles.','Same delegation concept and manager/subordinate actors.','Yes','Definition and role terminology are recall.','PASS',''),
'Apply':('Understand','Classifies cases using conceptual distinctions, but no delegated responsibility set or decision rule is supplied.','Same delegation-of-authority topic.','No','No responsibilities or classification criteria are provided; even with them, bare classification may be Understand.','MAJOR','Does not clearly require executing a procedure; referenced cases are absent.'),
'Analyze':('Analyze','Examines an authority structure to identify manager-subordinate relationships.','Same authority-delegation context.','No','The organizational structure is referenced but not included.','REVIEW','The structure to analyze is missing.'),
'Evaluate':('Evaluate','Would judge clarity of the authority relationships.','Same organizational relationships.','No','No organization details or criterion for "clear" are supplied.','REVIEW','The basis and evidence needed for judgment are absent.'),
'Create':('Create','Constructs an authority chart, a new representation of delegation relationships.','Same managers/subordinates and delegation concept.','No','The specified actors and delegation links are missing.','REVIEW','Cannot construct the requested chart without source actors and relationships.')},
138:{
'Remember':('Remember','Retrieves the variables shown on a dose-response curve.','Same drug-dose and pharmacological-response concepts.','Yes','Variable names are recall.','PASS',''),
'Understand':('Understand','Explains the conceptual trend shown by a dose-response relationship.','Same dose-response concept.','No','Curve/data are referenced but not present; general knowledge may permit a general explanation, but the question explicitly requires supplied curve.','REVIEW','Referenced curve is missing.'),
'Apply':('Apply','Reads/interpolates a response for a specified dose from a curve.','Same dose-response relation.','No','Specified dose and curve are absent.','REVIEW','Dose and curve/data are missing.'),
'Evaluate':('Understand','Checks whether a claim matches data; no explicit criterion or meaningful judgment beyond interpreting the curve.','Same dose-response relation.','No','Claim and supplied curve are not provided.','MAJOR','Target is closer to interpretation than Evaluate, and the evidence/evaluand is missing.'),
'Create':('Create','Constructs a graph from experimental dose-response data.','Same drug-dose and response variables.','No','The dose-response data are absent.','REVIEW','Graph cannot be constructed without the referenced data.')},
139:{
'Remember':('Remember','Recalls standard block-reduction rules.','Same series, parallel, and feedback system topic.','Yes','Rules are established knowledge.','PASS',''),
'Understand':('Understand','Explains how block combinations correspond to equivalent transfer functions.','Same reduction concepts.','Yes','Conceptual explanation does not depend on a particular diagram.','PASS',''),
'Analyze':('Analyze','Examines interconnections to determine reduction dependencies/order.','Same feedback control system and reduction goal.','No','The supplied block diagram is absent.','REVIEW','Diagram needed for structural analysis is missing.'),
'Evaluate':('Apply','Checks equivalence of a proposed transfer function, ordinarily by applying reduction/algebraic rules; no explicit evaluative criteria are named.','Same system and equivalent-transfer-function goal.','No','The diagram and proposed function are missing.','MAJOR','Verification is procedural checking and is not a defensible Evaluate task as written.'),
'Create':('Apply','Produces a derivation by executing standard reduction steps; the task does not require a new design or synthesis.','Same reduction process and equivalent-transfer-function objective.','No','The diagram to reduce is absent.','MAJOR','A worked reduction is procedural application, not meaningful creation; diagram is also missing.')},
140:{
'Remember':('Remember','Recalls the height-of-instrument method and required observation types.','Same surveying level-book method and staff-reading context.','Yes','Method name and reading types are established knowledge.','PASS',''),
'Understand':('Understand','Explains relationships among instrument height, staff readings, and reduced levels.','Same method and quantities.','Yes','Conceptual relationship can be explained without particular measurements.','PASS',''),
'Analyze':('Understand','Distinguishes observation categories; this is classification/interpretation, not analysis of relationships or dependencies.','Same staff-reading types and reduced-level calculation context.','No','The cited readings are missing.','MAJOR','Actual demand is basic classification, and referenced readings are absent.'),
'Evaluate':('Apply','Checks numerical consistency using the arithmetic relation/check; no substantive judgment against explicit quality criteria is required.','Same HI method and reduced levels.','No','Readings and purported reduced levels are absent.','MAJOR','A numerical arithmetic check is Apply, not Evaluate; values needed to check are absent.'),
'Create':('Apply','Fills a level-book table by carrying out the prescribed HI calculation; this is a procedural representation, not meaningful construction.','Same sequence of survey points, method, and reduced levels.','No','Staff readings and point labels are absent.','MAJOR','A table of calculated values is superficial Create; required readings are missing.')}
}
fields=['example_id','discipline_course_topic','source_question','source_bloom_declared','target_bloom_declared','transformed_question','target_bloom_assessed','actual_cognitive_demand_reasoning','content_preservation','self_containment','technical_correctness','status','status_reason','audit_state','correction_status','training_eligibility','audit_status']
rows=[]
for eid in range(136,141):
 x=by[eid]
 for tr in x['transformations']:
  actual,reason,preserve,selfc,tech,status,status_reason=notes[eid][tr['target_bloom']]
  rows.append(dict(example_id=eid,discipline_course_topic=x['discipline_course_topic'],source_question=x['original_question'],source_bloom_declared=x['original_bloom'],target_bloom_declared=tr['target_bloom'],transformed_question=tr['transformed_question'],target_bloom_assessed=actual,actual_cognitive_demand_reasoning=reason,content_preservation=preserve,self_containment=selfc,technical_correctness=tech,status=status,status_reason=status_reason,audit_state='Independent legacy re-audit batch 136-140',correction_status=('REVIEW_REQUIRED' if status=='REVIEW' else ('CORRECTION_RECOMMENDED' if status=='MAJOR' else 'UNCHANGED')),training_eligibility='PENDING',audit_status=status))
out=p/'transformation_batch_legacy_reaudit_136_140.csv'
with out.open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
# build cumulative independent legacy assessments, retaining all original batch files
parts=[]
for start in range(1,141,5):
 end=start+4; q=p/f'transformation_batch_legacy_reaudit_{start:03d}_{end:03d}.csv'
 if not q.exists(): raise SystemExit(f'Missing batch {q.name}')
 with q.open(encoding='utf-8-sig',newline='') as f: parts.extend(csv.DictReader(f))
assert len(parts)==700 and len({(r['example_id'],r['target_bloom_declared']) for r in parts})==700
con=p/'transformation_legacy_reaudit_consolidated_001_140.csv'
with con.open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(parts)
# Status summary for checkpoint
from collections import Counter
counts=Counter(r['status'] for r in parts)
cp={'covered_examples':140,'assessed_transformations':700,'expected_transformations':700,'status_counts':dict(counts),'next_example':141,'source_docx_modified':False,'note':'Legacy provisional audit records retained in their original files; the independent batch files and this consolidated audit hold question-specific reassessments. Assessments do not imply corrections have been applied or training approval.'}
(p/'legacy_reaudit_checkpoint_001_140.json').write_text(json.dumps(cp,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'batch':str(out),'batch_rows':len(rows),'consolidated':str(con),'total_rows':len(parts),'status_counts':dict(counts)},ensure_ascii=False))
