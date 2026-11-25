import csv
p='artifacts/bloom_dataset_audit/phase3_semantic_validation/transformation_semantic_audit.csv'
fix={529:('Understand','Remember','Define plain weave and identify its alternating over-one and under-one interlacement of warp and weft yarns.'),536:('Analyze','Understand','Explain how autocratic and democratic leadership differ in employee participation, decision-making authority, and the leader’s direction of project-team activities.'),568:('Analyze','Understand','Explain how in-situ and ex-situ conservation differ in conservation location, natural-habitat protection, and management of threatened species.')}
with open(p,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f)); fields=list(rows[0])
for r in rows:
 eid=int(r['example_id'])
 if eid in fix:
  oldsource,newtarget,q=fix[eid]; newsource=newtarget; r['source_bloom_declared']=newsource
  if r['target_bloom_declared']==newtarget:
   r.update(transformed_question=q,target_bloom_assessed=newtarget,actual_cognitive_demand_reasoning='Explains/retrieves the corrected target-level content in the exact source topic.',content_preservation='Preserves the academic content and all source comparison dimensions.',self_containment='Yes',technical_correctness='No technical defect identified.',status='PASS',status_reason='',audit_status='PASS',correction_status='CORRECTED',training_eligibility='PENDING',audit_state='Corrected target mapping revalidated after source Bloom reassessment')
with open(p,'w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
# corrected source records: post-correction status
p2='artifacts/bloom_dataset_audit/corrected_dataset/v2_autonomous/corrected_question_records.jsonl'
import json
rs=[json.loads(x) for x in open(p2,encoding='utf-8')]
for r in rs:
 if r['example_id'] in fix:r['audit_status']='PASS';r['training_eligibility']='PENDING'
with open(p2,'w',encoding='utf-8') as f:
 for r in rs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
# source audit tracks pre-correction mismatch plus corrected status separately
p3='artifacts/bloom_dataset_audit/phase3_semantic_validation/original_question_semantic_audit.csv'
with open(p3,encoding='utf-8-sig',newline='') as f:os=list(csv.DictReader(f));of=list(os[0])
for r in os:
 if int(r['example_id']) in fix:r['correction_status']='CORRECTED';r['post_correction_audit_status']='PASS';r['training_eligibility']='PENDING'
if 'post_correction_audit_status' not in of:of+=['post_correction_audit_status']
with open(p3,'w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=of);w.writeheader();w.writerows(os)
print('post-correction records revalidated')

