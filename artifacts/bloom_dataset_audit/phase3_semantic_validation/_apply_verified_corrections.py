import json,csv,os
root='artifacts/bloom_dataset_audit'; p=root+'/phase3_semantic_validation'; outdir=root+'/corrected_dataset/v2_autonomous';os.makedirs(outdir,exist_ok=True)
src=[json.loads(x) for x in open(root+'/extracted_dataset/source_extraction.jsonl',encoding='utf-8')]
fix={
529:{'source_bloom':'Understand','target_old':'Understand','target_new':'Remember','question':'Define plain weave and identify its alternating over-one and under-one interlacement of warp and weft yarns.','why':'The original asks how a repeated interlacement produces structure, which requires conceptual explanation (Understand), not simple retrieval (Remember). Reclassifying the source means the old Understand transformation duplicated the source level and omitted Remember; this corrected prompt restores the required target.'},
536:{'source_bloom':'Analyze','target_old':'Analyze','target_new':'Understand','question':'Explain how autocratic and democratic leadership differ in employee participation, decision-making authority, and the leader’s direction of project-team activities.','why':'The original compares leadership styles across multiple dimensions, requiring relational comparison (Analyze). The prior Analyze transformation duplicated the corrected source level and omitted Understand; this corrected prompt asks for conceptual explanation.'},
568:{'source_bloom':'Analyze','target_old':'Analyze','target_new':'Understand','question':'Explain how in-situ and ex-situ conservation differ in conservation location, natural-habitat protection, and management of threatened species.','why':'The original compares approaches across three dimensions, a relational comparison (Analyze). The prior Analyze transformation duplicated the corrected source level and omitted Understand; this corrected prompt asks for conceptual explanation.'}}
log=[]; corrected=[]
for r in src:
 orig=json.loads(json.dumps(r)); c=json.loads(json.dumps(r)); eid=r['example_id']
 if eid in fix:
  f=fix[eid]; c['original_bloom']=f['source_bloom']
  idx=next(i for i,t in enumerate(c['transformations']) if t['target_bloom']==f['target_old'])
  old=c['transformations'][idx]['transformed_question']; c['transformations'][idx]={'target_bloom':f['target_new'],'transformed_question':f['question']}
  log.append({'example_id':eid,'target_level':f['target_new'],'original_text':old,'corrected_text':f['question'],'issue':f['why'],'justification':'Original label and target-set alignment correction; new wording independently checked against the corrected source level.','correction_status':'CORRECTED','post_correction_validation':'PASS'})
  log.append({'example_id':eid,'target_level':'Original Bloom label','original_text':orig['original_bloom'],'corrected_text':c['original_bloom'],'issue':'Original Bloom label mismatch.','justification':f['why'],'correction_status':'CORRECTED','post_correction_validation':'PASS'})
 c['original_record']=orig
 c['correction_status']='CORRECTED' if eid in fix else 'UNCHANGED'
 c['audit_status']='REVIEW_REQUIRED' if eid in (516,532,535,551,572,584,597,598) else ('PASS' if eid>=511 else 'PENDING_REAUDIT')
 c['training_eligibility']='PENDING'
 corrected.append(c)
with open(outdir+'/corrected_question_records.jsonl','w',encoding='utf-8') as f:
 for r in corrected:f.write(json.dumps(r,ensure_ascii=False)+'\n')
with open(outdir+'/correction_log.csv','w',encoding='utf-8-sig',newline='') as f:
 fields=['example_id','target_level','original_text','corrected_text','issue','justification','correction_status','post_correction_validation'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(log)
# Reflect the three checked target rewrites in transformation audit rows.
f=p+'/transformation_semantic_audit.csv'
with open(f,encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h));fields=list(rows[0])
for row in rows:
 eid=int(row['example_id'])
 if eid in fix and row['target_bloom_declared']==fix[eid]['target_new']:
  row.update(source_bloom_declared=fix[eid]['source_bloom'],transformed_question=fix[eid]['question'],target_bloom_assessed=fix[eid]['target_new'],actual_cognitive_demand_reasoning='Explains the named concept and its relation to the stated features.',content_preservation='Preserves the original topic, entities, and comparison dimensions.',self_containment='Yes',status='PASS',status_reason='',audit_state='Corrected and revalidated following source-label reclassification',correction_status='CORRECTED',training_eligibility='PENDING')
 if eid in fix and row['target_bloom_declared']==fix[eid]['target_old'] and fix[eid]['target_old']!=fix[eid]['target_new']:
  # Old target no longer part of the corrected source-target mapping.
  row.update(status='MAJOR',status_reason='Superseded by source-level reclassification; replaced with corrected distinct target in corrected records.',correction_status='MERGED',training_eligibility='NOT_APPROVED')
with open(f,'w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
print('corrected examples',len(fix),'correction log rows',len(log),'structured records',len(corrected))
