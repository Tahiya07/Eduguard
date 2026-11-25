import json,csv
base='artifacts/bloom_dataset_audit';p=base+'/phase3_semantic_validation';out=base+'/corrected_dataset/v2_autonomous';f=out+'/corrected_question_records.jsonl';rs=[json.loads(x) for x in open(f,encoding='utf8')];r=next(x for x in rs if x['example_id']==29);r['original_bloom']='Analyze';r['correction_status']='CORRECTED';r['audit_status']='PASS'
t=next(t for t in r['transformations'] if t['target_bloom']=='Analyze');old=t['transformed_question'];new='Assess whether replication by itself guarantees database availability and fault tolerance, using continued service when a replica fails as the criterion.';t.update(target_bloom='Evaluate',transformed_question=new)
with open(f,'w',encoding='utf8') as h:
 for x in rs:h.write(json.dumps(x,ensure_ascii=False)+'\n')
f=out+'/correction_log.csv'
with open(f,encoding='utf-8-sig',newline='') as h:logs=list(csv.DictReader(h));fields=list(logs[0])
logs += [dict(example_id=29,target_level='Original Bloom label',original_text='Evaluate',corrected_text='Analyze',issue='“Evaluate how” is used to request explanation of causal effects, without an explicit judgment standard.',justification='The question examines the relationship between replication, availability, and fault tolerance; dominant operation is relational analysis.',correction_status='CORRECTED',post_correction_validation='PASS'),dict(example_id=29,target_level='Evaluate',original_text=old,corrected_text=new,issue='Source relabeling requires an Evaluate target instead of Analyze.',justification='The corrected prompt requires a judgment against the explicit service-continuity criterion while preserving the replication/availability/fault-tolerance objective.',correction_status='CORRECTED',post_correction_validation='PASS')]
with open(f,'w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(logs)
f=p+'/transformation_semantic_audit.csv'
with open(f,encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h));fields=list(rows[0])
for x in rows:
 if int(x['example_id'])==29:
  x['source_bloom_declared']='Analyze'
  if x['target_bloom_declared']=='Analyze':x.update(target_bloom_declared='Evaluate',transformed_question=new,target_bloom_assessed='Evaluate',actual_cognitive_demand_reasoning='Judges replication against continued-service criterion when a replica fails.',content_preservation='Preserves replication, availability, fault tolerance, and service continuity.',self_containment='Yes',technical_correctness='A replica failure can impair service unless another replica remains available and serving.',status='PASS',status_reason='',audit_status='PASS',correction_status='CORRECTED',training_eligibility='PENDING',audit_state='Corrected and revalidated after source Bloom reclassification')
with open(f,'w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
f=p+'/original_question_semantic_audit.csv'
with open(f,encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h));fields=list(rows[0])
for x in rows:
 if int(x['example_id'])==29:x['correction_status']='CORRECTED';x['post_correction_audit_status']='PASS';x['training_eligibility']='PENDING'
with open(f,'w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
print('corrected and revalidated Example 29')
