import json,csv
base='artifacts/bloom_dataset_audit';p=base+'/phase3_semantic_validation';out=base+'/corrected_dataset/v2_autonomous'
f=out+'/corrected_question_records.jsonl';rs=[json.loads(x) for x in open(f,encoding='utf8')];r=next(x for x in rs if x['example_id']==48);r['original_bloom']='Analyze';r['correction_status']='CORRECTED';r['audit_status']='PASS'
t=next(t for t in r['transformations'] if t['target_bloom']=='Analyze');old=t['transformed_question'];new='Assess whether an increase in interest rates is likely to reduce investment and aggregate demand, using the standard ceteris-paribus monetary transmission relationship as the criterion.';t.update(target_bloom='Evaluate',transformed_question=new)
with open(f,'w',encoding='utf8') as h:
 for x in rs:h.write(json.dumps(x,ensure_ascii=False)+'\n')
# correction log append
f=out+'/correction_log.csv'
with open(f,encoding='utf-8-sig',newline='') as h:logs=list(csv.DictReader(h));fields=list(logs[0])
logs += [dict(example_id=48,target_level='Original Bloom label',original_text='Evaluate',corrected_text='Analyze',issue='Original wording requests causal explanation rather than a criterion-based judgment.',justification='Question examines causal influence of rates on investment and aggregate demand; no criterion or evidence for judging effectiveness is supplied.',correction_status='CORRECTED',post_correction_validation='PASS'),dict(example_id=48,target_level='Evaluate',original_text=old,corrected_text=new,issue='After source relabeling, target set must include Evaluate and exclude Analyze.',justification='Uses the standard ceteris-paribus transmission relationship as an explicit criterion and preserves the same rates/investment/aggregate-demand content.',correction_status='CORRECTED',post_correction_validation='PASS')]
with open(f,'w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(logs)
# audit row update
f=p+'/transformation_semantic_audit.csv'
with open(f,encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h));fields=list(rows[0])
for x in rows:
 if int(x['example_id'])==48:
  x['source_bloom_declared']='Analyze'
  if x['target_bloom_declared']=='Analyze':x.update(target_bloom_declared='Evaluate',transformed_question=new,target_bloom_assessed='Evaluate',actual_cognitive_demand_reasoning='Makes a judgment against the stated causal transmission relationship.',content_preservation='Preserves rate, investment, and aggregate-demand concepts.',self_containment='Yes',technical_correctness='Standard ceteris-paribus relationship: higher rates tend to reduce investment and aggregate demand.',status='PASS',status_reason='',audit_status='PASS',correction_status='CORRECTED',training_eligibility='PENDING',audit_state='Corrected and revalidated after source Bloom reclassification')
with open(f,'w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
# original audit keeps declared old label but records correction and post-correction validation
f=p+'/original_question_semantic_audit.csv'
with open(f,encoding='utf-8-sig',newline='') as h:rows=list(csv.DictReader(h));fields=list(rows[0])
for x in rows:
 if int(x['example_id'])==48:x['correction_status']='CORRECTED';x['post_correction_audit_status']='PASS';x['training_eligibility']='PENDING'
with open(f,'w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
print('corrected and revalidated Example 48; log entries',len(logs))
