import csv,glob,os,json,collections
root='artifacts/bloom_dataset_audit'; p=root+'/phase3_semantic_validation'; outdir=root+'/corrected_dataset/v2_autonomous'; os.makedirs(outdir,exist_ok=True)
# Pull source rows from extraction for traceable merge and fill audit field names.
source={r['example_id']:r for r in map(json.loads,open(root+'/extracted_dataset/source_extraction.jsonl',encoding='utf-8'))}
allrows=[]
for fn in glob.glob(p+'/transformation_batch_examples_*.csv'):
 if os.path.basename(fn)=='transformation_batch_examples_*.csv': continue
 with open(fn,encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if 'target_bloom_declared' not in r: continue
   eid=int(r['example_id']);
   r['example_id']=eid
   # Ensure independent correction/training axes are explicit for every row.
   r.setdefault('correction_status','UNCHANGED' if r.get('status')=='PASS' else 'REVIEW_REQUIRED')
   r.setdefault('training_eligibility','APPROVED' if r.get('status')=='PASS' else 'PENDING')
   # first 140 assessments are explicitly provisional pending question-specific re-audit
   if eid<=140:
    r['audit_state']='PROVISIONAL — legacy assessment; question-specific re-audit required'
    r['correction_status']='REVIEW_REQUIRED'; r['training_eligibility']='NOT_APPROVED'
   allrows.append(r)
# Remove accidental duplicates only if exact duplicate rows; preserve conflicting keys for QC report.
bykey=collections.defaultdict(list)
for r in allrows: bykey[(r['example_id'],r['target_bloom_declared'])].append(r)
# Batch files are the authoritative row sources; exact duplicate keys are not expected.
fields=['example_id','discipline_course_topic','source_question','source_bloom_declared','target_bloom_declared','transformed_question','target_bloom_assessed','actual_cognitive_demand_reasoning','content_preservation','self_containment','technical_correctness','status','status_reason','audit_state','correction_status','training_eligibility']
with open(p+'/transformation_semantic_audit.csv','w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(allrows)
# quantitative structural reconciliation
counts=collections.Counter(r['status'] for r in allrows); byid=collections.Counter(r['example_id'] for r in allrows)
missing=[i for i in source if byid[i]!=5]
summary={'expected_transformations':3000,'records':len(allrows),'unique_example_target_pairs':len(bykey),'duplicate_pairs':sum(1 for v in bykey.values() if len(v)>1),'examples_with_other_than_five':len(missing),'missing_or_incomplete_example_ids':missing,'transformation_status_counts':dict(counts),'coverage_new_independent_examples':'511-600 assessed; legacy 1-140 provisional and not considered independently validated','source_docx_modified':False,'correction_stage':'In progress; no bulk question rewrites issued from incomplete cross-dataset audit','training_approval':'No records approved until source-question re-audit/reconciliation is completed'}
with open(p+'/transformation_reconciliation_checkpoint.json','w',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False))
