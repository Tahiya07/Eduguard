import csv
p='artifacts/bloom_dataset_audit/phase3_semantic_validation/transformation_semantic_audit.csv'
changes={529:('Understand','Remember'),536:('Analyze','Understand'),568:('Analyze','Understand')}
with open(p,encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h)); fields=list(rows[0])
for r in rows:
 eid=int(r['example_id'])
 if eid in changes and r['target_bloom_declared']==changes[eid][0]:
  old,new=changes[eid]; r['target_bloom_declared']=new
  r['audit_state']='Corrected target mapping revalidated after source Bloom reassessment'
  r['correction_status']='CORRECTED';r['training_eligibility']='PENDING'
with open(p,'w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
print('target labels remapped')
