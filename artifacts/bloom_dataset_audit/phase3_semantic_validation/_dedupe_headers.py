import csv
for p in ['artifacts/bloom_dataset_audit/phase3_semantic_validation/transformation_semantic_audit.csv','artifacts/bloom_dataset_audit/phase3_semantic_validation/original_question_semantic_audit.csv']:
 with open(p,encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f); rows=list(r); fields=[]
 for x in r.fieldnames:
  if x not in fields:fields.append(x)
 with open(p,'w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 print(p,len(rows),len(fields))
