import csv,json
p='artifacts/bloom_dataset_audit/phase3_semantic_validation/transformation_semantic_audit.csv'
fix={529:('Understand','Remember'),536:('Analyze','Understand'),568:('Analyze','Understand')}
with open(p,encoding='utf-8-sig',newline='') as f:rs=list(csv.DictReader(f));fields=list(rs[0])
for r in rs:
 eid=int(r['example_id'])
 if eid in fix:r['source_bloom_declared']=fix[eid][0]
with open(p,'w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)
