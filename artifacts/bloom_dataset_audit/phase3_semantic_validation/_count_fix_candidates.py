import csv,collections
p='artifacts/bloom_dataset_audit/phase3_semantic_validation/'
with open(p+'original_question_semantic_audit.csv',encoding='utf-8-sig',newline='') as f:o=list(csv.DictReader(f))
with open(p+'transformation_semantic_audit.csv',encoding='utf-8-sig',newline='') as f:t=list(csv.DictReader(f))
# Identify records whose audit_state indicates detailed independent review vs legacy provisional
reviewed_o=[r for r in o if 'Independent source-question re-audit' in r.get('audit_state','') or 'original DOCX mathematical structure' in r.get('audit_state','') or 'Corrected and revalidated' in r.get('audit_state','')]
reviewed_ids={int(r['example_id']) for r in reviewed_o}
reviewed_t=[r for r in t if int(r['example_id']) in reviewed_ids and 'legacy assessment' not in r.get('audit_state','')]
nonpassids={int(r['example_id']) for r in reviewed_t if r.get('status') not in ('PASS','')}
misids={int(r['example_id']) for r in reviewed_o if r.get('declared_bloom')!=r.get('independently_assessed_bloom')}
print('source reviewed',len(reviewed_o),'ids',len(reviewed_ids),'transformation rows for those',len(reviewed_t))
print('nonpass transformation ids among source reviewed',len(nonpassids))
print('source mismatch ids',len(misids))
print('union potential fix/review IDs',len(nonpassids|misids),'of reviewed ids')
print('unreviewed source IDs',600-len(reviewed_ids))
print('all nonpass transformations',len({int(r['example_id']) for r in t if r.get('status') not in ('PASS','')}))
print('source statuses',dict(collections.Counter(r.get('status','') for r in o)))
print('trans status',dict(collections.Counter(r.get('status','') for r in t)))
