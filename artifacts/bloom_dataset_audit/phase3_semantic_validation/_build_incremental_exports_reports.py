import csv,json,os,collections,random,hashlib
root='artifacts/bloom_dataset_audit'; p=root+'/phase3_semantic_validation'; out=root+'/training_ready_exports/v2_autonomous_incremental';os.makedirs(out,exist_ok=True)
# Load complete corrected source records and audits
corr=[json.loads(x) for x in open(root+'/corrected_dataset/v2_autonomous/corrected_question_records.jsonl',encoding='utf-8')]; src={r['example_id']:r for r in corr}
with open(p+'/original_question_semantic_audit.csv',encoding='utf-8-sig',newline='') as f: orig=list(csv.DictReader(f)); of=list(orig[0])
with open(p+'/transformation_semantic_audit.csv',encoding='utf-8-sig',newline='') as f: trans=list(csv.DictReader(f)); tf=list(trans[0])
# Explicit separate status fields; original/source review coverage remains visible.
for r in orig:
 eid=int(r['example_id']); r['audit_status']=r['status'] if eid>=511 else 'REVIEW'; r['correction_status']='CORRECTED' if eid in (529,536,568) else ('UNCHANGED' if r['audit_status']=='PASS' else 'REVIEW_REQUIRED'); r['training_eligibility']='PENDING' if eid<511 or r.get('post_correction_audit_status',r['audit_status']) in ('REVIEW','REJECT') else ('APPROVED' if r.get('post_correction_audit_status',r['audit_status'])=='PASS' else 'PENDING')
with open(p+'/original_question_semantic_audit.csv','w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=of);w.writeheader();w.writerows(orig)
origby={int(r['example_id']):r for r in orig}
for r in trans:
 eid=int(r['example_id']); r['audit_status']=r['status'] if eid>=511 or eid>=141 else 'REVIEW'
 if eid<=140: r['audit_status']='REVIEW'
 r['correction_status']=r.get('correction_status') or ('UNCHANGED' if r['audit_status']=='PASS' else 'REVIEW_REQUIRED')
 # Gate approval at whole-source level: no source ID under 511 is approved yet.
 r['training_eligibility']='PENDING' if eid<511 else ('APPROVED' if r['audit_status']=='PASS' and origby[eid]['audit_status']=='PASS' and origby[eid]['independently_assessed_bloom']==src[eid]['original_bloom'] else ('NOT_APPROVED' if r['audit_status'] in ('MAJOR','REJECT') else 'PENDING'))
# verify 5 unique target labels per source after corrected labels
byid=collections.defaultdict(list)
for r in trans: byid[int(r['example_id'])].append(r)
for eid,rr in byid.items():
 labels=[x['target_bloom_declared'] for x in rr]; source=src[eid]['original_bloom']
 good=len(rr)==5 and len(set(labels))==5 and source not in labels
 if not good:
  for x in rr: x['training_eligibility']='NOT_APPROVED'
with open(p+'/transformation_semantic_audit.csv','w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=tf);w.writeheader();w.writerows(trans)
# Build eligible record list only where every member and source are approved and the five-way mapping is complete.
approved=[]; review=[]; rejected=[]
for eid,rr in sorted(byid.items()):
 srcrow=src[eid]; origrow=origby[eid]; labels=[x['target_bloom_declared'] for x in rr]
 eligible=(eid>=511 and origrow.get('post_correction_audit_status',origrow['audit_status'])=='PASS' and origrow['independently_assessed_bloom']==srcrow['original_bloom'] and len(rr)==5 and len(set(labels))==5 and srcrow['original_bloom'] not in labels and all(x['audit_status']=='PASS' and x['target_bloom_assessed']==x['target_bloom_declared'] for x in rr))
 for x in rr:
  rec={'example_id':eid,'source_question':srcrow['original_question'],'source_bloom':srcrow['original_bloom'],'target_bloom':x['target_bloom_declared'],'transformed_question':x['transformed_question'],'discipline':srcrow['discipline_course_topic'],'course_topic':srcrow['discipline_course_topic'],'validation_status':'PASS' if eligible else ('REJECT' if x['audit_status']=='REJECT' else 'REVIEW_REQUIRED')}
  if eligible:approved.append(rec)
  elif x['audit_status']=='REJECT':rejected.append(rec)
  else:
   rec['review_reason']=x.get('status_reason') or (f'Whole-example gate not satisfied; source audit={origrow["audit_status"]}, transformation audit={x["audit_status"]}.' if eid<511 or not eligible else '');review.append(rec)
for name,records in [('approved_training.jsonl',approved),('review_required.jsonl',review),('rejected_examples.jsonl',rejected)]:
 with open(out+'/'+name,'w',encoding='utf-8') as f:
  for rec in records:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
# source-ID grouped reproducible split over approved source IDs only
ids=sorted({r['example_id'] for r in approved}); random.Random(20261125).shuffle(ids)
n=len(ids); ntest=max(1,round(n*.1)) if n>=10 else 0; nval=max(1,round(n*.1)) if n>=10 else 0
split={'train':ids[:max(0,n-ntest-nval)],'validation':ids[max(0,n-ntest-nval):max(0,n-ntest)],'test':ids[max(0,n-ntest):]}
byname={k:[] for k in split}
for r in approved:
 for k,sel in split.items():
  if r['example_id'] in sel: byname[k].append(r);break
for k,rr in byname.items():
 with open(out+f'/{k}.jsonl','w',encoding='utf-8') as f:
  for rec in rr:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
manifest={'seed':20261125,'split_unit':'original example_id','approved_source_ids':ids,'partitions':split,'record_counts':{k:len(v) for k,v in byname.items()},'leakage_free':not(bool(set(split['train'])&set(split['validation']) or set(split['train'])&set(split['test']) or set(split['validation'])&set(split['test']))),'scope_note':'Incremental approved subset only; not a final dataset split while Examples 1–510 source audits remain provisional.'}
with open(out+'/split_manifest.json','w',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
# Audit findings: every non-pass plus original label differences, including re-audit scope.
find=[]
for r in orig:
 mismatch=r['declared_bloom']!=r['independently_assessed_bloom']
 if r['audit_status']!='PASS' or mismatch:
  find.append({'record_type':'original','example_id':r['example_id'],'target_level':'','audit_status':r['audit_status'],'finding':r.get('issue_reason') or ('Declared/independent Bloom mismatch.' if mismatch else 'Provisional; re-audit pending.'),'correction_status':r['correction_status'],'training_eligibility':r['training_eligibility']})
for r in trans:
 if r['audit_status']!='PASS' or r['target_bloom_assessed']!=r['target_bloom_declared']:
  find.append({'record_type':'transformation','example_id':r['example_id'],'target_level':r['target_bloom_declared'],'audit_status':r['audit_status'],'finding':r.get('status_reason') or ('Declared target does not match independent cognitive assessment.' if r['target_bloom_assessed']!=r['target_bloom_declared'] else ''),'correction_status':r['correction_status'],'training_eligibility':r['training_eligibility']})
with open(p+'/audit_findings.csv','w',encoding='utf-8-sig',newline='') as f:
 fs=['record_type','example_id','target_level','audit_status','finding','correction_status','training_eligibility'];w=csv.DictWriter(f,fieldnames=fs);w.writeheader();w.writerows(find)
# Distribution and transition reports from corrected records, all 600 examples and 3,000 pairs.
levels=['Remember','Understand','Apply','Analyze','Evaluate','Create']; source_dist=collections.Counter(r['original_bloom'] for r in corr); assessed_dist=collections.Counter(r['independently_assessed_bloom'] for r in orig)
target_dist=collections.Counter(x['target_bloom'] for s in corr for x in s['transformations']); transitions=collections.Counter((s['original_bloom'],x['target_bloom']) for s in corr for x in s['transformations'])
disc=collections.Counter(s['discipline_course_topic'] for s in corr)
analysis={'example_count':len(corr),'transformation_count':len(trans),'source_bloom_declared_corrected_distribution':{l:source_dist[l] for l in levels},'source_bloom_independent_assessment_distribution':{l:assessed_dist[l] for l in levels},'target_bloom_distribution':{l:target_dist[l] for l in levels},'directed_transition_distribution':{f'{a}->{b}':transitions[(a,b)] for a in levels for b in levels if a!=b},'discipline_course_topic_distribution':dict(sorted(disc.items())),'provisional_source_audits':sum(r['audit_status']=='REVIEW' for r in orig),'independently_reaudited_source_questions':sum('Independent source-question re-audit' in r.get('audit_state','') or 'Original DOCX mathematical structure' in r.get('audit_state','') for r in orig),'source_bloom_mismatches':sum(r['declared_bloom']!=r['independently_assessed_bloom'] for r in orig),'transformation_audit_status_counts':dict(collections.Counter(r['audit_status'] for r in trans)),'approved_training_records':len(approved),'review_required_records':len(review),'rejected_records':len(rejected),'split_manifest':manifest}
with open(p+'/bloom_distribution_analysis.json','w',encoding='utf-8') as f:json.dump(analysis,f,ensure_ascii=False,indent=2)
with open(p+'/pattern_analysis.json','w',encoding='utf-8') as f:json.dump({'scope':'Screening and interim pattern counts; not a complete semantic deduplication adjudication.','transformation_target_level_counts':analysis['target_bloom_distribution'],'target_cognitive_mismatch_counts':dict(collections.Counter(x['target_bloom_declared'] for x in trans if x['target_bloom_assessed']!=x['target_bloom_declared'])),'status_counts':analysis['transformation_audit_status_counts'],'duplicate_screening_file':'near_duplicate_report.csv','near_duplicate_adjudication':'Candidates remain REVIEW until semantic context adjudicated.','repeated_calculation_verification_pattern_count':sum('Verify whether' in x['transformed_question'] or 'Verify whether' in x['transformed_question'][:40] for x in trans),'approved_examples_by_discipline':dict(collections.Counter(src[i]['discipline_course_topic'] for i in manifest['approved_source_ids']))},f,ensure_ascii=False,indent=2)
# Near duplicate file remains screening-only; exact candidates retained without deletion.
with open(p+'/near_duplicate_report.csv',encoding='utf-8-sig',newline='') as f: dup=list(csv.DictReader(f));df=list(dup[0]) if dup else []
if dup:
 if 'adjudication_status' not in df:df+=['adjudication_status']
 for r in dup:r['adjudication_status']='REVIEW_REQUIRED — context-level adjudication pending; no records removed.'
 with open(p+'/near_duplicate_report.csv','w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=df);w.writeheader();w.writerows(dup)
print(json.dumps({'approved_records':len(approved),'approved_ids':ids,'review_records':len(review),'rejected':len(rejected),'transitions':sum(transitions.values()),'findings':len(find),'split_integrity':manifest['leakage_free']},ensure_ascii=False))


