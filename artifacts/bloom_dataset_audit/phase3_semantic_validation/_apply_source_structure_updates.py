import csv,glob,os,json
p='artifacts/bloom_dataset_audit/phase3_semantic_validation'
# Update structurally verified source questions using OMML inspected in authoritative DOCX.
sourcefix={551:'Approximate the value of ∫₀² (x²+1) dx using Simpson’s one-third rule with n=2 equal subintervals.',571:'Determine reaction order from [A] changing 0.1 to 0.2 mol/L while rate changes 0.02 to 0.08 mol L⁻¹ s⁻¹, using r=k[A]ⁿ.',597:'Calculate decimal reduction time when microbial population decreases from 10⁶ to 10³ cells over 6 minutes under constant conditions and logarithmic linear inactivation.'}
f=p+'/original_question_semantic_audit.csv'
with open(f,encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h)); fields=list(rows[0])
for r in rows:
 eid=int(r['example_id'])
 if eid in sourcefix:
  r['original_question']=sourcefix[eid]
  r['independently_assessed_bloom']='Apply'; r['cognitive_operation_reasoning']={551:'Uses Simpson’s one-third formula on a specified integral with two subintervals.',571:'Infers exponent n from rate and concentration ratios; rate quadruples as concentration doubles, giving n=2.',597:'Computes log reduction: log10(10^6/10^3)=3; D=6/3=2 minutes.'}[eid]
  r['status']='PASS'; r['issue_reason']='Original DOCX OMML confirms the flattened notation; values are recoverable.'
  r['audit_state']='Re-audited; original DOCX mathematical structure inspected'
with open(f,'w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
# Correct transformation-audit metadata for affected math rows.
f=p+'/transformation_semantic_audit.csv'
with open(f,encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h)); fields=list(rows[0])
for r in rows:
 eid=int(r['example_id'])
 if eid in sourcefix:
  r['source_question']=sourcefix[eid]
  if eid==551:
   if r['target_bloom_declared']=='Remember': r.update(target_bloom_assessed='Remember',status='PASS',status_reason='',technical_correctness='Integral is ∫₀²(x²+1)dx; notation verified in DOCX.')
   elif r['target_bloom_declared']=='Understand': r.update(target_bloom_assessed='Understand',status='PASS',status_reason='',self_containment='Yes',technical_correctness='Integrand, limits, n=2 verified in DOCX.')
   elif r['target_bloom_declared']=='Analyze': r.update(target_bloom_assessed='Analyze',status='PASS',status_reason='',self_containment='Yes',technical_correctness='Integrand and interval verified in DOCX.')
   elif r['target_bloom_declared']=='Evaluate': r.update(target_bloom_assessed='Apply',status='MAJOR',status_reason='Reported 143 is incorrect; Simpson’s rule gives 14/3≈4.667. Checking by calculation is Apply rather than Evaluate.',self_containment='Yes',technical_correctness='Simpson result 14/3≈4.667; 143 is wrong.',correction_status='REVIEW_REQUIRED',training_eligibility='NOT_APPROVED')
   elif r['target_bloom_declared']=='Create': r.update(target_bloom_assessed='Apply',status='MAJOR',status_reason='A complete numerical derivation is procedural application, not meaningful Create.',self_containment='Yes',technical_correctness='Simpson result 14/3≈4.667.',correction_status='REVIEW_REQUIRED',training_eligibility='NOT_APPROVED')
  if eid==571:
   r.update(technical_correctness='OMML confirms r=k[A]^n; observed rate ratio 4 for concentration ratio 2 gives n=2.',self_containment='Yes')
   if r['target_bloom_declared']=='Evaluate': r.update(target_bloom_assessed='Apply',status='MAJOR',status_reason='Checking the proposed exponent against the supplied rate law is procedural application, not Evaluate.',correction_status='REVIEW_REQUIRED',training_eligibility='NOT_APPROVED')
   if r['target_bloom_declared']=='Create': r.update(target_bloom_assessed='Apply',status='MAJOR',status_reason='Deriving n from ratios is procedural application, not Create.',correction_status='REVIEW_REQUIRED',training_eligibility='NOT_APPROVED')
  if eid==597:
   r.update(technical_correctness='DOCX OMML confirms 10^6 to 10^3 cells; 3-log reduction in 6 min gives D=2 min.',self_containment='Yes')
   if r['target_bloom_declared']=='Remember': r.update(status='PASS',status_reason='',target_bloom_assessed='Remember')
   elif r['target_bloom_declared']=='Understand': r.update(status='PASS',status_reason='',target_bloom_assessed='Understand')
   elif r['target_bloom_declared']=='Analyze': r.update(status='PASS',status_reason='',target_bloom_assessed='Analyze')
   elif r['target_bloom_declared']=='Evaluate': r.update(target_bloom_assessed='Apply',status='MAJOR',status_reason='Verifying D-value requires executing the log-reduction calculation (Apply), not evaluation against an independent standard.',correction_status='REVIEW_REQUIRED',training_eligibility='NOT_APPROVED')
   elif r['target_bloom_declared']=='Create': r.update(target_bloom_assessed='Apply',status='MAJOR',status_reason='A mathematical derivation of D is procedural application, not Create.',correction_status='REVIEW_REQUIRED',training_eligibility='NOT_APPROVED')
with open(f,'w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
print('Updated DOCX-verified math cases')
