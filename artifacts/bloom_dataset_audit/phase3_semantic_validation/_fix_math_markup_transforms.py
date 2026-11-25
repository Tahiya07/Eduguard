import csv
p='artifacts/bloom_dataset_audit/phase3_semantic_validation/transformation_semantic_audit.csv'
with open(p,encoding='utf-8-sig',newline='') as h:rs=list(csv.DictReader(h)); fields=list(rs[0])
for r in rs:
 if int(r['example_id'])==551:
  if r['target_bloom_declared']=='Evaluate':
   r['transformed_question']='Verify whether the reported approximation of 14/3 is correct for ∫₀² (x²+1) dx using Simpson’s one-third rule with two equal subintervals.'
   r['target_bloom_assessed']='Apply'; r['status']='MAJOR';r['status_reason']='The reported 14/3 is correct, but checking the calculation is procedural application rather than criterion-based evaluation.';r['technical_correctness']='The Simpson estimate is 14/3 and is exact for this quadratic integrand.'
  if r['target_bloom_declared']=='Create':
   r['transformed_question']=r['transformed_question'].replace('143','14/3')
   r['technical_correctness']='The Simpson estimate is 14/3 and is exact for this quadratic integrand.'
 if int(r['example_id'])==597:
  r['transformed_question']=r['transformed_question'].replace('106','10^6').replace('103','10^3')
with open(p,'w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rs)
print('Updated math-notation audit rows')
