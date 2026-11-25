import json,re
from collections import Counter
recs=json.load(open('work/structured.json',encoding='utf-8'))
# dump compact
with open('work/all.txt','w',encoding='utf-8') as f:
    for r in recs:
        f.write(f"#{r['id']:03d} [{r['original_level']}] {r['title']}\nO: {' / '.join(r['original_lines'])}\n")
        for k,v in r['transformations'].items(): f.write(f"  {k[:3]}: {v}\n")
        if r['stray_create']: f.write(f"  STRAY: {r['stray_create']}\n")
C=Counter(); E=Counter(); first=Counter()
for r in recs:
    for k,v in r['transformations'].items():
        w=v.split()[0]
        first[(k,w)]+=1
for k in ['Remember','Understand','Apply','Analyze','Evaluate','Create']:
    print(k, [(w,c) for (kk,w),c in first.most_common() if kk==k][:15])
cr=[(r['id'],r['transformations']['Create']) for r in recs if 'Create' in r['transformations']]
pat=re.compile(r'\b(table|worksheet|sheet|list|summary|checklist|chart|record|matrix|derivation|calculation|solution procedure|step-by-step|representation|diagram)\b',re.I)
weak=[(i,t) for i,t in cr if re.match(r'(Construct|Prepare|Produce|Create|Develop)\s+(a|an)\s+(complete\s+|step-by-step\s+|structured\s+|worked\s+|full\s+)?(calculation|derivation|solution|worked|computation|tabular|table|worksheet|sheet|summary|list|record|representation)',t,re.I)]
print('weak create',len(weak))
ev=[(r['id'],r['transformations']['Evaluate']) for r in recs if 'Evaluate' in r['transformations']]
print('evaluate starting Verify',sum(1 for i,t in ev if t.startswith('Verify')),'of',len(ev))
print('evaluate lacking justify/criteri',sum(1 for i,t in ev if not re.search(r'justif|criteri|based on|by (checking|examining|comparing|consider)|in terms of|using|with reference',t,re.I)))
ext=re.compile(r'\b(supplied|attached|provided|given|following|specified|stated|above|below|shown|presented)\b',re.I)
print('items w external refs (orig):',sum(1 for r in recs if ext.search(' '.join(r['original_lines']))))
same=[(r['id'],k) for r in recs for k,v in r['transformations'].items() if re.sub(r'\W','',v.lower())==re.sub(r'\W','',' '.join(r['original_lines']).lower())]
print('identical to original',same)
