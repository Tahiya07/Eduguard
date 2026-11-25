import json,re,sys
raw=json.load(open('work/docx_raw.json',encoding='utf-8'))['items']
LV=['Remember','Understand','Apply','Analyze','Evaluate','Create']
SECTIONS={'Pharmacy','EEE','Law','Civil Engineering','Architecture','CSE','BBA'}
out=[];issues=[]
for it in raw:
    paras=list(it['paras']); rec={'id':it['id'],'title':it['title']}
    t=it['title']
    if ' — ' in t: rec['discipline'],rec['topic']=t.split(' — ',1)
    elif ' - ' in t: rec['discipline'],rec['topic']=t.split(' - ',1)
    else: rec['discipline'],rec['topic']=t,''
    sec=[p for p in paras if p in SECTIONS]
    paras=[p for p in paras if p not in SECTIONS]
    rec['section_header_after']=sec[0] if sec else None
    li=[i for i,p in enumerate(paras) if p.startswith('Original Bloom Level:')]
    assert len(li)==1,it['id']
    rec['original_level']=paras[li[0]].split(':',1)[1].strip()
    q=paras[:li[0]]; assert q[0].startswith('Original:')
    q[0]=q[0][len('Original:'):].strip()
    rec['original_lines']=q
    rec['clo']=None; rec['stray_create']=None
    for p in paras[li[0]+1:]:
        if p.startswith('Proposed CLO:'): rec['clo']=p
        elif p.startswith('| Create |'): rec['stray_create']=p
        else: issues.append((it['id'],'extra para',p))
    tabs=it['tables']
    trans=[t for t in tabs if t and t[0] and t[0][0].strip().lower().startswith('target')]
    data=[t for t in tabs if t not in trans]
    assert len(trans)==1,(it['id'],[t[0] for t in tabs])
    rec['data_tables']=data
    rows=trans[0][1:]; tr={}
    for r in rows:
        if len(r)!=2: issues.append((it['id'],'rowlen',r)); continue
        lv=r[0].strip()
        if lv not in LV: issues.append((it['id'],'badlevel',lv)); continue
        if lv in tr: issues.append((it['id'],'duplevel',lv))
        tr[lv]=r[1]
    rec['header']=trans[0][0]
    rec['transformations']=tr
    if rec['original_level'] not in LV: issues.append((it['id'],'bad orig level',rec['original_level']))
    exp=set(LV)-{rec['original_level']}
    if set(tr)!=exp: issues.append((it['id'],'levelset',sorted(tr),'missing',sorted(exp-set(tr)),'extra',sorted(set(tr)-exp)))
    for k,v in tr.items():
        if not v.strip(): issues.append((it['id'],'empty',k))
    out.append(rec)
json.dump(out,open('work/structured.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
from collections import Counter
print(Counter(r['original_level'] for r in out))
print(Counter(tuple(r['header']) for r in out))
for i in issues: print(i)
print('data tables:',[(r['id'],len(r['data_tables'])) for r in out if r['data_tables']])
print('no topic:',[r['id'] for r in out if not r['topic']][:50])
