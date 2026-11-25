import sys; sys.path.insert(0, r"C:/Users/USER/AppData/Local/Temp/claude/a4dc50f46edb93c0/a15f57c1-6094-4e38-93ba-1bc45a85b28a/scratchpad/pylib")
import docx, json, re
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omml import text as txt
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
def txt_old(el):
    out=[]
    for t in el.iter():
        if t.tag in (W+'t', M+'t'): out.append(t.text or '')
        elif t.tag==W+'tab': out.append('\t')
        elif t.tag==W+'br': out.append('\n')
    return ''.join(out)
d=docx.Document(sys.argv[1])
items=[]; cur=None; stray=[]
for el in d.element.body.iterchildren():
    tag=el.tag.split('}')[1]
    if tag=='p':
        s=txt(el).strip()
        if not s: continue
        m=re.match(r'Example (\d{3}):\s*(.*)',s)
        if m:
            cur={'id':int(m.group(1)),'title':m.group(2),'paras':[],'tables':[]}; items.append(cur); continue
        if cur is None: stray.append(s); continue
        cur['paras'].append(s)
    elif tag=='tbl':
        rows=[]
        for tr in el.iter(W+'tr'):
            rows.append([txt(tc).strip() for tc in tr.iter(W+'tc')])
        (cur['tables'] if cur else stray).append(rows)
json.dump({'items':items,'stray':stray},open(sys.argv[2],'w',encoding='utf-8'),ensure_ascii=False,indent=1)
print(len(items),'items; stray',len(stray))
ids=[i['id'] for i in items]; print('missing',sorted(set(range(1,601))-set(ids)),'dups',[x for x in set(ids) if ids.count(x)>1])
for i in items:
    nt=len(i['tables']); np_=len(i['paras'])
    if nt!=1 or np_!=2: print(i['id'], 'tables',nt,'paras',np_, [p[:70] for p in i['paras']])
for s in stray[:20]: print('STRAY', str(s)[:150])
