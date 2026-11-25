import sys; sys.path.insert(0, r"C:/Users/USER/AppData/Local/Temp/claude/a4dc50f46edb93c0/a15f57c1-6094-4e38-93ba-1bc45a85b28a/scratchpad/pylib")
import pymupdf, json, re, unicodedata
doc=pymupdf.open('src/source.pdf'); print('pages',len(doc))
full=''.join(p.get_text() for p in doc)
open('work/pdf_text.txt','w',encoding='utf-8').write(full)
def norm(s):
    s=unicodedata.normalize('NFKC',s)
    return re.sub(r'[^0-9a-z]','',s.lower())
N=norm(full)
recs=json.load(open('work/structured.json',encoding='utf-8'))
LV=['Remember','Understand','Apply','Analyze','Evaluate','Create']
# also remove level words from N variant to handle interleaving
NL=N
for l in LV: NL=NL.replace(l.lower(),'')
miss=[]
for r in recs:
    chunks=[('orig',' '.join(r['original_lines']))]+list(r['transformations'].items())
    for k,v in chunks:
        n=norm(v)
        if n in N: continue
        nl=n
        for l in LV: nl=nl.replace(l.lower(),'')
        if nl in NL: continue
        # partial: check prefix/suffix
        miss.append((r['id'],k,len(n), n[:40] in N, n[-40:] in N, v[:100]))
print('mismatches',len(miss))
for m in miss[:80]: print(m)
