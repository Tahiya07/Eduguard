import json,re,sys
recs=json.load(open('work/structured.json',encoding='utf-8'))
heads=set(sys.argv[1].split(','))
fields=sys.argv[2].split(',') if len(sys.argv)>2 else ['Create']
for r in recs:
    t=r['transformations'].get('Create')
    if not t: continue
    m=re.match(r'(\w+)\s+(?:a|an|the)?\s*((?:[\w\-–]+\s+){0,5}?)(\w+)\s+(?:for|of|that|to|showing|which|in|using|from|with|by|based|describing|explaining|comparing|linking|identifying|demonstrating|consisting|covering|organizing|detailing|outlining|including|connecting|mapping|relating|integrating|summarizing|illustrating|addressing|evaluating|incorporating|specifying|on|across|at|where)\b',t)
    h=m.group(3).lower() if m else '?'
    if h in heads:
        print(f"#{r['id']} [{r['original_level']}] {r['title']}\n O: {' '.join(r['original_lines'])[:260]}\n C: {t}")
