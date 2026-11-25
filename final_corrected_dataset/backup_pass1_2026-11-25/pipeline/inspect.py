import sys; sys.path.insert(0, r"C:/Users/USER/AppData/Local/Temp/claude/a4dc50f46edb93c0/a15f57c1-6094-4e38-93ba-1bc45a85b28a/scratchpad/pylib")
import docx,sys
d=docx.Document(sys.argv[1])
body=d.element.body
n_p=n_t=0
for i,el in enumerate(body.iterchildren()):
    tag=el.tag.split('}')[1]
    if tag=='p': n_p+=1
    elif tag=='tbl': n_t+=1
print('paras',n_p,'tables',n_t, 'doc tables', len(d.tables))
for i,el in enumerate(list(body.iterchildren())[:40]):
    tag=el.tag.split('}')[1]
    if tag=='p':
        print('P', repr(''.join(t.text or '' for t in el.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'))[:120]))
    else:
        print('T')
