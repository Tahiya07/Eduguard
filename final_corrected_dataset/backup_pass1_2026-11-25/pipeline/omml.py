W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
def _simple(s):
    s=s.strip()
    return len(s)<=1 or s.replace('.','').isalnum()
def wrap(s):
    s=s.strip()
    return s if _simple(s) else '('+s+')'
def kids(el,name):
    return [c for c in el if c.tag==M+name]
def sub(el,name):
    k=kids(el,name); return conv(k[0]) if k else ''
def conv(el):
    tag=el.tag
    if tag==M+'t': return el.text or ''
    if tag==W+'t': return el.text or ''
    loc=tag.split('}')[1]
    if tag.startswith(M):
        if loc=='sSup': return wrap(sub(el,'e')) if False else base(el)+'^'+wrap(sub(el,'sup'))
        if loc=='sSub': return base(el)+'_'+wrap(sub(el,'sub'))
        if loc=='sSubSup': return base(el)+'_'+wrap(sub(el,'sub'))+'^'+wrap(sub(el,'sup'))
        if loc=='f': return wrap(sub(el,'num'))+'/'+wrap(sub(el,'den'))
        if loc=='rad':
            dg=sub(el,'deg').strip(); e=sub(el,'e')
            return ('√' if not dg else dg+'√')+wrap(e)
        if loc=='nary':
            chr_='∫'
            pr=kids(el,'naryPr')
            if pr:
                for c in pr[0]:
                    if c.tag==M+'chr': chr_=c.get(M+'val') or chr_
            lo=sub(el,'sub').strip(); hi=sub(el,'sup').strip(); e=sub(el,'e').strip()
            s=chr_
            if lo: s+='_'+wrap(lo)
            if hi: s+='^'+wrap(hi)
            return s+' '+e
        if loc=='d':
            beg,end='(',')'; sep=','
            pr=kids(el,'dPr')
            if pr:
                for c in pr[0]:
                    if c.tag==M+'begChr': beg=c.get(M+'val','')
                    if c.tag==M+'endChr': end=c.get(M+'val','')
                    if c.tag==M+'sepChr': sep=c.get(M+'val',',')
            return beg+sep.join(conv(e) for e in kids(el,'e'))+end
        if loc=='m':
            rows=[]
            for mr in kids(el,'mr'):
                rows.append('['+', '.join(conv(e) for e in kids(mr,'e'))+']')
            return '['+', '.join(rows)+']'
        if loc=='func': return sub(el,'fName')+' '+sub(el,'e')
        if loc=='limLow': return sub(el,'e')+'_'+wrap(sub(el,'lim'))
        if loc=='limUpp': return sub(el,'e')+'^'+wrap(sub(el,'lim'))
        if loc=='bar' or loc=='acc' or loc=='groupChr' or loc=='box' or loc=='borderBox':
            return sub(el,'e')
        if loc in ('rPr','ctrlPr','naryPr','dPr','fPr','sSupPr','sSubPr','radPr','mPr','funcPr','sSubSupPr'): return ''
    if tag==W+'tab': return '\t'
    if tag==W+'br': return '\n'
    if tag==W+'rPr' or tag==W+'pPr': return ''
    return ''.join(conv(c) for c in el)
def base(el): return wrap(sub(el,'e')) if not _simple(sub(el,'e')) else sub(el,'e')
def text(el): return conv(el)
