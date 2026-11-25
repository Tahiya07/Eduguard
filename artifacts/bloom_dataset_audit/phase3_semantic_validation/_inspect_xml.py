import zipfile,glob,xml.etree.ElementTree as ET
f=glob.glob('artifacts/bloom_dataset_audit/original_source/*.docx')[0]
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with zipfile.ZipFile(f) as z:r=ET.fromstring(z.read('word/document.xml'))
for p in r.findall('.//w:p',ns):
 txt=''.join(t.text or '' for t in p.findall('.//w:t',ns))
 if 'reported approximation' in txt:
  print('PARA',repr(txt))
  print(ET.tostring(p,encoding='unicode')[:3000].encode('ascii','backslashreplace').decode())



