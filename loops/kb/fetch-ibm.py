# Fetches every IBM Docs page a sources map names into ibm/pages/<sha1>.json as text, with its title and retrieval date.
import json,subprocess,html,re,os,datetime,hashlib,time,sys
K=os.path.dirname(os.path.abspath(__file__))
src=json.load(open(os.path.join(K,sys.argv[1] if len(sys.argv)>1 else 'sources-utility.json')))
pages=sorted({p for v in src.values() for p in v})
os.makedirs(K+'/ibm/pages',exist_ok=True)
today=datetime.date.today().isoformat()
def text_of(h):
    h=re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>',' ',h)
    h=re.sub(r'(?i)<br\s*/?>|</(p|div|li|tr|h\d|pre|dt|dd|table)>','\n',h)
    t=re.sub(r'<[^>]+>',' ',h); t=html.unescape(t)
    t=re.sub(r'[ \t ]+',' ',t); t=re.sub(r' *\n\s*\n*','\n',t)
    return t.strip()
ok=0
for p in pages:
    url='https://www.ibm.com/docs/api/v1/content/SSLTBW_3.1.0/'+p
    out=K+'/ibm/pages/'+hashlib.sha1(p.encode()).hexdigest()[:12]+'.json'
    if os.path.exists(out): ok+=1; continue
    r=subprocess.run(['curl','-s','-m','60','-w','\n%{http_code}',url],capture_output=True,text=True)
    body,code=r.stdout.rsplit('\n',1)
    if code!='200': print('FAIL',code,p); continue
    title=re.search(r'(?is)<title>(.*?)</title>',body); title=html.unescape(title.group(1)).strip() if title else p
    json.dump({'href':p,'url':'https://www.ibm.com/docs/en/zos/3.1.0?topic='+p.split('/')[-1].replace('.htm',''),'api':url,'title':title,'retrieved':today,'text':text_of(body)},open(out,'w'))
    ok+=1; time.sleep(0.3)
print('pages',ok,'of',len(pages))
