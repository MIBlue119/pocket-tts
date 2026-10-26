"""Transcribe the 24 fixed probe4 artifacts with the unchanged Breeze adapter."""
import json,time
from pathlib import Path
from difflib import SequenceMatcher
from training.taigi.audio import sha256_file
from training.taigi.transcriber import breeze_transcriber,BREEZE_MODEL,BREEZE_REVISION,BREEZE_ADAPTER_VERSION
p=Path('/mnt/HDD2/proj_tts/pocket-tts/data/taigi-adi-20261026/semantic-control-20261026/probe4-step800')
f=p/'results.json';x=json.loads(f.read_text());assert x['state']=='complete' and len(x['rows'])==4
out=p/'asr-results.json';assert not out.exists()
rows=[]
for i,r in enumerate(x['rows']):
 for v,a in r['variants'].items():
  path=Path(a['audio_path']);assert sha256_file(path)==a['audio_sha256']
  rows.append({'row_index':i,'mandarin_target':r['mandarin_target'],'variant':v,'audio_path':str(path),'audio_sha256':a['audio_sha256'],'seconds':a['seconds'],'normalized_latent_mse':a['normalized_latent_mse']})
assert len(rows)==24
report={'state':'running','input_report_sha256':sha256_file(f),'model':BREEZE_MODEL,'revision':BREEZE_REVISION,'adapter':BREEZE_ADAPTER_VERSION,'batch_size':4,'quality_verified':False,'started_epoch':time.time(),'rows':[]}
def save():
 t=out.with_suffix('.tmp');t.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');t.replace(out)
save()
try:
 tr=breeze_transcriber('cuda:0')
 for start in range(0,24,4):
  chunk=rows[start:start+4];texts=tr.transcribe_batch([(Path(r['audio_path']),0,r['seconds']) for r in chunk])
  for r,text in zip(chunk,texts,strict=True):
   r['asr_text']=text
   norm=lambda s:''.join(c for c in s if c.isalnum())
   r['mandarin_target_sequence_similarity']=SequenceMatcher(None,norm(r['mandarin_target']),norm(text)).ratio()
   report['rows'].append(r)
  save();print('Transcribed',len(report['rows']),'/ 24',flush=True)
 report.update(state='complete',ended_epoch=time.time())
except BaseException as exc:
 report.update(state='failed',error=f'{type(exc).__name__}: {exc}');raise
finally:save()
print(out)
