from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,time
S=Path('/private/tmp/dmv2-route-coverage-20261021')
P=S/'protocol_sip_metadata_v3.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P)=='586c5d47198eedd92002c3df3e862950d9d48df7fe7502dd1269fb8ff4621503'
a=json.loads(P.read_text());out=S/'sip_metadata_download-v2'
if out.exists():raise SystemExit('Preserve old attempt: output exists')
out.mkdir()
receipt={'schema':'dmv2-sip-metadata-download/v1','status':'STARTED','protocol_sha256':sha(P),'started_at_utc':datetime.now(timezone.utc).isoformat(),'account_expected':a['account'],'project_expected':a['project'],'objects':[],'cloud_writes':0,'financial_values_decoded':False,'automatic_retries':0}
def save(): (out/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
def call(args):
 result=subprocess.run(['gcloud','--configuration='+a['gcloud_configuration'],'--quiet',*args],capture_output=True,text=True,timeout=120)
 if result.returncode:raise RuntimeError(result.stderr[-4000:])
 return result.stdout
receipt['manual_retry_after_user_sign_in']=True
receipt['prior_failed_receipt_sha256']=sha(S/'sip_metadata_download-v1/receipt.json')
save()
try:
 account=call(['config','get-value','account']).strip();project=call(['config','get-value','project']).strip()
 receipt['account_used']=account;receipt['project_used']=project
 assert account==a['account'] and project==a['project'],'Dedicated account/project mismatch'
 total=0
 for entry in a['objects']:
  obj=entry['object'];uri=obj['uri']+'#'+str(obj['generation']);metadata_text=call(['storage','objects','describe',uri,'--format=json'])
  assert len(metadata_text.encode())<=a['bounds']['describe_response_bytes_per_object']
  metadata=json.loads(metadata_text)
  assert int(metadata['generation'])==obj['generation'] and int(metadata['size'])==obj['size_bytes'],'Generation/size changed'
  target=out/(entry['table']+'.parquet');assert not target.exists()
  call(['storage','cp',uri,str(target)])
  assert target.stat().st_size==obj['size_bytes'] and sha(target)==obj['sha256'].removeprefix('sha256:'),'Downloaded bytes differ'
  total+=target.stat().st_size;assert total<=a['bounds']['download_bytes']
  receipt['objects'].append({'table':entry['table'],'path':str(target),'uri':obj['uri'],'generation':obj['generation'],'size_bytes':target.stat().st_size,'sha256':sha(target)});save()
 receipt['status']='PASS_PINNED_METADATA_DOWNLOAD';receipt['download_bytes']=total
except Exception as error:
 receipt['status']='FAILED_READ_ONLY_METADATA_DOWNLOAD';receipt['error']=str(error);save();print(json.dumps(receipt));raise SystemExit(1)
save();print(json.dumps(receipt))
