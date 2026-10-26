"""Prepare one explicitly recorded LR-only stability fork; never launch training."""
import argparse,json,math
from pathlib import Path
import yaml
from training.scripts.continue_taigi_training import fork_run,verify_parent
from training.scripts.run_experiment import identity,write_json,digest
p=argparse.ArgumentParser()
p.add_argument('--parent-config',type=Path,required=True)
p.add_argument('--child',type=Path,required=True)
p.add_argument('--endpoint',type=int,required=True)
a=p.parse_args()
parent,_=verify_parent(a.parent_config)
assert math.isclose(parent['optim']['lr'],0.0002,rel_tol=0,abs_tol=1e-12)
assert a.endpoint==1000 and not a.child.exists()
config_path=fork_run(a.parent_config,a.child,a.endpoint)
config=yaml.safe_load(config_path.read_text());config['optim']['lr']=0.00005
before=dict(parent);after=dict(config)
for key in ['run_dir','max_steps']:before.pop(key);after.pop(key)
before['optim']=dict(before['optim']);after['optim']=dict(after['optim'])
assert before['optim'].pop('lr')==0.0002 and after['optim'].pop('lr')==0.00005
assert before==after
config_path.write_text(yaml.safe_dump(config,sort_keys=False))
lineage_path=a.child/'lineage.json';lineage=json.loads(lineage_path.read_text())
lineage.update(allowed_config_changes=['run_dir','max_steps','optim.lr'],experiment='Bounded LR-only stability hypothesis after failed semantic800 gate',optimizer_state='Restored unchanged; trainer sets configured constant LR each update',learning_rate={'parent':0.0002,'child':0.00005},preparation_script_sha256=digest(Path(__file__)))
write_json(lineage_path,lineage)
tokenizer=config['model_overrides']['flow_lm.lookup_table.tokenizer_path']
write_json(Path(config['run_dir'])/'run_manifest.json',{'schema_version':1,'identity':identity(config_path,config,[tokenizer,str(lineage_path)])})
verified,_=verify_parent(config_path)
assert verified['optim']['lr']==0.00005
print(config_path)
