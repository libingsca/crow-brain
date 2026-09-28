"""Run original frozen look driver, replacing only the digest-set implementation."""
import json,sys
from pathlib import Path
import branch_bi5_execute as original
from branch_b_round1 import sha
from branch_bi_codec import load_bank
import numpy as np
R=Path('artifacts/branch_bi/round5');LOCK=R/'engineering_revision02_lock.json'
def verify():
 m=json.loads(LOCK.read_text());assert all(sha(p)==h for p,h in m['files'].items())
def compare_partial(j):
 old=R/'implementation_revisions/prepare_memory/look4_partial';root=R/f'look{j}';checked=[];incomplete=[]
 for p in sorted(old.glob('*.npz')):
  try:
   with np.load(p) as f:d={k:f[k] for k in f.files}
  except Exception as e:incomplete.append(dict(path=str(p),error=type(e).__name__));continue
  with np.load(root/p.name) as f:
   assert set(d)==set(f.files)
   for k,v in d.items():np.testing.assert_array_equal(v,f[k])
  checked.append(dict(path=str(p),sha256=sha(p),reproduced=str(root/p.name),reproduced_sha256=sha(root/p.name)))
 (root/'partial_reproduction_audit.json').write_text(json.dumps(dict(passed=True,all_complete_archived_arrays_bitwise_equal=True,checked=checked,incomplete_archived_files=incomplete),indent=2)+'\n')
run_original=original.run
def run(script,args,j):
 verify();actual='branch_bi5_data_revision02.py' if script=='branch_bi5_data.py' else script;run_original(actual,args,j)
 if script=='branch_bi5_data.py' and args[0]=='prepare':compare_partial(j)
 verify()
if __name__=='__main__':
 verify();original.run=run;original.main(int(sys.argv[1]));verify()
