"""Reversible review barrier after a complete look; no scientific file changes."""
import json,os,signal,time,hashlib
from pathlib import Path
pid=119998;root=Path('artifacts/branch_bi/round4');p=root/'stage2/round_summary.json'
while True:
 try:
  raw=p.read_bytes();s=json.loads(raw)
  if s['quality_passed'] and s['aggregation_crosschecked']:break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(.5)
cmd=Path(f'/proc/{pid}/cmdline').read_bytes();assert b'scripts/branch_bi4_execute.py' in cmd
os.kill(pid,signal.SIGSTOP)
(root/'review_barrier_stage2.json').write_text(json.dumps(dict(pid=pid,paused_at=time.time(),summary_sha256=hashlib.sha256(raw).hexdigest(),reason='Review full stage2 result before generating later precision data; reversible scheduling barrier, not a scientific stopping decision'),indent=2)+'\n')
print('Stage2 review barrier ready',flush=True)
