"""Integrity audit with explicit mutable-planning-pointer exception."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from branch_bf_round1 import ROOT
from branch_b_round1 import digest,sha,save

def main():
    cache={};historical={};planning=[]
    def hashed(p):
        p=str(Path(p).resolve())
        if p not in cache:cache[p]=sha(p)
        return cache[p]
    for branch in ['b','bn','bt','br','bs','bl','bc','bw','bz']:
        p=Path(f'artifacts/branch_{branch}/reproducibility_manifest.json');m=json.loads(p.read_text());bad=[]
        for name,v in m['files'].items():
            expected=v['sha256'] if isinstance(v,dict) else v
            if Path(name).name=='NEXT_EXPERIMENT_PLAN.md':
                planning.append({'branch':branch,'expected':expected,'current':hashed(name)});continue
            if hashed(name)!=expected:bad.append(name)
        assert not bad,(branch,bad)
        historical[branch]={'entries':len(m['files']),'manifest_sha256':hashed(p)}
    code=json.loads((ROOT/'execution_code_manifest.json').read_text());assert all(hashed(p)==v for p,v in code.items())
    dm=json.loads((ROOT/'data_manifest.json').read_text());seen=set()
    for p,v in dm['prior_files'].items():
        assert hashed(p)==v
        with np.load(p) as a:
            for key in ['images'] if 'images' in a else ['solid','ring']:seen.update(digest(x) for x in a[key].reshape(-1,1,32,32))
    assert len(seen)==294000
    lock=json.loads((ROOT/'training_lock.json').read_text());assert lock['new_heads']==20
    assert all(hashed(p)==v for p,v in lock['files'].items());new=0
    for name,meta in dm['banks'].items():
        p=ROOT/f'{name}.npz';assert hashed(p)==meta['sha256']
        with np.load(p) as a:
            hs={digest(x) for x in a['images'].reshape(-1,1,32,32)};assert len(hs)==meta['images'] and not seen.intersection(hs);seen.update(hs);new+=len(hs)
        if name.startswith('test'):assert p.stat().st_mtime>=lock['locked_at']
    assert new==30000
    for seed in range(301,306):
        fit=json.loads((ROOT/f'seed{seed}/fits.json').read_text());assert fit['finished_at']<=lock['locked_at']
    save(ROOT/'integrity.json',{'passed':True,'checked_at':time.time(),'historical_science_unchanged':historical,'mutable_planning_pointer_entries':planning,'new_images':new,'prior_retained_images':294000,'all_heads_frozen_before_tests':True,'execution_code_unchanged':True,'restoration_record':'artifacts/session_recovery_20260925/RECOVERY.md'})
    print('Integrity passed: prior science, restored BL files, 294000 old + 30000 new images, code and chronology.',flush=True)
if __name__=='__main__':main()
