"""Pre-evaluation deterministic duplicate rejection, preserving completed banks."""
import json,time
import numpy as np
from scipy.ndimage import label
from crowbrain.task import COUNTS,draw_dots,balanced_pairs
from branch_bf_round1 import ROOT,verify_lock
from branch_b_round1 import digest,sha,save

verify_lock();assert not list(ROOT.glob('seed*/results.json')) and not list(ROOT.glob('seed*/evaluation_manifest.json'))
m=json.loads((ROOT/'data_manifest.json').read_text());seen=set();records=m['banks'];rejected=[]
for p,v in m['prior_files'].items():
    assert sha(p)==v
    with np.load(p) as d:
        for key in ['images'] if 'images' in d else ['solid','ring']:seen.update(digest(x) for x in d[key].reshape(-1,1,32,32))
assert len(seen)==294000
for name in ['fit','validation']:
    p=ROOT/f'{name}.npz';assert sha(p)==records[name]['sha256']
    with np.load(p) as d:seen.update(digest(x) for x in d['images'].reshape(-1,1,32,32))
for bi in range(20):
    name=f'test{bi:02d}';seed=990100+bi;k=200;n=2000;p=ROOT/f'{name}.npz'
    if p.exists():
        d=dict(np.load(p));assert int(d['seed'])==seed;images=d['images'];hs=[digest(x) for x in images.reshape(-1,1,32,32)]
        assert len(set(hs))==1000 and not seen.intersection(hs);seen.update(hs)
    else:
        rng=np.random.default_rng(seed)
        images=np.stack([draw_dots(int(count),rng)[0] for count in COUNTS for _ in range(k)]).reshape(5,k,1,32,32)
        # Generate the complete original bank first; replacements continue its RNG.
        own=set()
        for ci,count in enumerate(COUNTS):
            for ii in range(k):
                old=digest(images[ci,ii]);current=old;attempts=0
                while current in seen or current in own:
                    images[ci,ii,0]=draw_dots(int(count),rng)[0];current=digest(images[ci,ii]);attempts+=1
                if attempts:rejected.append({'bank':name,'seed':seed,'count':int(count),'image_index':ii,'rejected_sha256':old,'replacement_sha256':current,'draws':attempts})
                own.add(current)
        seen.update(own)
        prng=np.random.default_rng(seed+10000);pairs=balanced_pairs(50);prng.shuffle(pairs);a,b=pairs.T;ia,ib=prng.integers(0,k,(2,n));same=(a==b)&(ia==ib);ib[same]=(ib[same]+1)%k;y=(a==b).astype(np.int64)
        np.savez_compressed(p,images=images,a=a,b=b,ia=ia,ib=ib,y=y,seed=seed);d=dict(np.load(p))
    assert np.array_equal([label(x[0]>.499,structure=np.ones((3,3)))[1] for x in images.reshape(-1,1,32,32)],np.repeat(COUNTS,k))
    assert np.bincount(d['y']).tolist()==[1000,1000] and not np.any((d['a']==d['b'])&(d['ia']==d['ib']))
    for aa in range(5):
        for bb in range(5):assert ((d['a']==aa)&(d['b']==bb)).sum()==50*(4 if aa==bb else 1)
    records[name]={'sha256':sha(p),'seed':seed,'trials':n,'images':1000,'objects_and_balance_verified':True}
    print('verified',name,flush=True)
m.update(banks=records,test_created_after_all_heads_locked=True,duplicate_rejection_note='duplicate_rejection.json');save(ROOT/'data_manifest.json',m)
save(ROOT/'duplicate_rejection.json',{'time':time.time(),'before_any_test_evaluation':True,'existing_banks_preserved':15,'replacement_rule':'Generate original Bank draw sequence, then continue same bank RNG for rejected same-count images until globally pixel-unique. Trial RNG unchanged.','replacements':rejected,'script_sha256':sha(__file__)})
print(json.dumps(rejected),flush=True)
