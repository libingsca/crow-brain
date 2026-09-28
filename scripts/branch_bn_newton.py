"""Uniform BN002 solver override; all scientific conditions stay in branch_bn_round1."""
import argparse,json,warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import branch_bn_round1 as b
from branch_bn_round1 import Probe

def optimize(xx,y,w,c):
    hist=[]
    for limit in [200,1000]:
        clf=LogisticRegression(C=c,solver='newton-cholesky',tol=1e-7,max_iter=limit)
        with warnings.catch_warnings(record=True) as caught:warnings.simplefilter('always');clf.fit(xx,y,sample_weight=w)
        ok=not any(issubclass(z.category,ConvergenceWarning) for z in caught)
        hist.append({'solver':'newton-cholesky','max_iter':limit,'iterations':clf.n_iter_.tolist(),'converged':ok,'warnings':[str(z.message) for z in caught]})
        if ok:return clf,hist
    raise RuntimeError(f'Newton solver did not converge C={c}')

def main(seed):
    original_save=b.save
    def save(path,x):
        if Path(path).name=='manifest.json':x={**x,'solver_override_path':str(Path(__file__)),'solver_override_sha256':b.sha(__file__),'solver_protocol_sha256':b.sha('docs/decisions/branch_bn/002_solver_repair.md')}
        original_save(path,x)
    b.optimize=optimize;b.save=save
    try:b.run(seed)
    except Exception as e:save(b.R/f'seed{seed}'/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);main(p.parse_args().seed)
