import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from branch_bn_round1 import mixture,encode_states,noisy
from crowbrain.model import VisualRNN

def test_collapsed_duplicates_preserve_regularized_objective():
    rng=np.random.default_rng(61);clean=rng.normal(size=(80,6));noise=[clean+rng.normal(size=clean.shape)*.1 for _ in range(4)];y=np.tile([0,1],40)
    x,w=mixture(clean,noise)
    a=LogisticRegression(C=.3,tol=1e-10,max_iter=3000).fit(x,np.tile(y,5),sample_weight=w)
    full=np.concatenate([clean]*4+noise)
    b=LogisticRegression(C=.3,tol=1e-10,max_iter=3000).fit(full,np.tile(y,8),sample_weight=np.full(640,1/8))
    np.testing.assert_allclose(a.coef_,b.coef_,atol=1e-7)
    np.testing.assert_allclose(a.intercept_,b.intercept_,atol=1e-7)
    assert w.sum()==len(clean)

def test_variable_bank_size_and_zero_noise_continuation():
    torch.manual_seed(71);model=VisualRNN(normalize=True).eval();rng=np.random.default_rng(7)
    d={'images':rng.random((5,2,1,32,32),dtype=np.float32),'a':np.arange(5),'b':np.arange(5)[::-1].copy(),'ia':np.zeros(5,int),'ib':np.ones(5,int)}
    pre,end,et=encode_states(model,d)
    with torch.no_grad():
        _,h,_=model(torch.from_numpy(d['images'][d['a'],d['ia']]),torch.from_numpy(d['images'][d['b'],d['ib']]),delay=15,return_states=True)
    np.testing.assert_allclose(end,h[:,-1].numpy(),rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(noisy(model,pre,et,1,0,0)[1],end)
    a=noisy(model,pre,et,1,0,.01)[0];b=noisy(model,pre,et,1,0,.01)[0];c=noisy(model,pre,et,2,0,.01)[0]
    np.testing.assert_array_equal(a,b);assert not np.array_equal(a,c)

def test_newton_repair_solves_same_weighted_objective():
    from branch_bn_newton import optimize
    from diagnose_bn_solver import objective
    rng=np.random.default_rng(99);x=rng.normal(size=(120,5));y=rng.integers(0,2,120);w=rng.uniform(.1,1,120)
    old=LogisticRegression(C=.4,tol=1e-10,max_iter=3000).fit(x,y,sample_weight=w)
    new,hist=optimize(x,y,w,.4)
    assert hist[-1]['converged']
    np.testing.assert_allclose(old.predict_proba(x),new.predict_proba(x),atol=1e-6)
    assert abs(objective(x,y,w,old,.4)[0]-objective(x,y,w,new,.4)[0])<1e-10
    assert objective(x,y,w,new,.4)[1]<1e-6
