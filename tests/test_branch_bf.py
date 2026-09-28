import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import numpy as np
from branch_bf_solver import evaluate,value
from branch_bf_round1 import training_groups,active_heads,HEADS,DELAYS,REFERENCE_DELAYS

def test_unequal_group_analytic_derivatives():
    rng=np.random.default_rng(192);x=rng.normal(size=(21,4));y=rng.integers(0,2,21);w=rng.uniform(.1,1,21);theta=rng.normal(size=5)*.1;groups=[(0,3),(3,10),(10,21)]
    obj,g,h,_,q=evaluate(theta,x,y,w,'worst',.7,groups=groups,hessian=True);eps=1e-5
    ng=[];nh=[]
    for i in range(5):
        e=np.eye(5)[i]*eps
        ng.append((value(theta+e,x,y,w,'worst',.7,groups=groups)-value(theta-e,x,y,w,'worst',.7,groups=groups))/(2*eps))
        nh.append((evaluate(theta+e,x,y,w,'worst',.7,groups=groups)[1]-evaluate(theta-e,x,y,w,'worst',.7,groups=groups)[1])/(2*eps))
    np.testing.assert_allclose(g,ng,atol=1e-7);np.testing.assert_allclose(h,np.asarray(nh).T,atol=1e-6);assert abs(q.sum()-1)<1e-12

def test_group_partition_and_reference_routing():
    assert training_groups(2,[5,7],'base_split')==[(0,2),(2,10),(10,12),(12,20)]
    assert training_groups(2,[5,7],'base_mix')==[(0,10),(10,20)]
    new={n:object() for n in HEADS};refs={f'ref{d}':object() for d in REFERENCE_DELAYS}
    for d in DELAYS:
        a=active_heads(new,refs,d)
        assert all(a[n] is new[n] for n in HEADS)
        assert ('reference' in a)==(d in REFERENCE_DELAYS)

def test_mean_mixture_and_separate_groups_agree():
    rng=np.random.default_rng(23);x=rng.normal(size=(10,3));y=np.tile([0,1],5);w=np.r_[np.full(2,.5),np.full(8,.125)];theta=rng.normal(size=4)
    a=evaluate(theta,x,y,w,'mean',1,groups=[(0,10)],hessian=True)
    b=evaluate(theta,x,y,w,'mean',1,groups=[(0,2),(2,10)],hessian=True)
    for aa,bb in zip(a[:3],b[:3]):np.testing.assert_allclose(aa,bb,atol=1e-12)
