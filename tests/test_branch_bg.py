import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from branch_bg_solver import evaluate,fit_candidate

def test_sharp_temperature_hessian_and_fit():
 rng=np.random.default_rng(888);x=rng.normal(size=(60,3));y=rng.integers(0,2,60);w=np.full(60,8000/60);groups=[(0,10),(10,30),(30,60)];theta=rng.normal(size=4)*.1;eps=1e-6
 for tau in [.05,.005]:
  obj,g,H,_,_=evaluate(theta,x,y,w,'worst',10000,groups,tau,True)
  fd=np.column_stack([(evaluate(theta+np.eye(4)[i]*eps,x,y,w,'worst',10000,groups,tau)[1]-evaluate(theta-np.eye(4)[i]*eps,x,y,w,'worst',10000,groups,tau)[1])/(2*eps) for i in range(4)])
  np.testing.assert_allclose(H,fd,rtol=1e-5,atol=1e-7)
  clf,h=fit_candidate(x,y,w,10000,'worst',groups,tau)
  _,grad,_,_,_=evaluate(np.r_[clf.coef_[0],clf.intercept_],x,y,w,'worst',10000,groups,tau)
  assert max(abs(grad))<=1e-7 and h[-1]['passed']

def test_reconstruction_includes_blank_input_bias():
 import torch
 from crowbrain.model import VisualRNN
 from branch_bg_audit import reconstruct_clean
 torch.manual_seed(39);torch.set_num_threads(1);model=VisualRNN();model.eval()
 es=torch.randn(300,16);et=torch.randn(300,16)
 pre,end=reconstruct_clean(model,es,et,9)
 with torch.no_grad():
  _,state,_=model(es,et,delay=9,encoded=True,return_states=True)
 torch.testing.assert_close(pre,state[:,12],rtol=1e-6,atol=1e-6);torch.testing.assert_close(end,state[:,-1],rtol=1e-6,atol=1e-6)
