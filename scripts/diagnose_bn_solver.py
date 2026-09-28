import json
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from branch_bn_round1 import encode_states,noisy,mixture,save
from branch_bn_newton import optimize
from crowbrain.model import VisualRNN

def objective(x,y,w,clf,c):
    beta=clf.coef_[0];z=x@beta+clf.intercept_[0]
    loss=float(np.sum(w*(np.logaddexp(0,z)-y*z))/w.sum()+np.dot(beta,beta)/(2*c*w.sum()))
    err=w*(expit(z)-y)/w.sum();grad=np.r_[x.T@err+beta/(c*w.sum()),err.sum()]
    return loss,float(np.max(abs(grad)))

def main():
    torch.set_num_threads(1);rng=np.random.default_rng(13);x=rng.normal(size=(500,8));y=rng.integers(0,2,500);w=rng.uniform(.2,1,500);c=.3
    lb=LogisticRegression(C=c,tol=1e-10,max_iter=3000).fit(x,y,sample_weight=w);nt,h=optimize(x,y,w,c)
    np.testing.assert_allclose(lb.predict_proba(x),nt.predict_proba(x),atol=1e-6)
    assert abs(objective(x,y,w,lb,c)[0]-objective(x,y,w,nt,c)[0])<1e-10
    d=dict(np.load('artifacts/branch_bn/round1/fit.npz'));model=VisualRNN(normalize=True);model.load_state_dict(torch.load('artifacts/round6/final/rnn_lr0.001_seed302/final.pt',weights_only=True));model.eval();model.requires_grad_(False)
    pre,end,et=encode_states(model,d);sigma=.01*json.loads(Path('artifacts/branch_b/round1/seed302/result.json').read_text())['noise_scale_rms']
    noise=[noisy(model,pre,et,int(d['seed']),r,sigma)[1] for r in range(4)];xx,ww=mixture(end,noise);sc=StandardScaler().fit(end);xx=sc.transform(xx).astype(np.float64);yy=np.tile(d['y'],5)
    clf,h=optimize(xx,yy,ww,1);loss,grad=objective(xx,yy,ww,clf,1);assert grad<1e-6
    out={'synthetic_objective_and_probabilities_agree':True,'actual_model':302,'C':1,'objective':loss,'max_absolute_mean_objective_gradient':grad,'history':h}
    save('artifacts/branch_bn/round1/solver_audit.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
