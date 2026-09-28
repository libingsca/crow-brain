"""Convex mean / fixed-temperature smooth-max group logistic objectives."""
import warnings
import numpy as np
from scipy.special import expit,logsumexp
from scipy.linalg import solve,LinAlgWarning
from sklearn.linear_model import LogisticRegression
TAU=.05

def evaluate(theta,x,y,w,kind,c,groups=8,tau=TAU,hessian=False):
    assert kind in ['mean','worst'] and tau>0 and c>0
    n,p=x.shape; bounds=group_bounds(n,groups);counts=np.array([b-a for a,b in bounds]);groups=len(bounds)
    within=np.concatenate([w[a:b]/w[a:b].sum() for a,b in bounds])
    z=x@theta[:p]+theta[p];prob=expit(z)
    losses=np.array([np.sum(within[a:b]*(np.logaddexp(0,z[a:b])-y[a:b]*z[a:b])) for a,b in bounds])
    if kind=='mean':risk=float(losses.mean());q=np.full(groups,1/groups)
    else:
        risk=float(tau*(logsumexp(losses/tau)-np.log(groups)))
        q=np.exp(losses/tau-logsumexp(losses/tau))
    group_grad=[]
    for a,b in bounds:
        sl=slice(a,b);e=within[sl]*(prob[sl]-y[sl])
        group_grad.append(np.r_[x[sl].T@e,e.sum()])
    group_grad=np.asarray(group_grad);gbar=q@group_grad;lam=1/(c*w.sum())
    gradient=gbar.copy();gradient[:p]+=lam*theta[:p]
    objective=float(risk+.5*lam*np.dot(theta[:p],theta[:p]))
    H=None
    if hessian:
        curv=within*np.repeat(q,counts)*prob*(1-prob)
        H=np.empty((p+1,p+1));H[:p,:p]=x.T@(x*curv[:,None]);H[:p,p]=x.T@curv;H[p,:p]=H[:p,p];H[p,p]=curv.sum()
        H[:p,:p]+=np.eye(p)*lam
        if kind=='worst':
            centered=group_grad-gbar
            H+=(centered.T*q)@centered/tau
        H=(H+H.T)/2
    return objective,gradient,H,losses,q

def group_bounds(n,groups):
    if isinstance(groups,int):
        assert n%groups==0;return [(i*n//groups,(i+1)*n//groups) for i in range(groups)]
    bounds=list(groups);assert bounds[0][0]==0 and bounds[-1][1]==n
    assert all(a<b for a,b in bounds) and all(bounds[i][1]==bounds[i+1][0] for i in range(len(bounds)-1))
    return bounds

def value(theta,x,y,w,kind,c,groups=8,tau=TAU):
    z=x@theta[:-1]+theta[-1];losses=np.array([np.average(np.logaddexp(0,z[a:b])-y[a:b]*z[a:b],weights=w[a:b]) for a,b in group_bounds(len(x),groups)])
    risk=losses.mean() if kind=='mean' else tau*(logsumexp(losses/tau)-np.log(len(losses)))
    return float(risk+np.dot(theta[:-1],theta[:-1])/(2*c*w.sum()))

def fit_candidate(x,y,w,c,kind,groups=8):
    theta=np.zeros(x.shape[1]+1);history=[]
    for it in range(1001):
        obj,g,H,losses,q=evaluate(theta,x,y,w,kind,c,groups=groups,hessian=True)
        gn=float(np.abs(g).max());passed=gn<=1e-7
        row={'iteration':it,'phase_limit':200 if it<200 else 1000,'objective':obj,'gradient_inf':gn,'group_losses':losses.tolist(),'group_weights':q.tolist(),'passed':passed}
        history.append(row)
        if passed:
            clf=LogisticRegression(C=c,solver='newton-cholesky',tol=1e-7,max_iter=1000)
            clf.coef_=theta[:-1][None,:].copy();clf.intercept_=theta[-1:].copy();clf.classes_=np.array([0,1]);clf.n_features_in_=x.shape[1];clf.n_iter_=np.array([it])
            return clf,history
        if it==1000:break
        damping=0.;direction=None
        for attempt in range(12):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error',LinAlgWarning)
                    direction=solve(H+damping*np.eye(len(theta)),-g,assume_a='pos')
                if np.isfinite(direction).all() and float(g@direction)<0:break
            except (np.linalg.LinAlgError,LinAlgWarning):pass
            direction=None;damping=1e-10 if damping==0 else damping*10
        if direction is None:raise RuntimeError(f'No descent direction at {it}, gradient={gn}')
        step=1.;slope=float(g@direction)
        for backtrack in range(50):
            trial=theta+step*direction
            if value(trial,x,y,w,kind,c,groups=groups)<=obj+1e-4*step*slope:break
            step*=.5
        else:raise RuntimeError(f'Armijo failed at {it}, gradient={gn}')
        row.update(damping=damping,step=step,backtracks=backtrack)
        theta=trial
    raise RuntimeError(f'Unconverged {kind} C={c}, final gradient={history[-1]["gradient_inf"]}')
