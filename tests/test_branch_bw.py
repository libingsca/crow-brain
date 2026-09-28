import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import numpy as np
import pytest
import branch_bw_round1 as bw
from branch_bw_analyze import classify

@pytest.mark.parametrize('method',bw.HEADS)
def test_coverage_and_noise_weights_preserve_each_original_trial(method):
 y=np.array([0,1,0,1]);raw={d:{'end':np.full((4,3),d),'noise_end':np.full((4,4,3),d+.1)} for d in bw.TRAIN_DELAYS}
 ds=bw.TRAIN_DELAYS
 x,yy,w=bw.pooled_pack(raw,y,method,ds);m=len(ds)
 np.testing.assert_allclose(w.reshape(-1,4).sum(0),1,atol=1e-14)
 np.testing.assert_allclose(w.reshape(m,-1).sum(1),4/m)
 np.testing.assert_array_equal(x[:,0].astype(int),np.repeat(ds,4*(5 if method.endswith('aug') else 1)))
 vx,vy,vw=bw.pooled_pack(raw,y,method,bw.TRAIN_DELAYS)
 assert set(vx[:,0].astype(int))==set(bw.TRAIN_DELAYS)
 np.testing.assert_allclose(vw.reshape(-1,4).sum(0),1)
 if method.endswith('aug'):
  a=w.reshape(m,5,4);np.testing.assert_allclose(a[:,0],.5/m);np.testing.assert_allclose(a[:,1:],.125/m)

def test_fixed_shared_objects_across_test_delays():
 new={n:object() for n in bw.HEADS};refs={f'ref{d}':object() for d in bw.TRAIN_DELAYS}
 for d in bw.DELAYS:
  active=bw.active_heads(new,refs,d)
  assert all(active[n] is new[n] for n in bw.HEADS)
  assert ('reference' in active)==(d in bw.TRAIN_DELAYS)

def test_test_data_requires_audited_lock(tmp_path,monkeypatch):
 monkeypatch.setattr(bw,'ROOT',tmp_path)
 with pytest.raises(FileNotFoundError):bw.prepare(True)
 assert not list(tmp_path.iterdir())

def test_training_refuses_test_data(tmp_path,monkeypatch):
 monkeypatch.setattr(bw,'ROOT',tmp_path);(tmp_path/'test00.npz').touch()
 with pytest.raises(AssertionError):bw.train(301)
 assert not (tmp_path/'seed301').exists()

def test_absolute_success_does_not_override_gain_failure():
 a={'absolute':{'lower':.9,'upper':.95,'threshold':.85},'gain':{'lower':.005,'upper':.015,'threshold':.02}}
 assert classify(a)['decision']=='excluded'
 a['gain'].update(lower=.02,upper=.03)
 assert classify(a)['decision']=='inconclusive'
 a['gain'].update(lower=.021,upper=.03)
 assert classify(a)['decision']=='supported'

from branch_bw_solver import evaluate,value,fit_candidate
from sklearn.linear_model import LogisticRegression

@pytest.mark.parametrize('kind',['mean','worst'])
def test_analytic_gradient_and_hessian_against_finite_differences(kind):
 rng=np.random.default_rng(882);x=rng.normal(size=(80,4));y=rng.integers(0,2,80);w=rng.uniform(.2,1,80);theta=rng.normal(size=5)*.3;c=.7
 f,g,h,losses,q=evaluate(theta,x,y,w,kind,c,hessian=True)
 eps=1e-5;eye=np.eye(5)*eps
 fd=np.array([(value(theta+e,x,y,w,kind,c)-value(theta-e,x,y,w,kind,c))/(2*eps) for e in eye])
 hd=np.column_stack([(evaluate(theta+e,x,y,w,kind,c)[1]-evaluate(theta-e,x,y,w,kind,c)[1])/(2*eps) for e in eye])
 np.testing.assert_allclose(g,fd,rtol=1e-5,atol=1e-7);np.testing.assert_allclose(h,hd,rtol=1e-5,atol=1e-7)
 assert np.linalg.eigvalsh(h).min()>-1e-10
 if kind=='worst':
  risk=f-(theta[:-1]@theta[:-1])/(2*c*w.sum())
  assert max(losses)-.05*np.log(8)<=risk<=max(losses)+1e-14
  np.testing.assert_allclose(q.sum(),1)

def test_mean_solver_matches_existing_logistic_objective_and_predictions():
 rng=np.random.default_rng(883);x=rng.normal(size=(160,5));y=rng.integers(0,2,160);w=np.ones(160);c=.7
 actual,h=fit_candidate(x,y,w,c,'mean')
 reference=LogisticRegression(C=c,solver='newton-cholesky',tol=1e-10,max_iter=1000).fit(x,y,sample_weight=w)
 np.testing.assert_allclose(actual.predict_proba(x),reference.predict_proba(x),atol=1e-6)
 theta=np.r_[reference.coef_[0],reference.intercept_]
 assert abs(value(theta,x,y,w,'mean',c)-h[-1]['objective'])<1e-10
 assert h[-1]['gradient_inf']<=1e-7

def test_smooth_solver_converges_and_shifts_weight_to_high_loss_groups():
 rng=np.random.default_rng(884);x=rng.normal(size=(160,4));y=(x[:,0]>0).astype(int);y[:40]=rng.integers(0,2,40);w=np.ones(160)
 clf,h=fit_candidate(x,y,w,1.,'worst');v,g,_,losses,q=evaluate(np.r_[clf.coef_[0],clf.intercept_],x,y,w,'worst',1.)
 assert max(abs(g))<=1e-7 and np.argmax(losses)==np.argmax(q)
 assert h[-1]['objective']<=h[0]['objective']

def test_intercept_is_not_regularized():
 rng=np.random.default_rng(885);x=rng.normal(size=(80,3));y=rng.integers(0,2,80);w=np.ones(80);theta=np.array([.1,.2,.3,.7])
 a,ga,_,_,_=evaluate(theta,x,y,w,'worst',.1)
 b,gb,_,_,_=evaluate(theta,x,y,w,'worst',10.)
 np.testing.assert_allclose(ga[-1],gb[-1],atol=1e-14)
 np.testing.assert_allclose(a-b,(theta[:-1]@theta[:-1])/(2*len(y))*(1/.1-1/10))
