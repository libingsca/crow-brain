import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from branch_b_round1 import Probe
from branch_bi3_vectorized import FastPredictor
from branch_bi3_vectorized_audit import explicit_predict

def heads_fixture():
 rng=np.random.default_rng(765);heads={}
 for i in range(17):
  scaler=StandardScaler().fit(rng.normal(size=(100,64))*rng.uniform(.001,10,64)+rng.normal(size=64));clf=LogisticRegression();clf.coef_=rng.normal(size=(1,64))*5;clf.intercept_=rng.normal(size=1);clf.classes_=np.array([0,1]);clf.n_features_in_=64;heads[str(i)]=Probe(scaler,clf)
 return heads

def test_predictions_match_original_and_explicit():
 h=heads_fixture();rng=np.random.default_rng(66);x=rng.normal(size=(2000,64)).astype(np.float32);expected=np.array([p.predict(x) for p in h.values()]);np.testing.assert_array_equal(FastPredictor(h).predict(x),expected);np.testing.assert_array_equal(explicit_predict(x,h),expected)

def test_near_boundary_uses_original():
 h=heads_fixture();xs=[]
 for p in h.values():
  x=p.scaler.mean_.copy();j=np.argmax(np.abs(p.clf.coef_[0]));x[j]-=p.clf.intercept_[0]*p.scaler.scale_[j]/p.clf.coef_[0,j];xs.append(x)
 x=np.array(xs);fast=FastPredictor(h);expected=np.array([p.predict(x) for p in h.values()]);np.testing.assert_array_equal(fast.predict(x),expected);assert fast.fallback_heads>0;np.testing.assert_array_equal(explicit_predict(x,h),expected)

def test_float32_scaler_rounding_is_preserved():
 scaler=StandardScaler();scaler.mean_=np.array([1.23456789]);scaler.scale_=np.array([.123456789]);scaler.var_=scaler.scale_**2;scaler.n_features_in_=1;scaler.n_samples_seen_=100
 x=np.array([[2.0000001]],np.float32);a=float(scaler.transform(x)[0,0]);b=float(scaler.transform(x.astype(np.float64))[0,0]);assert a!=b
 clf=LogisticRegression();clf.coef_=np.array([[1.]]);clf.intercept_=np.array([-(a+b)/2]);clf.classes_=np.array([0,1]);clf.n_features_in_=1;h=Probe(scaler,clf);assert h.predict(x)[0]!=h.predict(x.astype(np.float64))[0]
 fp=FastPredictor({'h':h});np.testing.assert_array_equal(fp.predict(x)[0],h.predict(x));assert fp.fallback_heads==1;np.testing.assert_array_equal(explicit_predict(x,{'h':h})[0],h.predict(x))
