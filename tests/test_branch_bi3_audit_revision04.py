import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from branch_b_round1 import Probe
from branch_bi3_audit_revision04 import explicit_predict,permutation_means
from branch_bi3_vectorized import FastPredictor

def test_scaler_parameter_cast_changes_boundary_label():
 rng=np.random.default_rng(445)
 for _ in range(100):
  sc=StandardScaler().fit(rng.normal(size=(50,1))*rng.uniform(.1,10)+rng.uniform(-10,10));x=rng.normal(size=(1,1)).astype(np.float32)
  actual=float(sc.transform(x)[0,0]);old=x.copy();old-=sc.mean_;old/=sc.scale_;incorrect=float(old[0,0])
  if actual!=incorrect:break
 assert actual!=incorrect
 clf=LogisticRegression();clf.coef_=np.array([[1.]]);clf.intercept_=np.array([-(actual+incorrect)/2]);clf.classes_=np.array([0,1]);clf.n_features_in_=1;h=Probe(sc,clf)
 expected=h.predict(x)
 assert bool(incorrect+clf.intercept_[0]>0)!=bool(expected[0])
 np.testing.assert_array_equal(explicit_predict(x,{'h':h})[0],expected)
 np.testing.assert_array_equal(FastPredictor({'h':h}).predict(x)[0],expected)

def test_permutation_counts_match_all_explicit_replicates():
 rng=np.random.default_rng(98);n=2000;y=np.repeat([0,1],n//2);rng.shuffle(y)
 yi=np.array([rng.permutation(n) for _ in range(64)]);ri=np.array([rng.permutation(n) for _ in range(64)])
 raw={'predictions':rng.integers(0,2,(7,11,n),dtype=np.int8),'reset':rng.integers(0,2,(7,2,n),dtype=np.int8)}
 expected=np.empty((7,2,3))
 for j in range(7):
  for c in range(2):
   p=raw['predictions'][j,c];expected[j,c]=[(raw['reset'][j,c]==y).mean(),np.mean([(p==y[ix]).mean() for ix in yi]),np.mean([(p[ix]==y).mean() for ix in ri])]
 np.testing.assert_allclose(permutation_means(raw,y,yi,ri),expected,rtol=0,atol=1e-15)
