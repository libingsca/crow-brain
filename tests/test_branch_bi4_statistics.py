import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy.stats import beta,t
from branch_bi4_statistics import prefix_update,upper_ci

def test_unpassed_gate_blocks_later_small_p():
 assert prefix_update(0,[.849,.851,.848])==1
 assert prefix_update(0,[.85,.849])==0

def test_prior_certificates_are_retained_between_valid_looks():
 assert prefix_update(1,[.86,.849,.849])==3
 assert prefix_update(1,[.86,.851,.849])==1

def test_fixed_first_true_null_is_necessary_for_any_false_prefix():
 for first_true in range(4):
  for mask in range(16):
   reject=[bool(mask&(1<<i)) for i in range(4)];n=prefix_update(0,[.84 if x else .86 for x in reject])
   if n>first_true:assert reject[first_true]

def test_round4_budget_and_count_boundary():
 assert abs(sum([.1,.15,.25,.5])-1)<1e-15
 assert beta.ppf(1-.0015625/2,97-32,32)>=.8
 assert beta.ppf(1-.0015625/2,97-33,33)<.8

def test_one_sided_mean_ci_uses_banks():
 x=np.array([.84,.85,.86,.87]);a=.01;ci=upper_ci(x,a)
 np.testing.assert_allclose(ci['upper'],x.mean()+t.isf(a,3)*x.std(ddof=1)/2);assert ci['n_banks']==4
