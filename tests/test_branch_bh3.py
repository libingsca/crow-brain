import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import t,sem
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from branch_bh3_analyze import interval

def test_remaining_condition_uses_160_banks_and_allocated_alpha():
 v=np.random.default_rng(1603).normal(.8513,.0036,160);r=interval(v)
 expected=t.interval(.975,159,loc=v.mean(),scale=sem(v))
 np.testing.assert_allclose([r['lower'],r['upper']],expected,atol=1e-14,rtol=0)
 assert r['n_banks']==160 and r['confidence']==.975
 with pytest.raises(AssertionError):interval(v[:80])
