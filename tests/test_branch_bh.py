import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import t,sem
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from branch_bh2_analyze import interval

def test_eighty_bank_confirmation_interval_and_sample_size():
 v=np.random.default_rng(802).normal(.86,.008,80);r=interval(v)
 lo,hi=t.interval(1-.05/110,79,loc=v.mean(),scale=sem(v))
 np.testing.assert_allclose([r['lower'],r['upper']],[lo,hi],atol=1e-14,rtol=0)
 assert r['n_banks']==80
 with pytest.raises(AssertionError):interval(v[:20])
