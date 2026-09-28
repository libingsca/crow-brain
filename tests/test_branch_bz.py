import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import numpy as np
from branch_bz_analyze import endpoints,classify,interval

def test_refit_variation_not_bank_pseudoreplication():
 a=np.empty((5,5,4,11,2,2))
 for r in range(5):a[r]=.8+.02*r
 panel,individual,_=endpoints(a)
 expected=interval(np.arange(5)*.02+.8)
 np.testing.assert_allclose([panel['d5_clean'][k] for k in ['mean','lower','upper']],[expected[k] for k in ['mean','lower','upper']])
 assert panel['d5_clean']['n_refits']==5

def test_minimum_before_refit_average_and_fixed_core_identity():
 a=np.full((5,5,4,11,2,2),.95)
 for r in range(5):a[r,0,:,r,1,0]=.7
 panel,individual,_=endpoints(a)
 assert abs(individual['301']['mean']-.7)<1e-12
 assert individual['302']['mean']==.95
 assert classify(panel)['decision']=='supported'
 assert classify(individual)['decision']=='excluded'

def test_threshold_equal_is_unresolved():
 assert classify({'x':{'lower':.85,'upper':.9}})['decision']=='inconclusive'
