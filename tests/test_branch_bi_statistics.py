import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from branch_bi_statistics import *
def test_uncertain_not_failure():
 assert rate(['U']*6)['upper']==1 and rate(['U']*6)['lower']==0
 assert rate(['F']*5+['U'])['decision']=='excluded'
def test_all_success_insufficient():
 assert rate(['S']*6)['decision']=='inconclusive'
 assert rate(['S']*23)['decision']=='supported'
def test_boundaries():
 assert classify([(.85,.9)]*110)=='U'
 assert classify([(.8,.85)]*110)=='U'
 assert classify([(.851,.9)]*110)=='S'
 assert classify([(.8,.849)]+[(.9,.95)]*109)=='F'
def test_alpha_spend():
 assert sum(sum(budgets(r)) for r in range(1,50))<=.05
 assert budgets(2)==(.00625,.00625)
def test_bank_unit():
 import pytest
 with pytest.raises(AssertionError):measurement(np.zeros((40,10)),6)
 lo,hi=measurement(np.linspace(.8,.9,40),6)
 assert lo<.85<hi
