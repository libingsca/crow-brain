import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from branch_bi3_analyze import gate_classify,main_decision
from branch_bi_statistics import rate

def test_gate_success_is_not_full_success():
 g=rate(['S']*96,r=3);assert g['decision']=='supported';assert main_decision(g)=='inconclusive'
def test_necessary_failure_suffices():
 g=rate(['F']*32+['U']*64,r=3);assert main_decision(g)=='excluded'
 assert main_decision(rate(['F']*31+['U']*65,r=3))=='inconclusive'
def test_gate_boundaries():
 assert gate_classify([(.85,.9)]*22)=='U'
 assert gate_classify([(.851,.9)]*22)=='S'
 assert gate_classify([(.8,.849)]+[(.9,.95)]*21)=='F'

def test_bank_shards_cover_exactly_once():
 from branch_bi3_round import bank_indices
 for n in [1,5,640]:
  a=list(bank_indices(0,n));b=list(bank_indices(1,n));assert a+b==list(range(n)) and not set(a)&set(b)
