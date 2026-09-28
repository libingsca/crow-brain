import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy.stats import binom
from branch_bi5_statistics import reused_rate,union_states
from branch_bi5_common import CFG,HEADS,BY_DELAY,CANDIDATE_PAIRS,bank_path,bounds

def test_reused_count_event_is_monotone_for_every_latent_count():
 uppers=[reused_rate(['F']*f+['U']*(96-f))['upper'] for f in range(97)]
 assert all(uppers[f]>=uppers[f+1] for f in range(96))
 assert uppers[31]>=.8 and uppers[32]<.8
 for truth in range(97):
  for certificate in range(truth+1):assert uppers[certificate]>=uppers[truth]

def test_original_cp_upper_coverage_at_probability_grid():
 alpha=.003125
 for q in np.linspace(.01,.99,99):
  probability=sum(binom.pmf(96-f,96,q) for f in range(97) if reused_rate(['F']*f+['U']*(96-f))['upper']<q)
  assert probability<=alpha/2+1e-12

def test_duplicate_conditions_do_not_add_fitting_units():
 old={str(i):'U' for i in range(1,97)};old['1']='F';old['2']='S';combined=union_states(old,['b1','b3','b3']);r=reused_rate(list(combined.values()));assert r['K']==96 and r['F']==2 and r['S']==1

def test_family_is_full_original_uncertainty_and_all_original_looks():
 assert len(CFG['targets'])==68 and len(HEADS)==47 and CFG['family_look_sizes']==[1024,2048,4096,8192]
 slots={(x['head'],x['delay'],x['input']) for x in CFG['targets']};assert len(slots)==68
 assert CANDIDATE_PAIRS==len({(h,d) for h,d,c in slots})
 for x in CFG['targets']:assert x['head'] in BY_DELAY[x['delay']]
 assert abs((CFG['measurement_alpha']/272)*68*4-CFG['measurement_alpha'])<1e-15

def test_existing_and_future_bank_prefix_identity():
 assert bounds(2)==(0,2048) and bounds(3)==(2048,4096) and bounds(4)==(4096,8192)
 assert 'round4/stage1' in str(bank_path(0)) and 'round4/stage2' in str(bank_path(2047)) and 'round5/look3' in str(bank_path(2048))
