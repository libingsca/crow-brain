import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import numpy as np
import pytest
import branch_bc_round1 as bc
from branch_bc_analyze import classify

@pytest.mark.parametrize('method',bc.HEADS)
def test_coverage_and_noise_weights_preserve_each_original_trial(method):
 y=np.array([0,1,0,1]);raw={d:{'end':np.full((4,3),d),'noise_end':np.full((4,4,3),d+.1)} for d in bc.TRAIN_DELAYS}
 ds=bc.BASE_DELAYS if method.startswith('base_') else bc.TRAIN_DELAYS
 x,yy,w=bc.pooled_pack(raw,y,method,ds);m=len(ds)
 np.testing.assert_allclose(w.reshape(-1,4).sum(0),1,atol=1e-14)
 np.testing.assert_allclose(w.reshape(m,-1).sum(1),4/m)
 np.testing.assert_array_equal(x[:,0].astype(int),np.repeat(ds,4*(5 if method.endswith('aug') else 1)))
 vx,vy,vw=bc.pooled_pack(raw,y,method,bc.TRAIN_DELAYS)
 assert set(vx[:,0].astype(int))==set(bc.TRAIN_DELAYS)
 np.testing.assert_allclose(vw.reshape(-1,4).sum(0),1)
 if method.endswith('aug'):
  a=w.reshape(m,5,4);np.testing.assert_allclose(a[:,0],.5/m);np.testing.assert_allclose(a[:,1:],.125/m)

def test_fixed_shared_objects_across_test_delays():
 new={n:object() for n in bc.HEADS};refs={f'ref{d}':object() for d in bc.TRAIN_DELAYS}
 for d in bc.DELAYS:
  active=bc.active_heads(new,refs,d)
  assert all(active[n] is new[n] for n in bc.HEADS)
  assert ('reference' in active)==(d in bc.TRAIN_DELAYS)

def test_test_data_requires_audited_lock(tmp_path,monkeypatch):
 monkeypatch.setattr(bc,'ROOT',tmp_path)
 with pytest.raises(FileNotFoundError):bc.prepare(True)
 assert not list(tmp_path.iterdir())

def test_training_refuses_test_data(tmp_path,monkeypatch):
 monkeypatch.setattr(bc,'ROOT',tmp_path);(tmp_path/'test00.npz').touch()
 with pytest.raises(AssertionError):bc.train(301)
 assert not (tmp_path/'seed301').exists()

def test_absolute_success_does_not_override_gain_failure():
 a={'absolute':{'lower':.9,'upper':.95,'threshold':.85},'gain':{'lower':.005,'upper':.015,'threshold':.02}}
 assert classify(a)['decision']=='excluded'
 a['gain'].update(lower=.02,upper=.03)
 assert classify(a)['decision']=='inconclusive'
 a['gain'].update(lower=.021,upper=.03)
 assert classify(a)['decision']=='supported'
