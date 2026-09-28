import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import numpy as np
import pytest
import branch_bs_round1 as bs

def test_pooling_preserves_trial_and_delay_weights():
    y=np.array([0,1,0,1]);raw={d:{'end':np.full((4,3),d),'noise_end':np.full((4,4,3),d+.1)} for d in bs.TRAIN_DELAYS}
    for name in ['clean','aug']:
        x,yy,w=bs.pooled_pack(raw,y,name)
        np.testing.assert_allclose(w.reshape(-1,4).sum(0),np.ones(4))
        np.testing.assert_allclose(w.reshape(5,-1).sum(1),np.full(5,.8))
        np.testing.assert_array_equal(yy,np.tile(y,5 if name=='clean' else 25))
        if name=='aug':
            ww=w.reshape(5,5,4);np.testing.assert_allclose(ww[:,0],.1);np.testing.assert_allclose(ww[:,1:],.025)

def test_same_shared_objects_across_all_delays():
    new={'clean':object(),'aug':object()};refs={f'ref{d}':object() for d in bs.TRAIN_DELAYS}
    for d in bs.DELAYS:
        active=bs.active_heads(new,refs,d)
        assert active['clean'] is new['clean'] and active['aug'] is new['aug']
        assert ('reference' in active)==(d in bs.TRAIN_DELAYS)
    assert set(bs.TRAIN_DELAYS).isdisjoint(bs.HELDOUT)
    assert set(bs.TRAIN_DELAYS+bs.HELDOUT)==set(bs.DELAYS)

def test_test_generation_requires_lock(tmp_path,monkeypatch):
    monkeypatch.setattr(bs,'ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):bs.prepare(True)
    assert not list(tmp_path.iterdir())
