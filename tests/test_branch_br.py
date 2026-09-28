import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import numpy as np
import pytest
import branch_br_round1 as br

def test_augmentation_preserves_each_trials_weight():
    clean=np.arange(24,dtype=float).reshape(8,3)
    noise=[clean+i for i in range(1,5)]
    x,w=br.mixture(clean,noise)
    np.testing.assert_array_equal(w.reshape(5,8).sum(0),np.ones(8))
    np.testing.assert_array_equal(w[:8],np.full(8,.5))
    assert x.shape==(40,3)

def test_test_data_generation_requires_completed_training_lock(tmp_path,monkeypatch):
    monkeypatch.setattr(br,'ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):br.prepare(test=True)
    assert not list(tmp_path.iterdir())

def test_training_refuses_preexisting_test_data(tmp_path,monkeypatch):
    monkeypatch.setattr(br,'ROOT',tmp_path);(tmp_path/'test00.npz').touch()
    with pytest.raises(AssertionError):br.train(301)
    assert not (tmp_path/'seed301').exists()
