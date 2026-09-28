import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import numpy as np
import pytest
import torch
import branch_bl_round1 as bl
from branch_bt_round1 import encoded_pairs,extract,perturb,check_forward
from crowbrain.model import VisualRNN

@pytest.mark.parametrize('delay',[8,11,12])
def test_bl_time_indices_pixel_and_reset(delay):
    torch.manual_seed(124);model=VisualRNN(normalize=True).eval();model.requires_grad_(False)
    rng=np.random.default_rng(224)
    d={'images':rng.random((5,3,1,32,32),dtype=np.float32),'a':np.arange(5),'b':np.arange(5)[::-1].copy(),'ia':np.zeros(5,int),'ib':np.ones(5,int)}
    es,et=encoded_pairs(model,d);pre,end=extract(model,es,et,delay)
    check_forward(model,d,es,et,delay,pre,end)
    _,zero=perturb(model,pre,et,950100,0,0);torch.testing.assert_close(zero,end)

def test_bl_test_data_requires_audited_lock(tmp_path,monkeypatch):
    monkeypatch.setattr(bl,'ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):bl.prepare(test=True)
    assert not list(tmp_path.iterdir())

def test_bl_train_refuses_test_files(tmp_path,monkeypatch):
    monkeypatch.setattr(bl,'ROOT',tmp_path);(tmp_path/'test00.npz').touch()
    with pytest.raises(AssertionError):bl.train(301)
    assert not (tmp_path/'seed301').exists()

def test_bl_analysis_rejects_incomplete_matrix(tmp_path,monkeypatch):
    import branch_bl_analyze as analysis
    monkeypatch.setattr(analysis,'ROOT',tmp_path)
    for name in ['training_audit','independent_audit','data_chronology_integrity']:(tmp_path/f'{name}.json').write_text('{"passed":true}')
    (tmp_path/'seed301').mkdir();(tmp_path/'seed301/results.json').write_text('{"cells":[]}')
    with pytest.raises(AssertionError):analysis.main()
    assert not (tmp_path/'summary.json').exists()
