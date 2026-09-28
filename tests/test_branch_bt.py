import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import numpy as np
import pytest
import torch
from crowbrain.model import VisualRNN
from branch_bt_round1 import encoded_pairs,extract,perturb,check_forward,permutation_indices,frozen_snapshot

@pytest.mark.parametrize('delay',[5,7,10,13,15])
def test_variable_delay_matches_pixel_forward_and_reset(delay):
    torch.manual_seed(122);model=VisualRNN(normalize=True).eval();model.requires_grad_(False)
    rng=np.random.default_rng(222)
    d={'images':rng.random((5,3,1,32,32),dtype=np.float32),'a':np.arange(5),'b':np.arange(5)[::-1].copy(),'ia':np.zeros(5,int),'ib':np.ones(5,int)}
    before=frozen_snapshot(model,{})
    es,et=encoded_pairs(model,d);pre,end=extract(model,es,et,delay)
    check_forward(model,d,es,et,delay,pre,end)
    _,zero=perturb(model,pre,et,920100,0,0)
    torch.testing.assert_close(zero,end)
    assert frozen_snapshot(model,{})==before


def test_negative_controls_preserve_strata_and_use_independent_permutations():
    d={'seed':920100,'y':np.tile([0,1],100),'b':np.repeat(np.arange(5),40)}
    yi,ri=permutation_indices(d)
    assert yi.shape==ri.shape==(64,200)
    for a,b in zip(yi,ri):
        np.testing.assert_array_equal(np.sort(a),np.arange(200))
        np.testing.assert_array_equal(np.sort(b),np.arange(200))
        np.testing.assert_array_equal(d['b'][b],d['b'])
    assert len({x.tobytes() for x in yi})==64
    assert len({x.tobytes() for x in ri})==64
