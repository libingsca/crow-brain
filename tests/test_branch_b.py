"""Check the intervention boundary used by branch B, independent of data fitting."""
import importlib.util
from pathlib import Path
import torch
import pytest
from crowbrain.model import VisualRNN

spec=importlib.util.spec_from_file_location('branch_b_round1',Path(__file__).parents[1]/'scripts/branch_b_round1.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

@pytest.mark.parametrize('delay',[0,5,15,40])
def test_continuation_and_reset_match_original_forward(delay):
    torch.manual_seed(71)
    model=VisualRNN(normalize=True).eval()
    sample,test=torch.randn(12,16),torch.randn(12,16)
    with torch.no_grad():
        logits,states,_=model(sample,test,delay=delay,encoded=True,return_states=True)
        pre=states[:,3+delay]
        continued=b.continue_test(model,pre,test)
        assert torch.equal(continued,states[:,-1])
        assert torch.equal(model.readout(continued),logits)
        reset_logits=model(sample,test,delay=delay,encoded=True,reset=True)
        assert torch.equal(model.readout(b.continue_test(model,torch.zeros_like(pre),test)),reset_logits)
