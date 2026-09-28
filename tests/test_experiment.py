import numpy as np
import torch
from crowbrain.task import Bank, balanced_pairs, draw_dots
from crowbrain.model import VisualRNN
from crowbrain.analysis import decode, tuning

def test_balance():
    pairs=balanced_pairs(3)
    y=pairs[:,0]==pairs[:,1]
    assert y.mean()==.5
    for axis in (0,1):
        for k in range(5): assert y[pairs[:,axis]==k].mean()==.5

def test_independent_images():
    bank=Bank(1,10)
    a,b,y,*_=bank.batch(np.random.default_rng(1),pairs=balanced_pairs(2))
    assert not torch.any(torch.all(a==b,dim=(1,2,3)))
    other=Bank(2,10)
    assert not torch.equal(bank.images,other.images)

def test_dots_and_protocols():
    from scipy.ndimage import label
    rng=np.random.default_rng(3)
    for mode in ['standard','area','shift']:
        for n in range(2,7):
            for _ in range(20):
                im,s=draw_dots(n,rng,mode)
                assert label(im>.5)[1]==n
                assert np.isfinite(s).all()

def test_reset_removes_sample_dependency():
    torch.manual_seed(1)
    m=VisualRNN()
    a=torch.rand(4,1,32,32); b=torch.rand_like(a)
    assert torch.equal(m(a,b,reset=True),m(a.flip(0),b,reset=True))

def test_dale_signs_and_gradients():
    m=VisualRNN(kind='ei')
    optimizer=torch.optim.Adam(m.parameters(),lr=.01)
    for _ in range(3):
        loss=m(torch.rand(4,1,32,32),torch.rand(4,1,32,32)).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        w=m.weights()
        assert torch.all(w[:,m.signs>0]>=0)
        assert torch.all(w[:,m.signs<0]<=0)

def test_analysis_positive_negative_controls():
    rng=np.random.default_rng(2)
    y=np.repeat(np.arange(5),100)
    center=np.arange(5)[None,:]
    a=np.exp(-((y[:,None]-center)**2)/.6)+rng.normal(0,.03,(500,5))
    b=np.exp(-((y[:,None]-center)**2)/.6)+rng.normal(0,.03,(500,5))
    assert decode(a,y,b,y)>.95
    t=tuning(a,y,b,y)
    assert t['interior_peaked_fraction']==.6
    noise=rng.normal(size=(500,64)); noise2=rng.normal(size=(500,64))
    assert decode(noise,y,noise2,y)<.3
    assert tuning(noise,y,noise2,y)['interior_peaked_fraction']<.05

def test_normalized_eval_independent_of_batch_and_sample_reset():
    torch.manual_seed(4)
    m=VisualRNN(normalize=True)
    a=torch.rand(8,1,32,32);b=torch.rand_like(a)
    m(a,b);m.eval()
    pred=m(a,b)
    single=torch.cat([m(a[i:i+1],b[i:i+1]) for i in range(len(a))])
    assert torch.allclose(pred,single,atol=1e-5)
    assert torch.equal(m(a,b,reset=True),m(a.flip(0),b,reset=True))

def test_matched_initial_effective_weights_and_cached_forward():
    torch.manual_seed(7);a=VisualRNN(kind='rnn',normalize=True,matched_init=True)
    torch.manual_seed(7);b=VisualRNN(kind='ei',normalize=True,matched_init=True)
    assert torch.equal(a.weights(),b.weights())
    assert torch.equal(a.input.weight,b.input.weight)
    assert torch.equal(a.readout.weight,b.readout.weight)
    a.eval();b.eval()
    x=torch.rand(4,1,32,32);z=torch.rand_like(x)
    assert torch.equal(a(x,z),b(x,z))
    assert torch.equal(a(x,z),a(a.encode(x),a.encode(z),encoded=True))

def test_ring_bank_preserves_components_after_rejecting_raster_fragmentation():
    from crowbrain.challenge import bank
    from scipy.ndimage import label
    # This exact held-out stream previously fragmented one six-ring array.
    images=bank(230001,True,n=400).numpy()
    for index,n in enumerate(range(2,7)):
        for im in images[index,:,0]:
            assert label(im>.499,structure=np.ones((3,3)))[1]==n
