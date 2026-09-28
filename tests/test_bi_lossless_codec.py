import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from crowbrain.task import draw_dots
from branch_bi_codec import draw_recorded,decode_geometry,pack_binary,unpack_binary,save_predictions,load_predictions

def test_geometry_matches_original_stream_and_pixels():
 for seed in [1140100,1240100]:
  a=np.random.default_rng(seed);b=np.random.default_rng(seed);ims=[];geo=[]
  for c in range(2,7):
   for _ in range(20):
    im,_=draw_dots(c,a);rec,g=draw_recorded(c,b);np.testing.assert_array_equal(im,rec);ims.append(im);geo.append(g)
  np.testing.assert_array_equal(decode_geometry(np.array(geo).reshape(5,20,13)),np.array(ims).reshape(5,20,1,32,32));assert a.bit_generator.state==b.bit_generator.state

def test_binary_roundtrip_non_byte_aligned_and_file(tmp_path):
 rng=np.random.default_rng(0);a=rng.integers(0,2,(7,11,2003),dtype=np.int8);np.testing.assert_array_equal(unpack_binary(pack_binary(a),2003),a)
 p=tmp_path/'p.npz';save_predictions(p,predictions=a,original=a[0],reset=a[:,:2]);out=load_predictions(p)
 for k,v in dict(predictions=a,original=a[0],reset=a[:,:2]).items():np.testing.assert_array_equal(out[k],v)

def test_shared_predictions_mask_and_lzma(tmp_path):
 from branch_bi_codec import save_lzma_npz
 rng=np.random.default_rng(2);base=rng.integers(0,2,(11,2000),dtype=np.int8);a=np.tile(base,(96,1,1));a[:,2,:30]=rng.integers(0,2,(96,30))
 p=tmp_path/'shared.npz';save_predictions(p,predictions=a,original=base,reset=a[:,:2]);out=load_predictions(p)
 for k,v in dict(predictions=a,original=base,reset=a[:,:2]).items():np.testing.assert_array_equal(out[k],v)
 p=tmp_path/'lzma.npz';v=rng.permutation(2000).astype(np.int16);save_lzma_npz(p,indices=v)
 with np.load(p) as f:np.testing.assert_array_equal(f['indices'],v)
