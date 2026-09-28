"""Lossless experiment storage: exact dot geometry and binary prediction deltas."""
import io,json,zlib
from pathlib import Path
import numpy as np
from crowbrain.task import draw_dots as original_draw,COUNTS

def draw_recorded(n,rng,protocol='mixed',size=32):
    if protocol in ('mixed','standard','area'):
        target=rng.uniform(45,75);width=int(rng.integers(21,27))
    elif protocol=='shift':target=rng.uniform(80,100);width=int(rng.integers(25,30))
    else:raise ValueError(protocol)
    lo=(size-width)//2;hi=lo+width-1;initial_r=np.sqrt(target/(np.pi*n));anchors=np.array([[0.,0.],[1.,1.]])
    if rng.random()<.5:anchors[:,1]=1-anchors[:,1]
    points=list(anchors)
    for _ in range(n-2):
        for attempt in range(1000):
            p=rng.uniform(.05,.95,2)
            if all(np.linalg.norm(p-q)*(width-1-2*initial_r)>2*initial_r+1.5 for q in points):points.append(p);break
        else:return draw_recorded(n,rng,protocol,size)
    points=np.array(points);yy,xx=np.mgrid[:size,:size]
    def render(radius):
        centers=lo+radius+points*(hi-lo-2*radius);dist=np.sqrt((yy[None]-centers[:,0,None,None])**2+(xx[None]-centers[:,1,None,None])**2)
        return np.clip(radius+.5-dist,0,1).max(0)
    left,right=initial_r*.8,initial_r*1.2
    for _ in range(24):
        middle=(left+right)/2
        if render(middle).sum()<target:left=middle
        else:right=middle
    radius=(left+right)/2;image=render(radius).astype(np.float32);centers=lo+radius+points*(hi-lo-2*radius)
    geometry=np.zeros(13,np.float64);geometry[0]=radius;geometry[1:1+2*n]=centers.ravel()
    return image,geometry

def decode_geometry(geometry):
    assert geometry.ndim==3 and geometry.shape[0]==5 and geometry.shape[2]==13
    yy,xx=np.mgrid[:32,:32];images=[]
    for ci,n in enumerate(COUNTS):
        for g in geometry[ci]:
            radius=g[0];centers=g[1:1+2*n].reshape(n,2)
            dist=np.sqrt((yy[None]-centers[:,0,None,None])**2+(xx[None]-centers[:,1,None,None])**2)
            images.append(np.clip(radius+.5-dist,0,1).max(0).astype(np.float32))
    return np.stack(images).reshape(5,geometry.shape[1],1,32,32)

def load_bank(path):
    with np.load(path) as f:d={k:f[k] for k in f.files}
    if 'geometry' in d:d['images']=decode_geometry(d.pop('geometry'))
    return d

def pack_binary(a):
    a=np.asarray(a,dtype=np.uint8);assert ((a==0)|(a==1)).all();delta=np.concatenate([a[:1],a[1:]^a[:-1]],axis=0)
    return np.packbits(delta,axis=-1,bitorder='little')

def unpack_binary(packed,n):
    delta=np.unpackbits(packed,axis=-1,count=n,bitorder='little');return np.bitwise_xor.accumulate(delta,axis=0).astype(np.int8)

def save_lzma_npz(path,**arrays):
    import zipfile
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_LZMA) as z:
        for k,v in arrays.items():
            buf=io.BytesIO();np.lib.format.write_array(buf,np.asarray(v),allow_pickle=False);z.writestr(k+'.npy',buf.getvalue())

def save_predictions(path,**arrays):
    payload={}
    for key,arr in arrays.items():
        a=np.asarray(arr,np.uint8);assert ((a==0)|(a==1)).all();flat=a.reshape(a.shape[0],-1);base=flat[0];vary=np.any(flat!=base,axis=0);values=flat[:,vary]
        packed=pack_binary(a);bp=np.packbits(base,bitorder='little');vp=np.packbits(vary,bitorder='little');vv=np.packbits(values,axis=0,bitorder='little')
        masked_size=sum(len(zlib.compress(x.tobytes(),6)) for x in [bp,vp,vv])+256
        if masked_size<len(zlib.compress(packed.tobytes(),6)):
            payload.update({key+'_base':bp,key+'_vary':vp,key+'_values':vv})
        else:payload[key+'_packed']=packed
        payload[key+'_shape']=np.array(a.shape,np.int64)
    np.savez_compressed(path,**payload)

def load_predictions(path):
    out={}
    with np.load(path) as f:
        for field in f.files:
            if not field.endswith('_shape'):continue
            key=field[:-6];shape=tuple(f[field]);n=int(np.prod(shape[1:]))
            if key+'_packed' in f:out[key]=unpack_binary(f[key+'_packed'],shape[-1]);continue
            base=np.unpackbits(f[key+'_base'],count=n,bitorder='little');vary=np.unpackbits(f[key+'_vary'],count=n,bitorder='little').astype(bool)
            a=np.broadcast_to(base,(shape[0],n)).copy();a[:,vary]=np.unpackbits(f[key+'_values'],axis=0,count=shape[0],bitorder='little');out[key]=a.reshape(shape).astype(np.int8)
    return out

def save_gzip_json(path,obj):
    import gzip
    with gzip.open(path,'wt') as f:json.dump(obj,f,separators=(',',':'))

def load_gzip_json(path):
    import gzip
    with gzip.open(path,'rt') as f:return json.load(f)
