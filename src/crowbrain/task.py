"""Visual-only dot arrays and exactly balanced delayed-match trials."""
import numpy as np
import torch

COUNTS = np.arange(2, 7)

def draw_dots(n, rng, protocol='mixed', size=32):
    if protocol in ('mixed', 'standard', 'area'):
        target = rng.uniform(45, 75)
        width = int(rng.integers(21, 27))
    elif protocol == 'shift':
        target = rng.uniform(80, 100)
        width = int(rng.integers(25, 30))
    else:
        raise ValueError(protocol)
    # Foreground mass and measured extent independent of count. Two opposite
    # anchors set extrema; all other centers random inside the support.
    lo = (size - width)//2
    hi = lo + width - 1
    initial_r = np.sqrt(target/(np.pi*n))
    anchors = np.array([[0.,0.],[1.,1.]])
    if rng.random()<.5:
        anchors[:,1] = 1-anchors[:,1]
    points = list(anchors)
    for _ in range(n-2):
        for attempt in range(1000):
            p = rng.uniform(.05,.95,2)
            if all(np.linalg.norm(p-q)*(width-1-2*initial_r) > 2*initial_r+1.5 for q in points):
                points.append(p); break
        else:
            return draw_dots(n,rng,protocol,size)
    points=np.array(points)
    yy,xx=np.mgrid[:size,:size]
    def render(radius):
        centers=lo+radius+points*(hi-lo-2*radius)
        dist=np.sqrt((yy[None]-centers[:,0,None,None])**2+(xx[None]-centers[:,1,None,None])**2)
        return np.clip(radius+.5-dist,0,1).max(0)
    left,right=initial_r*.8,initial_r*1.2
    for _ in range(24):
        middle=(left+right)/2
        if render(middle).sum()<target: left=middle
        else: right=middle
    image=render((left+right)/2).astype(np.float32)
    y,x=np.nonzero(image>.499)
    extent=float((x.max()-x.min()+1)*(y.max()-y.min()+1))
    measured=np.array([image.sum(),extent,image.sum()/extent,image.max()],np.float32)
    return image, measured

class Bank:
    def __init__(self, seed, per_count=1000, protocol='mixed'):
        rng = np.random.default_rng(seed)
        imgs, stats = zip(*(draw_dots(int(n), rng, protocol) for n in COUNTS for _ in range(per_count)))
        self.images = torch.from_numpy(np.stack(imgs).reshape(5, per_count, 1, 32, 32))
        self.stats = np.stack(stats).reshape(5, per_count, -1)
        self.per_count = per_count
        self.seed, self.protocol = seed, protocol

    def batch(self, rng, batch_size=64, pairs=None):
        if pairs is None:
            a = rng.integers(0, 5, batch_size)
            y = rng.integers(0, 2, batch_size)
            b = np.where(y == 1, a, (a+rng.integers(1, 5, batch_size)) % 5)
        else:
            a, b = pairs[:, 0], pairs[:, 1]
            y = (a == b).astype(np.int64)
            batch_size = len(a)
        ia, ib = rng.integers(0, self.per_count, (2, batch_size))
        same = (a == b) & (ia == ib)
        ib[same] = (ib[same] + 1) % self.per_count
        return (self.images[a, ia], self.images[b, ib], torch.tensor(y, dtype=torch.long),
                np.asarray(a), np.asarray(b), np.concatenate([self.stats[a, ia], self.stats[b, ib]], 1))

def balanced_pairs(repeats=40):
    # Off-diagonal weight 1, diagonal weight 4 => equal labels and count marginals.
    return np.array([(a,b) for a in range(5) for b in range(5)
                     for _ in range(repeats * (4 if a == b else 1))])
