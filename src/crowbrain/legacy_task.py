"""Visual-only dot arrays and exactly balanced delayed-match trials."""
import numpy as np
import torch

COUNTS = np.arange(1, 6)

def draw_dots(n, rng, protocol='mixed', size=32):
    mode = rng.choice(['standard', 'area']) if protocol == 'mixed' else protocol
    if mode == 'standard':
        radius = rng.uniform(1.5, 2.5)
    elif mode == 'area':
        radius = np.sqrt(rng.uniform(7, 11) / n)
    elif mode == 'shift':
        radius = np.sqrt(rng.uniform(9, 13)) / n**0.65
    else:
        raise ValueError(mode)
    margin = radius + 1
    inset = rng.uniform(0, 3) if mode == 'shift' else 0
    lo, hi = margin + inset, size - 1 - margin - inset
    centers = []
    for _ in range(n):
        for attempt in range(1000):
            p = rng.uniform(lo, hi, 2)
            if all(np.linalg.norm(p-q) > 2*radius + 1 for q in centers):
                centers.append(p)
                break
        else:
            raise RuntimeError('Non-overlap rejection exhausted')
    yy, xx = np.mgrid[:size, :size]
    image = np.zeros((size, size), np.float32)
    for cy, cx in centers:
        # Antialias boundary; no count information supplied besides pixels.
        d = np.sqrt((yy-cy)**2 + (xx-cx)**2)
        image = np.maximum(image, np.clip(radius + .5-d, 0, 1))
    y, x = np.nonzero(image > .5)
    extent = float((x.max()-x.min()+1)*(y.max()-y.min()+1))
    stats = np.array([image.sum(), extent, image.sum()/extent, image.max()], np.float32)
    return image, stats

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
