"""UC-R01: screening round for the unified cognition plan (2026-09-27).

Scope: task audit + H01 (content/rule separation) x H02 (selective write gate)
small screening, DEVELOPMENT ONLY. Produces no confirmatory conclusions for
the 12 hypotheses. Source data layer follows h01.py conventions (proven on the
stopped H01 branch) but lives in a new namespace and writes to new paths.

Models:
  F0  fused-state GRU, no selective write gate (candidate always input as-is).
  F1  fused-state GRU + learned WriteGate modulating the candidate write at t=2.
  S0  content/rule split GRU, no selective write gate.
  S1  content/rule split GRU + learned WriteGate on the content stream.
  G   single standard GRU, width tuned so parameter count matches S1 (strong
      baseline). Tests whether split+gate beats a well-sized plain GRU.
  Neg no-memory negative control: linear classifier on the question one-hot only.
  Oracle algorithmic, non-learning: stores the effective pair and computes the
      relation directly. 100% on solvable; independent truth cross-check in audit.

Task T1 (structured input):
  4 colors x 4 shapes = 16 object types; each object also has a position in
  {0,1,2,3} (non-queried nuisance attribute). A trial = pair A (two objects) at
  t=0, hold/update instruction at t=1, candidate pair B at t=2, blank delay,
  relation question (color-same? / shape-same?) at the last step. "hold" -> pair A
  is effective (B is a distractor); "update" -> pair B replaces A. Delays 2/4/6 in
  training, 8 held out for extrapolation. Held-out object types = diagonal
  {c==s} (a latin square), so every color and shape value is still seen in train.
"""
import argparse, hashlib, json, math, time, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/unified_cognition/r01'

D = 28                       # input width: obj_a(12)+obj_b(12)+instr(2)+question(2)
CONTENT_DIM = 24             # dims 0..23: the two objects of the current pair
RULE_DIM = 4                 # dims 24..27: instruction(2)+question(2)
INSTR_OFF = 24               # instruction one-hot at dims 24,25
Q_OFF = 26                   # question one-hot at dims 26,27

# ---- object/pair algebra (color x shape; position added per-sample) ----
PAIRS = np.array([(a // 4, a % 4, b // 4, b % 4) for a in range(16) for b in range(16)], dtype=np.int64)
REL = np.stack([PAIRS[:, 0] == PAIRS[:, 2], PAIRS[:, 1] == PAIRS[:, 3]], 1).astype(np.int64)
# Held-out (OOD) object types = diagonal c==s (a latin square: one per color-row and shape-col).
HELD = (PAIRS[:, 0] == PAIRS[:, 1]) | (PAIRS[:, 2] == PAIRS[:, 3])
POOLS = {(ood, c, s): np.flatnonzero((HELD == ood) & (REL[:, 0] == c) & (REL[:, 1] == s))
         for ood in (False, True) for c in (0, 1) for s in (0, 1)}
assert all(len(v) > 0 for v in POOLS.values())


def dump(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def partition(rec):
    """Injective mixed-radix id over the seven discrete factors; splitmix64 hash.
    Positions are mod 4 (R01 widens the binary p0/p1 used in h01.py)."""
    a, b, u, q, d, p0, p1 = rec.T.astype(np.uint64)
    key = (((((((a * 256 + b) * 2 + u) * 2 + q) * 10 + d) * 4 + p0) * 4 + p1))
    with np.errstate(over='ignore'):
        z = key + np.uint64(0x9e3779b97f4a7c15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        z = z ^ (z >> np.uint64(31))
    return (z % 10).astype(int)


def sample(rng, n, split='train', ood=False, delay=4):
    """Exact balance over update x query x color_rel x shape_rel (16 strata).
    Position p0,p1 drawn uniform in {0,1,2,3}; per-object position = (p+j) mod 4."""
    assert n % 16 == 0
    out = []
    for u in (0, 1):
        for q in (0, 1):
            for c in (0, 1):
                for s in (0, 1):
                    chunks = []; left = n // 16
                    while left:
                        m = max(32, left * 12 if split != 'train' else left * 2)
                        rel = rng.choice(POOLS[ood, c, s], m)
                        irr_c = rng.integers(2, size=m); irr_s = rng.integers(2, size=m)
                        irr = np.empty(m, dtype=int)
                        for ic in (0, 1):
                            for iss in (0, 1):
                                ix = (irr_c == ic) & (irr_s == iss)
                                irr[ix] = rng.choice(POOLS[False, ic, iss], ix.sum())
                        a, b = (rel, irr) if u == 0 else (irr, rel)
                        rec = np.column_stack([a, b, np.full(m, u), np.full(m, q),
                                               np.full(m, delay),
                                               rng.integers(4, size=m), rng.integers(4, size=m)])
                        bucket = partition(rec)
                        valid = bucket < 8 if split == 'train' else bucket == (8 if split == 'val' else 9)
                        v = rec[valid][:left]; chunks.append(v); left -= len(v)
                    out.append(np.concatenate(chunks))
    rec = np.concatenate(out).astype(np.int16); rng.shuffle(rec)
    return rec


def labels(rec):
    a, b, u, q = rec[:, :4].T
    return REL[np.where(u, b, a), q]


def encode(rec):
    delay = int(rec[0, 4]); assert np.all(rec[:, 4] == delay)
    n = len(rec); x = np.zeros((n, delay + 4, D), np.float32)
    for t, col, flip in ((0, 0, 5), (2, 1, 6)):
        p = PAIRS[rec[:, col]]
        for j, off in ((0, 0), (1, 12)):       # obj_a at 0..11, obj_b at 12..23
            x[np.arange(n), t, off + p[:, 2 * j]] = 1            # color (4)
            x[np.arange(n), t, off + 4 + p[:, 2 * j + 1]] = 1    # shape (4)
            x[np.arange(n), t, off + 8 + ((rec[:, flip] + j) % 4)] = 1  # position (4)
    x[np.arange(n), 1, INSTR_OFF + rec[:, 2]] = 1               # instruction
    x[np.arange(n), -1, Q_OFF + rec[:, 3]] = 1                  # question
    return torch.from_numpy(x), torch.from_numpy(labels(rec).copy()).long()


class WriteGate(nn.Module):
    """Learned write gate: g = sigmoid(W . instr_onehot + b), per content feature."""

    def __init__(self, content_dim=CONTENT_DIM, instr_dim=2):
        super().__init__()
        self.lin = nn.Linear(instr_dim, content_dim)

    def forward(self, instr_onehot):  # (B,2) -> (B,content_dim)
        return torch.sigmoid(self.lin(instr_onehot))


class Net(nn.Module):
    def __init__(self, kind='F0', width=64):
        super().__init__(); self.kind = kind; self.width = width
        if kind in ('F0', 'F1'):
            self.core = nn.GRU(D, width, batch_first=True)
            self.head = nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 2))
            if kind == 'F1':
                self.gate = WriteGate(CONTENT_DIM)
        elif kind in ('S0', 'S1'):
            self.cw, self.rw = 48, 16  # total state 64
            self.content = nn.GRU(CONTENT_DIM, self.cw, batch_first=True)
            self.rule = nn.GRU(RULE_DIM, self.rw, batch_first=True)
            self.head = nn.Sequential(nn.Linear(self.cw + self.rw, 32), nn.Tanh(), nn.Linear(32, 2))
            if kind == 'S1':
                self.gate = WriteGate(CONTENT_DIM)
        elif kind == 'G':
            self.core = nn.GRU(D, width, batch_first=True)
            self.head = nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 2))
        elif kind == 'Neg':
            self.head = nn.Sequential(nn.Linear(2, 2))  # question one-hot only
        else:
            raise ValueError(kind)

    def _gated(self, x):
        if not hasattr(self, 'gate') or x.size(1) < 3:
            return x  # no candidate frame (t=2) to gate on short/reset sequences
        instr = x[:, 1, INSTR_OFF:INSTR_OFF + 2]      # (B,2)
        g = self.gate(instr)                          # (B,24)
        t2 = x[:, 2]                                  # (B,D)
        new_t2 = torch.cat([t2[:, :CONTENT_DIM] * g, t2[:, CONTENT_DIM:]], dim=1)
        frames = list(x.unbind(dim=1)); frames[2] = new_t2
        return torch.stack(frames, dim=1)

    def states(self, x):
        h = self._gated(x)
        if self.kind in ('F0', 'F1', 'G'):
            return self.core(h)[0]
        c, _ = self.content(h[:, :, :CONTENT_DIM]); r, _ = self.rule(h[:, :, CONTENT_DIM:])
        return torch.cat([c, r], -1)

    def forward(self, x):
        if self.kind == 'Neg':
            return self.head(x[:, -1, Q_OFF:Q_OFF + 2])
        return self.head(self.states(x)[:, -1])

    def reset(self, x):
        # Clear ALL memory immediately before the question; run question from zero.
        if self.kind == 'Neg':
            return self.forward(x)
        return self.forward(x[:, -1:])


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def model_specs():
    target = nparams(Net('S1', 64))
    w = min(range(16, 193), key=lambda z: abs(nparams(Net('G', z)) - target))
    return {'F0': 64, 'F1': 64, 'S0': 64, 'S1': 64, 'G': w, 'Neg': 0}


@torch.no_grad()
def evaluate(model, rec, save=None):
    if model is not None and getattr(model, 'kind', None) == 'Neg':
        # Neg only sees the question; report chance-level metrics.
        logits, reset = [], []
        for start in range(0, len(rec), 512):
            x, y = encode(rec[start:start + 512])
            logits.append(model(x).numpy()); reset.append(model.reset(x).argmax(1).numpy())
        logits = np.concatenate(logits); pred = logits.argmax(1); y = labels(rec)
        m = {'accuracy': float((pred == y).mean()), 'loss': float(
            nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y)).item()),
            'reset_accuracy': float((np.concatenate(reset) == y).mean())}
        if save: np.savez_compressed(save, records=rec, labels=y, logits=logits)
        return m
    logits, reset, gates = [], [], []
    for start in range(0, len(rec), 512):
        x, y = encode(rec[start:start + 512])
        logits.append(model(x).numpy()); reset.append(model.reset(x).argmax(1).numpy())
        if hasattr(model, 'gate'):
            gates.append(model.gate(x[:, 1, INSTR_OFF:INSTR_OFF + 2]).numpy())
    logits = np.concatenate(logits); pred = logits.argmax(1); y = labels(rec)
    loss = float(nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y)).item())
    m = {'accuracy': float((pred == y).mean()), 'loss': loss,
         'reset_accuracy': float((np.concatenate(reset) == y).mean())}
    for u in (0, 1):
        ix = rec[:, 2] == u
        m['update' if u else 'hold'] = float((pred[ix] == y[ix]).mean())
    if gates:
        g = np.concatenate(gates)
        m['gate_mean_update'] = float(g[rec[:, 2] == 1].mean())
        m['gate_mean_hold'] = float(g[rec[:, 2] == 0].mean())
    payload = dict(records=rec, labels=y, logits=logits, reset=np.concatenate(reset))
    if gates:
        payload['gate'] = np.concatenate(gates)
    if save:
        np.savez_compressed(save, **payload)
    return m


@torch.no_grad()
def oracle_eval(rec):
    """Algorithmic oracle: effective pair = b if update else a; relation direct."""
    a, b, u, q = rec[:, :4].T
    eff = np.where(u, b, a)
    truth = REL[eff, q]
    return {'oracle_accuracy': 1.0, 'n': len(rec), 'mean_label': float(truth.mean())}


def run(args):
    spec = model_specs()
    width = spec[args.model] if args.model != 'Neg' else 0
    torch.manual_seed(args.seed)
    model = Net(args.model, width) if args.model != 'Neg' else Net('Neg')
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = BASE / f'{args.model}_s{args.seed}_lr{args.lr:g}'
    if (out / 'result.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    records = []; curve = []; peak_mem = 0
    start = time.monotonic()
    for step in range(args.steps):
        delay = int(rng.choice([2, 4, 6])); rec = sample(rng, 64, delay=delay); records.append(rec)
        x, y = encode(rec)
        optim.zero_grad(); loss = nn.functional.cross_entropy(model(x), y); loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
        peak_mem = max(peak_mem, _rss_mb())
        if step % 200 == 0 or step == args.steps - 1:
            curve.append({'step': step + 1, 'loss': loss.item(), 'grad_norm': float(norm.detach())})
    torch.save({'state': model.state_dict(), 'kind': args.model, 'width': width}, out / 'model.pt')
    np.savez_compressed(out / 'training.npz', records=np.stack(records))
    result = {'model': args.model, 'seed': args.seed, 'lr': args.lr, 'steps': args.steps,
              'nparams': nparams(model), 'width': width, 'curve': curve,
              'peak_rss_mb': peak_mem, 'source_sha256': digest(__file__)}
    for name, split, ood, delay in [('val', 'val', False, 4), ('id', 'test', False, 4),
                                    ('ood', 'test', True, 4), ('delay8', 'test', False, 8)]:
        rr = sample(np.random.default_rng(np.random.SeedSequence(
            [args.seed, {'val': 2, 'id': 3, 'ood': 4, 'delay8': 5}[name]])),
            args.eval_n, split, ood, delay)
        result[name] = evaluate(model, rr, out / f'{name}.npz')
        result['oracle_' + name] = oracle_eval(rr)
    result['seconds'] = time.monotonic() - start
    dump(out / 'result.json', result)
    print(json.dumps({k: result[k] for k in ['model', 'seed', 'lr', 'seconds', 'id', 'ood']},
                     ensure_ascii=False), flush=True)


def _rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


def audit():
    specs = model_specs()
    info = {k: {'kind': k, 'width': v, 'params': nparams(Net(k, v) if k != 'Neg' else Net('Neg'))}
            for k, v in specs.items()}
    rng = np.random.default_rng(270000)
    checks = []
    for split in ['train', 'val', 'test']:
        for ood in [False, True]:
            rec = sample(rng, 4096, split, ood); x, y = encode(rec)
            # Independent truth from input tensors, using the observed update instruction.
            obs = x.numpy(); use = 2 * (obs[:, 1, INSTR_OFF + 1] > 0).astype(int)  # update flag
            rows = np.arange(len(rec)); q = obs[:, -1, Q_OFF:Q_OFF + 2].argmax(1)
            # Effective pair object types from dims 0..3/4..7 (obj1) and 12..15/16..19 (obj2) of t=use.
            c0 = obs[rows, use, 0:4].argmax(1); c1 = obs[rows, use, 12:16].argmax(1)
            s0 = obs[rows, use, 4:8].argmax(1); s1 = obs[rows, use, 16:20].argmax(1)
            truth = np.where(q == 0, c0 == c1, s0 == s1).astype(int)
            assert np.array_equal(truth, y.numpy()), f'truth mismatch {split}/{ood}'
            assert len(np.unique(partition(rec))) == (8 if split == 'train' else 1)
            for u in (0, 1):
                for qv in (0, 1):
                    ix = (rec[:, 2] == u) & (rec[:, 3] == qv)
                    assert abs(y[ix].float().mean() - .5) < 1e-9, 'balance violation'
            p = BASE / f'audit_{split}_{int(ood)}.npz'; p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, records=rec, truth=truth)
            checks.append({'split': split, 'ood': ood, 'oracle_accuracy': float((truth == y.numpy()).mean()),
                           'n': len(rec)})
    # Uniqueness: no identical seven-factor tuples across train/val/test.
    seen = {}; dup = 0
    for split in ['train', 'val', 'test']:
        rec = sample(rng, 4096, split)
        for row in map(tuple, rec.tolist()):
            if row in seen and seen[row] != split:
                dup += 1
            seen[row] = split
    # Shortcut probes: single features must not predict the answer above chance.
    sc = sample(rng, 4096, 'test'); y = labels(sc)
    shortcuts = {
        'question_only': float((np.where(sc[:, 3] == 0, 0, 1) == y).mean()),
        'update_only': float((sc[:, 2] == y).mean()),
        'pair_a_only': float((REL[sc[:, 0], sc[:, 3]] == y).mean()),
    }
    # Split-state operation correctness: swapping the query leaves content invariant.
    rec = sample(rng, 128, 'test', True); x, y = encode(rec); xx = x.clone()
    xx[:, -1, Q_OFF:Q_OFF + 2] = xx[:, -1, Q_OFF:Q_OFF + 2].flip(1)
    model = Net('S0', 64); a = model.states(x); b = model.states(xx)
    assert torch.equal(a[:, :, :48], b[:, :, :48]), 'content not invariant under query swap'
    assert not torch.equal(a[:, -1, 48:], b[:, -1, 48:]), 'rule did not change'
    result = {'specs': info, 'checks': checks, 'cross_split_duplicates': dup,
              'shortcut_probes': shortcuts, 'split_content_invariance': True,
              'torch': torch.__version__, 'python': platform.python_version(),
              'source_sha256': digest(__file__)}
    dump(BASE / 'audit.json', result); print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['audit', 'run'])
    p.add_argument('--round', default='r01')
    p.add_argument('--model', default='F0')
    p.add_argument('--seed', type=int, default=270101)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--eval-n', type=int, default=4096)
    args = p.parse_args()
    audit() if args.action == 'audit' else run(args)
