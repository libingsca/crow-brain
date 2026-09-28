"""UC-R02: split-family rescue diagnosis + off-ceiling interference re-screen.

Development round, no confirmatory claims. Two parts (see R02_PROTOCOL.md):

Part A (rescue): can S0/S1 reach the ID ceiling of fused models under symmetric
or larger budgets? Grid: S0,S1 x lr {0.003, 0.01} x steps {2000, 4000}; F0/G run
as budget-matched references. Standard T1 task (reuses r01.sample unchanged).

Part B (off-ceiling re-screen): hard-distractor T1 variant where the lure pair
shares EXACTLY ONE relation (color or shape, 50/50) with the effective pair,
making binding harder; run F0/F1/S0/S1/G at fixed lr 0.003, 4000 steps. The
clean H02 contrast is F1-F0 on hold trials, informative only when F0 is off
ceiling; F0 staying at ceiling is itself a registered (negative-development)
outcome that redirects to stronger load levers in a future round.

Reuses the frozen r01.py data layer and models by import; r01.py is NOT
modified. All outputs go to artifacts/unified_cognition/r02/<task>/...
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

import r01
from r01 import (BASE as R01_BASE, D, CONTENT_DIM, INSTR_OFF, Q_OFF, PAIRS, REL, HELD,
                 POOLS, partition, labels, encode, evaluate, oracle_eval,
                 Net, model_specs, nparams, dump, digest, _rss_mb)

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/unified_cognition/r02'
R01_SHA = digest(ROOT / 'scripts/unified_cognition/r01.py')


def sample_hard(rng, n, split='train', ood=False, delay=4):
    """Hard-distractor T1: lure shares exactly one relation with the effective pair.

    Effective pair drawn from the (ood, c, s) stratum; lure from the ID stratum
    (False, lc, ls) where (lc, ls) keeps exactly one of (c, s) and flips the
    other; which one is kept alternates in a 32-stratum exact balance
    (u x q x c x s x share). Lures are always ID pairs (familiar objects)."""
    assert n % 32 == 0
    out = []
    for u in (0, 1):
        for q in (0, 1):
            for c in (0, 1):
                for s in (0, 1):
                    for share_c in (0, 1):
                        chunks = []; left = n // 32
                        while left:
                            m = max(32, left * 12 if split != 'train' else left * 2)
                            eff = rng.choice(POOLS[ood, c, s], m)
                            lc = c if share_c else 1 - c
                            ls = s if not share_c else 1 - s
                            lure = rng.choice(POOLS[False, lc, ls], m)
                            a, b = (eff, lure) if u == 0 else (lure, eff)
                            rec = np.column_stack([a, b, np.full(m, u), np.full(m, q),
                                                   np.full(m, delay),
                                                   rng.integers(4, size=m), rng.integers(4, size=m)])
                            bucket = partition(rec)
                            valid = bucket < 8 if split == 'train' else bucket == (8 if split == 'val' else 9)
                            v = rec[valid][:left]; chunks.append(v); left -= len(v)
                        out.append(np.concatenate(chunks))
    rec = np.concatenate(out).astype(np.int16); rng.shuffle(rec)
    return rec


SAMPLERS = {'standard': r01.sample, 'hard': sample_hard}


def run(args):
    sampler = SAMPLERS[args.task]
    spec = model_specs()
    width = spec[args.model] if args.model != 'Neg' else 0
    torch.manual_seed(args.seed)
    model = Net(args.model, width) if args.model != 'Neg' else Net('Neg')
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = BASE / args.task / f'{args.model}_s{args.seed}_lr{args.lr:g}_st{args.steps}'
    if (out / 'result.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    records = []; curve = []; peak_mem = 0.0
    start = time.monotonic()
    for step in range(args.steps):
        delay = int(rng.choice([2, 4, 6]))
        rec = sampler(rng, 64, delay=delay); records.append(rec)
        x, y = encode(rec)
        optim.zero_grad(); loss = nn.functional.cross_entropy(model(x), y); loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
        peak_mem = max(peak_mem, _rss_mb())
        if step % 400 == 0 or step == args.steps - 1:
            curve.append({'step': step + 1, 'loss': loss.item(), 'grad_norm': float(norm.detach())})
    torch.save({'state': model.state_dict(), 'kind': args.model, 'width': width}, out / 'model.pt')
    np.savez_compressed(out / 'training.npz', records=np.stack(records))
    result = {'round': 'UC-R02', 'task': args.task, 'model': args.model, 'seed': args.seed,
              'lr': args.lr, 'steps': args.steps, 'nparams': nparams(model), 'width': width,
              'curve': curve, 'peak_rss_mb': peak_mem,
              'source_r01_sha256': R01_SHA, 'source_sha256': digest(__file__)}
    for name, split, ood, delay in [('val', 'val', False, 4), ('id', 'test', False, 4),
                                    ('ood', 'test', True, 4), ('delay8', 'test', False, 8)]:
        rr = sampler(np.random.default_rng(np.random.SeedSequence(
            [args.seed, {'val': 2, 'id': 3, 'ood': 4, 'delay8': 5}[name]])),
            args.eval_n, split, ood, delay)
        result[name] = evaluate(model, rr, out / f'{name}.npz')
        result['oracle_' + name] = oracle_eval(rr)
    result['seconds'] = time.monotonic() - start
    dump(out / 'result.json', result)
    print(json.dumps({'task': args.task, 'model': args.model, 'seed': args.seed, 'lr': args.lr,
                      'steps': args.steps, 'seconds': result['seconds'],
                      'id_acc': result['id']['accuracy'], 'ood_acc': result['ood']['accuracy']},
                     ensure_ascii=False), flush=True)


def audit_hard():
    """Quality gates for the hard generator: truth cross-check, balance,
    partition disjointness, shortcut probes (pair_a expected 0.75 by design)."""
    rng = np.random.default_rng(270200)
    checks = []
    for split in ['train', 'val', 'test']:
        for ood in [False, True]:
            rec = sample_hard(rng, 4096, split, ood)
            x, y = encode(rec)
            obs = x.numpy(); use = 2 * (obs[:, 1, INSTR_OFF + 1] > 0).astype(int)
            rows = np.arange(len(rec)); q = obs[:, -1, Q_OFF:Q_OFF + 2].argmax(1)
            c0 = obs[rows, use, 0:4].argmax(1); c1 = obs[rows, use, 12:16].argmax(1)
            s0 = obs[rows, use, 4:8].argmax(1); s1 = obs[rows, use, 16:20].argmax(1)
            truth = np.where(q == 0, c0 == c1, s0 == s1).astype(int)
            assert np.array_equal(truth, y.numpy()), f'truth mismatch hard {split}/{ood}'
            assert len(np.unique(partition(rec))) == (8 if split == 'train' else 1)
            for u in (0, 1):
                for qv in (0, 1):
                    ix = (rec[:, 2] == u) & (rec[:, 3] == qv)
                    assert abs(y[ix].float().mean() - .5) < 1e-9, 'balance violation (hard)'
            # lure shares exactly one relation with the effective pair (structural check)
            eff = np.where(rec[:, 2] == 1, rec[:, 1], rec[:, 0])
            lure = np.where(rec[:, 2] == 1, rec[:, 0], rec[:, 1])
            shared = (REL[eff, 0] == REL[lure, 0]).astype(int) + (REL[eff, 1] == REL[lure, 1]).astype(int)
            assert np.all(shared == 1), 'lure must share exactly one relation'
            checks.append({'split': split, 'ood': ood, 'n': len(rec),
                           'oracle_accuracy': float((truth == y.numpy()).mean()),
                           'lure_shared_relations_all_exactly_one': True})
    sc = sample_hard(rng, 4096, 'test'); y = labels(sc)
    eff = np.where(sc[:, 2] == 1, sc[:, 1], sc[:, 0])
    shortcuts = {'question_only': float((np.where(sc[:, 3] == 0, 0, 1) == y).mean()),
                 'update_only': float((sc[:, 2] == y).mean()),
                 'effective_pair_only': float((REL[eff, sc[:, 3]] == y).mean()),
                 'pair_a_only_expected_0.75': float((REL[sc[:, 0], sc[:, 3]] == y).mean())}
    result = {'round': 'UC-R02', 'task': 'hard', 'checks': checks,
              'shortcut_probes': shortcuts, 'source_r01_sha256': R01_SHA,
              'source_sha256': digest(__file__)}
    dump(BASE / 'audit_hard.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['audit', 'run'])
    p.add_argument('--task', default='standard', choices=['standard', 'hard'])
    p.add_argument('--model', default='S0')
    p.add_argument('--seed', type=int, default=270201)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--eval-n', type=int, default=4096)
    args = p.parse_args()
    audit_hard() if args.action == 'audit' else run(args)
