"""UC-R02 development-round analysis: read-only aggregation of r02 result.json files.

Part A: rescue verdict for S0/S1 (any (lr,steps) config with mean id BA >= 99%).
Part B: hard-distractor re-screen; primary contrast F1-F0 on id hold BA,
informative only if F0 is off ceiling. No confirmatory statistics.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/unified_cognition/r02'
SEEDS = [270201, 270202, 270203, 270204, 270205]


def load(task):
    runs = {}
    for p in (BASE / task).glob('*/result.json'):
        r = json.loads(p.read_text())
        runs[(r['model'], r['seed'], r['lr'], r['steps'])] = r
    return runs


def agg(runs, model, lr, steps, cond, key):
    vals = [100.0 * runs[(model, s, lr, steps)][cond][key] for s in SEEDS
            if (model, s, lr, steps) in runs
            and runs[(model, s, lr, steps)][cond].get(key) is not None]
    if not vals:
        return None
    return {'mean': round(float(np.mean(vals)), 2), 'sd': round(float(np.std(vals, ddof=1)), 2),
            'min': round(float(min(vals)), 2), 'max': round(float(max(vals)), 2), 'n': len(vals)}


def paired(runs, m_a, m_b, cond, key, lr, steps):
    d = [100.0 * runs[(m_a, s, lr, steps)][cond][key] - 100.0 * runs[(m_b, s, lr, steps)][cond][key]
         for s in SEEDS if (m_a, s, lr, steps) in runs and (m_b, s, lr, steps) in runs]
    if not d:
        return None
    return {'mean': round(float(np.mean(d)), 2), 'sd': round(float(np.std(d, ddof=1)), 2),
            'per_seed': [round(x, 2) for x in d], 'n': len(d)}


def part_a():
    runs = load('standard')
    table = {}
    verdict = {}
    for m in ['F0', 'G', 'S0', 'S1']:
        for lr in [0.003, 0.01]:
            for st in [2000, 4000]:
                a = agg(runs, m, lr, st, 'id', 'accuracy')
                v = agg(runs, m, lr, st, 'val', 'accuracy')
                if a:
                    table[f'{m}_lr{lr:g}_st{st}'] = {'id': a, 'val': v,
                                                     'seconds_mean': round(float(np.mean(
                                                         [runs[(m, s, lr, st)]['seconds'] for s in SEEDS
                                                          if (m, s, lr, st) in runs])), 1)}
    for m in ['S0', 'S1']:
        best = None
        for lr in [0.003, 0.01]:
            for st in [2000, 4000]:
                a = agg(runs, m, lr, st, 'id', 'accuracy')
                if a and (best is None or a['mean'] > best[1]):
                    best = ((lr, st), a['mean'])
        if best:
            verdict[m] = {'best_config': {'lr': best[0][0], 'steps': best[0][1]},
                          'best_id_mean': best[1],
                          'ceiling_99_reached': bool(best[1] >= 99.0),
                          'partial_95_99': bool(95.0 <= best[1] < 99.0)}
    return {'config_table': table, 'rescue_verdict': verdict}


def part_b():
    runs = load('hard')
    lr, st = 0.003, 4000
    per_model = {}
    for m in ['F0', 'F1', 'S0', 'S1', 'G']:
        e = {}
        for cond in ['id', 'ood', 'delay8']:
            e[cond] = {k: agg(runs, m, lr, st, cond, k) for k in
                       ['accuracy', 'hold', 'update', 'reset_accuracy']}
        rows = [runs[(m, s, lr, st)] for s in SEEDS if (m, s, lr, st) in runs]
        if rows and 'gate_mean_update' in rows[0]['id']:
            e['gates'] = {'update': round(float(np.mean([r['id']['gate_mean_update'] for r in rows])), 3),
                          'hold': round(float(np.mean([r['id']['gate_mean_hold'] for r in rows])), 3)}
        per_model[m] = e
    f0_id = per_model['F0']['id']['accuracy']['mean'] if per_model.get('F0') else None
    f0_hold = per_model['F0']['id']['hold']['mean'] if per_model.get('F0') else None
    contrasts = {
        'F1-F0_id_hold': paired(runs, 'F1', 'F0', 'id', 'hold', lr, st),
        'F1-F0_id_accuracy': paired(runs, 'F1', 'F0', 'id', 'accuracy', lr, st),
        'F1-F0_ood_accuracy': paired(runs, 'F1', 'F0', 'ood', 'accuracy', lr, st),
        'S1-S0_id_hold': paired(runs, 'S1', 'S0', 'id', 'hold', lr, st),
        'S1-S0_id_accuracy': paired(runs, 'S1', 'S0', 'id', 'accuracy', lr, st),
        'guard_F1-F0_id_update': paired(runs, 'F1', 'F0', 'id', 'update', lr, st),
        'guard_S1-S0_id_update': paired(runs, 'S1', 'S0', 'id', 'update', lr, st),
    }
    return {'per_model': per_model,
            'f0_off_ceiling': {'id_accuracy': f0_id, 'id_hold': f0_hold,
                               'off_ceiling': bool(f0_id is not None and f0_id < 99.0)},
            'contrasts_pp': contrasts}


def main():
    a = part_a(); b = part_b()
    out = {'round': 'UC-R02', 'nature': 'development_only_unconfirmed',
           'part_a_rescue': a, 'part_b_hardscreen': b,
           'note': 'Exploratory development estimates; no confirmatory claims; no alpha spent.'}
    (BASE / 'summary.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print('=== A: rescue verdict ===')
    print(json.dumps(a['rescue_verdict'], indent=2))
    for k, v in a['config_table'].items():
        print(f"{k}: id={v['id']['mean']}±{v['id']['sd']}")
    print('=== B: F0 off ceiling? ===')
    print(json.dumps(b['f0_off_ceiling'], indent=2))
    for m, e in b['per_model'].items():
        print(f"{m}: id={e['id']['accuracy']} hold={e['id']['hold']} update={e['id']['update']}")
    print('=== B: contrasts (pp) ===')
    print(json.dumps(b['contrasts_pp'], indent=2))


if __name__ == '__main__':
    main()
