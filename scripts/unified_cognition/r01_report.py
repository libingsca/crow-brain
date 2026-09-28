"""UC-R01 development-round analysis: read-only aggregation of result.json files.

Selects lr per model by mean val loss (tie -> smaller lr), computes paired-by-seed
exploratory contrasts for H01 (ood) and H02 (hold) plus the interaction, checks
quality gates, and writes summary.json. No confirmatory statistics are produced.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/unified_cognition/r01'
MODELS = ['F0', 'F1', 'S0', 'S1', 'G', 'Neg']
SEEDS = [270101, 270102, 270103, 270104, 270105]
LRS = [0.001, 0.003]


def load_all():
    runs = {}
    for m in MODELS:
        for s in SEEDS:
            for lr in LRS:
                p = BASE / f'{m}_s{s}_lr{lr:g}' / 'result.json'
                if p.exists():
                    runs[(m, s, lr)] = json.loads(p.read_text())
    return runs


def select_lr(runs):
    """Mean val loss across seeds; tie -> smaller lr. Returns {(model,seed): run}."""
    chosen = {}
    table = []
    for m in MODELS:
        cand = []
        for lr in LRS:
            vals = [runs[(m, s, lr)]['val']['loss'] for s in SEEDS if (m, s, lr) in runs]
            if len(vals) == len(SEEDS):
                cand.append((float(np.mean(vals)), lr))
        if not cand:
            continue
        best = min(cand)[0]
        lr = min(lr for v, lr in cand if v <= best + 1e-12)
        table.append({'model': m, 'selected_lr': lr,
                      'val_loss_by_lr': {f'{l:g}': v for v, l in cand}})
        for s in SEEDS:
            chosen[(m, s)] = runs[(m, s, lr)]
    return chosen, table


def metric(run, cond, key):
    v = run[cond].get(key)
    return None if v is None else 100.0 * v


def paired(chosen, m_a, m_b, cond, key):
    d = [metric(chosen[(m_a, s)], cond, key) - metric(chosen[(m_b, s)], cond, key)
         for s in SEEDS if (m_a, s) in chosen and (m_b, s) in chosen]
    return {'mean': float(np.mean(d)), 'sd': float(np.std(d, ddof=1)),
            'per_seed': [round(x, 2) for x in d], 'n': len(d)}


def main():
    runs = load_all()
    n_expected = len(MODELS) * len(SEEDS) * len(LRS)
    chosen, lr_table = select_lr(runs)

    per_model = {}
    for m in MODELS:
        rows = [chosen[(m, s)] for s in SEEDS if (m, s) in chosen]
        if not rows:
            continue
        entry = {'nparams': rows[0]['nparams'], 'width': rows[0]['width'],
                 'seconds_mean': float(np.mean([r['seconds'] for r in rows])),
                 'peak_rss_mb_max': float(max(r['peak_rss_mb'] for r in rows))}
        for cond in ['val', 'id', 'ood', 'delay8']:
            entry[cond] = {}
            for key in ['accuracy', 'hold', 'update', 'reset_accuracy']:
                vals = [metric(r, cond, key) for r in rows if r[cond].get(key) is not None]
                if vals:
                    entry[cond][key] = {'mean': round(float(np.mean(vals)), 2),
                                        'sd': round(float(np.std(vals, ddof=1)), 2),
                                        'min': round(float(min(vals)), 2),
                                        'max': round(float(max(vals)), 2)}
            vals = [r[cond]['loss'] for r in rows]
            entry[cond]['loss'] = {'mean': round(float(np.mean(vals)), 4)}
        if 'gate_mean_update' in rows[0]['id']:
            entry['gates'] = {
                'update_mean': round(float(np.mean([r['id']['gate_mean_update'] for r in rows])), 3),
                'hold_mean': round(float(np.mean([r['id']['gate_mean_hold'] for r in rows])), 3)}
        per_model[m] = entry

    contrasts = {
        'H01_ood': {name: paired(chosen, a, b, 'ood', 'accuracy')
                    for name, a, b in [('S1-F1', 'S1', 'F1'), ('S0-F0', 'S0', 'F0'),
                                       ('S0-G', 'S0', 'G'), ('S1-G', 'S1', 'G')]},
        'H02_hold': {name: paired(chosen, a, b, 'id', 'hold')
                     for name, a, b in [('F1-F0', 'F1', 'F0'), ('S1-S0', 'S1', 'S0')]},
        'H02_update_guard': {name: paired(chosen, a, b, 'id', 'update')
                             for name, a, b in [('F1-F0', 'F1', 'F0'), ('S1-S0', 'S1', 'S0')]},
        'interaction_ood': {'(S1-S0)-(F1-F0)': {
            'mean': None, 'note': 'computed from H01/H02 cells below'}},
    }
    # interaction directly:
    if all((m, s) in chosen for m in ['F0', 'F1', 'S0', 'S1'] for s in SEEDS):
        d = [(metric(chosen[('S1', s)], 'ood', 'accuracy') - metric(chosen[('S0', s)], 'ood', 'accuracy'))
             - (metric(chosen[('F1', s)], 'ood', 'accuracy') - metric(chosen[('F0', s)], 'ood', 'accuracy'))
             for s in SEEDS]
        contrasts['interaction_ood']['(S1-S0)-(F1-F0)'] = {
            'mean': round(float(np.mean(d)), 2), 'sd': round(float(np.std(d, ddof=1)), 2),
            'per_seed': [round(x, 2) for x in d], 'n': len(d)}

    gates_status = None
    if 'Neg' in per_model:
        neg = per_model['Neg']['id']['accuracy']
        gates_status = {'neg_id_accuracy': neg,
                        'neg_in_band': bool(45 <= neg['mean'] <= 55)}
    if 'G' in per_model:
        g = per_model['G']['id']['accuracy']['mean']
        gates_status = gates_status or {}
        gates_status['G_learnability_id_ge_85'] = bool(g >= 85)
        gates_status['G_id_accuracy'] = g
    reset_vals = [per_model[m]['id']['reset_accuracy']['mean'] for m in per_model
                  if 'reset_accuracy' in per_model[m]['id']]
    if reset_vals:
        gates_status = gates_status or {}
        gates_status['reset_band_45_55_all_models'] = bool(all(45 <= v <= 55 for v in reset_vals))

    out = {'round': 'UC-R01', 'nature': 'development_only_unconfirmed',
           'runs_found': len(runs), 'runs_expected': n_expected,
           'lr_selection': lr_table, 'per_model': per_model,
           'contrasts_pp': contrasts, 'quality_gates': gates_status,
           'note': 'All values are exploratory development estimates; no confirmatory claims; no alpha spent.'}
    p = BASE / 'summary.json'
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(out['lr_selection'], indent=2))
    for m, e in per_model.items():
        print(f"{m}: id={e['id']['accuracy']} ood={e['ood']['accuracy']} "
              f"hold={e['id'].get('hold')} update={e['id'].get('update')}")
    print(json.dumps(contrasts['H01_ood'], indent=2))
    print(json.dumps(contrasts['H02_hold'], indent=2))
    print(json.dumps(gates_status, indent=2))


if __name__ == '__main__':
    main()
