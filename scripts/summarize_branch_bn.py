import json
from pathlib import Path
import numpy as np
from scipy.stats import t,sem
R=Path('artifacts/branch_bn/round1');S=[301,302,305]
def ci(v,confidence=1-.05/3):
    v=np.asarray(v,float);m=float(v.mean());h=float(t.ppf((1+confidence)/2,len(v)-1)*sem(v))
    return {'mean':m,'low':m-h,'high':m+h,'confidence':confidence,'n_independent_banks':len(v)}
def main():
    rows={s:json.loads((R/f'seed{s}'/'result.json').read_text()) for s in range(301,306)}
    assert all(len(r['banks'])==20 for r in rows.values())
    arrays={'aug_noisy':[],'noisy_gain':[],'clean_loss':[],'clean_gate':[]}
    for b in range(20):
        rs=[rows[s]['banks'][b] for s in S]
        arrays['aug_noisy'].append(float(np.mean([np.mean(r['noise']['primary']['aug']) for r in rs])))
        arrays['noisy_gain'].append(float(np.mean([np.mean(r['noise']['primary']['aug'])-np.mean(r['noise']['primary']['clean']) for r in rs])))
        arrays['clean_loss'].append(float(np.mean([r['clean']['clean']-r['clean']['aug'] for r in rs])))
        arrays['clean_gate'].append(float(np.mean([r['clean']['clean'] for r in rs])))
    intervals={k:ci(v) for k,v in arrays.items()}
    controls={name:{k:float(np.mean([np.mean(r['controls'][name][k]) for row in rows.values() for r in row['banks']])) for k in ['reset','noisy_reset','label_shuffle','state_shuffle']} for name in ['clean','aug']}
    gate=all(.47<=v<=.53 for c in controls.values() for v in c.values()) and intervals['clean_gate']['low']>.85 and all(all(row['integrity'].values()) for row in rows.values())
    a,g,c=[intervals[k] for k in ['aug_noisy','noisy_gain','clean_loss']]
    status='quality_gate_failed' if not gate else 'supported' if a['low']>.80 and g['low']>.15 and c['high']<.02 else 'useful_augmentation_excluded' if a['high']<.80 or g['high']<.15 or c['low']>.02 else 'inconclusive'
    permodel={}
    for s,row in rows.items():
        bs=row['banks'];p={'scales':row['scales'],'clean':{name:float(np.mean([b['clean'][name] for b in bs])) for name in bs[0]['clean']},'noise':{},'quantity':{},'C':{}}
        for cond in bs[0]['noise']:p['noise'][cond]={name:float(np.mean([np.mean(b['noise'][cond][name]) for b in bs])) for name in bs[0]['noise'][cond]}
        for pos in ['0','1']:p['quantity'][pos]={k:float(np.mean([np.mean(b['quantity'][pos][k]) for b in bs])) for k in ['all','match','nonmatch','conditional_permutation']}
        fits=json.loads((R/f'seed{s}'/'fits.json').read_text());p['C']={k:v['C'] for k,v in fits.items()};permodel[str(s)]=p
    result={'status':status,'quality_gate':gate,'fixed_sensitive_panel':S,'primary_intervals':intervals,'primary_per_bank':arrays,'controls_all_five_models':controls,'per_model':permodel,'scope':'Conditional on fixed models 301,302,305 and fixed fitted heads; intervals over 20 new independent image banks, not over new model seeds.'}
    (R/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['primary_per_bank','per_model']},indent=2))
if __name__=='__main__':main()
