import json
from pathlib import Path
import numpy as np
from summarize_branch_b import interval
R=Path('artifacts/branch_b/round2')
def main():
    rows=[json.loads((R/f'seed{s}'/'result.json').read_text()) for s in range(301,306)]
    primary={k:interval([r[k] for r in rows],.995 if k=='rescue15_ba' else .95) for k in ['original5_ba','original15_ba','rescue15_ba']}
    primary['paired_improvement']=interval([r['rescue15_ba']-r['original15_ba'] for r in rows],.995)
    controls={k:interval([r[k] for r in rows]) for k in ['reset_original_ba','reset_rescue_ba']}
    nulls={}
    for k in ['null_train','test_label_shuffle','state_shuffle']:
        permodel=[float(np.mean(r[k])) for r in rows]
        controls[k]=interval(permodel)
        draws=np.mean([r[k] for r in rows],axis=0)
        mc=interval(draws,.99);mc['n_randomizations']=mc.pop('n_models')
        nulls[k]={'n_randomizations':len(draws),'paired_model_average_per_draw':draws.tolist(),'mean':float(draws.mean()),'min':float(draws.min()),'max':float(draws.max()),'percentiles_2_5_97_5':np.quantile(draws,[.025,.5,.975]).tolist(),'mc_mean_99_interval':mc}
    data=json.loads((R/'data_manifest.json').read_text())
    gate=(primary['original5_ba']['low']>.75 and all(.47<=q['mean']<=.53 for q in controls.values()) and all(r['integrity']['state_dict_unchanged'] and r['integrity']['checkpoint_and_head_files_unchanged'] for r in rows) and data['counts_verified'] and data['disjoint_from_all_B1_partitions'])
    a=primary['rescue15_ba'];d=primary['paired_improvement']
    status='quality_gate_failed' if not gate else 'supported' if a['low']>.65 and d['low']>.15 else 'useful_linear_rescue_excluded' if a['high']<.65 or d['high']<.15 else 'inconclusive'
    out={'status':status,'quality_gate':gate,'primary':primary,'controls':controls,'randomization_distributions':nulls,'per_seed':[{'seed':r['seed'],'old':r['original15_ba'],'rescue':r['rescue15_ba'],'null_mean':float(np.mean(r['null_train'])),'fixed_C_real_label_ba':r['fixed_C_real_label_ba'],'complement_pair':r['original_permutation_fixed_C_pair']} for r in rows]}
    (R/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='randomization_distributions'},indent=2))
if __name__=='__main__':main()
