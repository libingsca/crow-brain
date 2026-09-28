"""Fixed, predeclared primary decision and descriptive secondary summaries."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t, sem
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('artifacts/branch_b/round1')
def interval(values,confidence=.95):
    x=np.asarray(values,dtype=float);m=float(x.mean());half=float(t.ppf((1+confidence)/2,len(x)-1)*sem(x))
    return {'mean':m,'low':m-half,'high':m+half,'confidence':confidence,'n_models':len(x)}
def main():
    rows=[json.loads((R/f'seed{s}'/'result.json').read_text()) for s in range(301,306)]
    original=interval([r['delays']['15']['original_ba'] for r in rows])
    rescue=interval([r['delays']['15']['rescue_ba'] for r in rows],.975)
    delta=interval([r['delays']['15']['rescue_ba']-r['delays']['15']['original_ba'] for r in rows],.975)
    standard=interval([r['delays']['5']['original_ba'] for r in rows])
    controls={k:interval([r['controls'][k] for r in rows]) for k in rows[0]['controls']}
    gate=(standard['low']>.75 and all(.47<=controls[k]['mean']<=.53 for k in ['reset_original_ba','permuted_head_ba']) and all(all(r['integrity'].values()) for r in rows))
    status='quality_gate_failed' if not gate else 'supported' if rescue['low']>.65 and delta['low']>.15 else 'useful_linear_rescue_excluded' if rescue['high']<.65 or delta['high']<.15 else 'inconclusive'
    secondary={'cross_time':{k:interval([r['cross_time'][k] for r in rows]) for k in rows[0]['cross_time']},'quantity':{},'noise':{},'delay_curve':{}}
    for delay in ['0','5','10','15','25','40']:
        secondary['delay_curve'][delay]={k:interval([r['delays'][delay][k] for r in rows]) for k in ['original_ba','rescue_ba']}
        secondary['quantity'][delay]={}
        for pos in rows[0]['delays'][delay]['quantity']:
            secondary['quantity'][delay][pos]={}
            for target in rows[0]['delays'][delay]['quantity'][pos]:
                qs=[r['delays'][delay]['quantity'][pos][target] for r in rows]
                summary={k:interval([q[k] for q in qs]) for k in ['test_ba','train_ba','validation_ba','match_ba','nonmatch_ba']}
                if target=='a':summary['conditional_permutation']=interval([np.mean(q['conditional_permutation_ba']) for q in qs])
                secondary['quantity'][delay][pos][target]=summary
    for condition in rows[0]['noise']:
        secondary['noise'][condition]={k:interval([np.mean([z[k] for z in r['noise'][condition]]) for r in rows]) for k in ['original_ba','rescue_ba']}
    result={'status':status,'quality_gate':gate,'standard_original':standard,'long_original':original,'long_rescue':rescue,'paired_improvement':delta,'controls':controls,'secondary':secondary,
            'per_seed':[{'seed':r['seed'],'old15':r['delays']['15']['original_ba'],'rescue15':r['delays']['15']['rescue_ba'],'C':r['delays']['15']['selected_C']} for r in rows]}
    (R/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for row in rows:axes[0].plot([0,1],[row['delays']['15']['original_ba'],row['delays']['15']['rescue_ba']],'o-',alpha=.7,label=str(row['seed']))
    axes[0].axhline(.65,ls='--',color='gray');axes[0].set(xticks=[0,1],xticklabels=['Original head','Refitted linear head'],ylim=(0,1.03),ylabel='Balanced accuracy',title='15-step delay; unchanged cores');axes[0].legend(fontsize=7)
    ds=[0,5,10,15,25,40]
    for key,label in [('original_ba','Original head'),('rescue_ba','Delay-specific head')]:
        axes[1].plot(ds,[secondary['delay_curve'][str(d)][key]['mean'] for d in ds],'o-',label=label)
    axes[1].set(xlabel='Delay steps',ylabel='Balanced accuracy',ylim=(0,1.03),title='Descriptive delay curve');axes[1].legend(fontsize=7)
    for delay in ['5','15']:
        axes[2].plot(range(5),[secondary['quantity'][delay][str(p)]['a']['test_ba']['mean'] for p in range(5)],'o-',label=f'Delay {delay}')
    axes[2].set(xticks=range(5),xticklabels=['Before test','Test 1','Test 2','Test 3','Test 4'],ylabel='Sample-count decoding',ylim=(0,1.03),title='Same-time linear probes');axes[2].legend(fontsize=7)
    fig.tight_layout();fig.savefig(R/'summary.png',dpi=170)
    print(json.dumps({k:v for k,v in result.items() if k!='secondary'},indent=2))
if __name__=='__main__':main()
