import warnings
import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score


def interval(values):
    x = np.asarray(values, float)
    mean = float(x.mean())
    half = float(stats.t.ppf(.975, len(x)-1)*stats.sem(x)) if len(x)>1 else float('nan')
    return {'mean': mean, 'low': mean-half, 'high': mean+half, 'n': len(x)}

def decode(train_x, train_y, test_x, test_y):
    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    clf.fit(train_x, train_y)
    return float(balanced_accuracy_score(test_y, clf.predict(test_x)))

def fdr(p, q=.05):
    p = np.nan_to_num(p, nan=1.)
    order = np.argsort(p)
    passes = p[order] <= q*np.arange(1, len(p)+1)/len(p)
    selected = np.zeros(len(p), bool)
    if passes.any():
        selected[order[:np.where(passes)[0][-1]+1]] = True
    return selected

def tuning(select_x, select_y, test_x, test_y):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = stats.f_oneway(*(select_x[select_y==k] for k in range(5)), axis=0).pvalue
    chosen = fdr(p)
    a = np.stack([select_x[select_y==k].mean(0) for k in range(5)])
    b = np.stack([test_x[test_y==k].mean(0) for k in range(5)])
    pref = a.argmax(0)
    peaked = chosen & (pref>0) & (pref<4) & (b.argmax(0)==pref)
    rows = np.arange(len(pref))
    prominence = b[pref,rows] - np.maximum(b[0],b[-1])
    peaked &= prominence > 0
    return {'selective_fraction':float(chosen.mean()), 'interior_peaked_fraction':float(peaked.mean()),
            'selected':chosen.tolist(), 'peaked':peaked.tolist(), 'preferred':pref.tolist(),
            'selection_curves':a.T.tolist(), 'heldout_curves':b.T.tolist()}
