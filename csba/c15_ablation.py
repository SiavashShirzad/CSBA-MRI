import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
import c05_calibrate as cal, subprocess
LAB = {'harm': 'Primary (N4 + acquisition augmentation)', 'main': 'Without N4 / acquisition augmentation', 'nouw': 'No precision weighting', 'global': 'Global-age supervision only',
       'uniform': 'No level normalization', 'noanchor': 'No marrow/CSF intensity anchor', 'marrowmask': 'Marrow removed from input'}
rows = []
for v in ['harm', 'main', 'nouw', 'global', 'uniform', 'noanchor', 'marrowmask']:
    src = 'main' if v == 'nouw' else v
    if not all((ROOT/'preds'/f'{src}_f{k}.csv').exists() for k in range(5)): continue
    if v not in ('main', 'nouw', 'harm'): cal.calibrate(v)
    G = load_gaps(src); P = G[G['role'] != 'retest'].copy()
    pre = 'au_' if v in ('nouw', 'global') else ''
    gs, rs = f'g_{pre}spine', f'raw_{pre}spine'
    T = P[P['role'] == 'test']; D = P[P['role'] == 'dev']
    r = dict(variant=v, label=LAB[v], test_mae=np.mean(np.abs(T[rs]-T['age'])), test_r=pearsonr(T[rs], T['age'])[0],
             dev_mae=np.mean(np.abs(D[rs]-D['age'])), test_cal_mae=np.mean(np.abs(T[gs])))
    lo, hi = boot_ci(lambda i: np.mean(np.abs(T[rs].values[i]-T['age'].values[i])), len(T)); r.update(test_mae_lo=lo, test_mae_hi=hi)
    E = P[(P['group'] == 'HC') & (P['center'] == 'Center1')]; r['ext_c1_mae'] = np.mean(np.abs(E[rs]-E['age']))
    HC = P['group'] == 'HC'
    for nm, (c, k, rhs) in {'C1 abnormal': (P[HC & (P['dataset'] == 'JHU_clinical')], P[P['group'] == 'JHU_abnormal'], 'grp + age'),
                            'DCM': (P[HC & P['center'].isin(['Center1', 'Center2'])], P[P['group'] == 'DCM'], 'grp + age + C(center)'),
                            'Duke abnormal': (P[HC & (P['dataset'] == 'Duke')], P[P['group'] == 'Duke_abnormal'], 'grp + age'),
                            'SG compression': (P[HC & (P['dataset'] == 'SpineGeneric')], P[P['group'] == 'SG_compression'], 'grp + age + C(site)')}.items():
        e = group_effect(pd.concat([c.assign(grp=0), k.assign(grp=1)]), gs, rhs)
        r[f'{nm} beta'] = e['beta']; r[f'{nm} lo'] = e['lo']; r[f'{nm} hi'] = e['hi']; r[f'{nm} p'] = e['p']
    cc = [f'g_{pre}{x}' for x in ['cord', 'csf', 'disc']] + ([f'g_{pre}bone'] if f'g_{pre}bone' in T else [])
    cm = T[cc].corr().values; r['between_compartment_r'] = float(np.median(cm[np.triu_indices(len(cc), 1)]))
    rows.append(r)
A = pd.DataFrame(rows); A.to_csv(TAB/'ablation_summary.csv', index=False)
print(A.round(3).T.to_string())
lo = []
for d in ['Duke', 'SpineGeneric']:
    f = ROOT/'preds'/(f'loso_{d}_f0.csv' if PRIMARY == 'main' else f'{PRIMARY}_loso_{d}_f0.csv')
    if not f.exists(): continue
    F = pd.read_csv(f); S = pd.read_csv(ROOT/'splits.csv')
    X = S[(S['dataset'] == d) & S['role'].isin(['dev', 'test'])].merge(F, on='img_id')
    e = X['a_spine'] - X['age']
    lo.append(dict(held_out=d, n=len(X), mae=np.mean(np.abs(e)), r=pearsonr(X['a_spine'], X['age'])[0], bias=e.mean(), slope=np.polyfit(X['age'], e, 1)[0]))
    G = load_gaps(); m = G[(G['dataset'] == d) & G['role'].isin(['dev', 'test'])]
    lo[-1].update(cv_mae_same_participants=np.mean(np.abs(m['raw_spine'] - m['age'])))
if lo: pd.DataFrame(lo).to_csv(TAB/'loso.csv', index=False); print(pd.DataFrame(lo).round(3).to_string())
