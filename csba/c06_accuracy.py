import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *

TAG = sys.argv[1] if len(sys.argv) > 1 else PRIMARY
G = load_gaps(TAG)
G = G[G['role'] != 'retest']
G['cal_spine'] = G['age'] + G['g_spine']
SETS = {'Cross-validated development': G[G['role'] == 'dev'],
        'Internal test': G[G['role'] == 'test'],
        'External Center 1 controls': G[(G['group'] == 'HC') & (G['center'] == 'Center1')],
        'External Center 2 controls': G[(G['group'] == 'HC') & (G['center'] == 'Center2')]}
rows = []
for nm, d in SETS.items():
    for kind, col in [('raw', 'raw_spine'), ('calibrated', 'cal_spine')]:
        r = accuracy(d, col, nm); r['kind'] = kind
        r['gap_mean'] = float((d[col] - d['age']).mean())
        rows.append(r)
    if nm.startswith('External'):
        g = d['g_spine'].values
        loo = np.array([np.delete(g, i).mean() for i in range(len(g))])
        e = g - loo
        lo, hi = boot_ci(lambda i: np.mean(np.abs(e[i])), len(e))
        rows.append(dict(set=nm, kind='calibrated + local offset (leave-one-out)', n=len(e), mae=np.mean(np.abs(e)),
                         mae_lo=lo, mae_hi=hi, r=pearsonr(d['age'] + e, d['age'])[0], bias=e.mean(),
                         slope=np.polyfit(d['age'], e, 1)[0], gap_mean=e.mean(),
                         age_min=d['age'].min(), age_max=d['age'].max()))
A = pd.DataFrame(rows)
A.to_csv(TAB/f'accuracy_{TAG}.csv', index=False)
print(A[['set', 'kind', 'n', 'mae', 'mae_lo', 'mae_hi', 'r', 'bias', 'slope', 'gap_mean']].round(3).to_string())

if TAG == PRIMARY:
    H = G[G['role'].isin(['dev', 'test'])].copy()
    H['age_band'] = pd.cut(H['age'], [17, 30, 45, 60, 75, 90]).astype(str)
    sub = []
    for by in ['dataset', 'protocol', 'sex', 'age_band']:
        for v, d in H.groupby(by):
            if len(d) < 15: continue
            r = accuracy(d, 'cal_spine', f'{by}={v}'); r['raw_mae'] = np.mean(np.abs(d['raw_spine'] - d['age']))
            sub.append(r)
    pd.DataFrame(sub).to_csv(TAB/'accuracy_subgroups.csv', index=False)
    print(pd.DataFrame(sub)[['set', 'n', 'mae', 'mae_lo', 'mae_hi', 'raw_mae', 'r', 'bias']].round(2).to_string())

    T = G[G['role'] == 'test']
    reg = []
    for k in REGIONS + ['bone']:
        p = T[f'raw_{k}']; ok = p.notna()
        reg.append(dict(region=k, r=pearsonr(p[ok], T['age'][ok])[0], mae=np.mean(np.abs(p[ok] - T['age'][ok])),
                        r_lo=boot_ci(lambda i: pearsonr(p[ok].values[i], T['age'][ok].values[i])[0], ok.sum())[0],
                        r_hi=boot_ci(lambda i: pearsonr(p[ok].values[i], T['age'][ok].values[i])[0], ok.sum())[1]))
    comp = ['cord', 'csf', 'disc', 'bone']
    cm = T[[f'g_{c}' for c in comp]].corr().values
    reg.append(dict(region='between-compartment gap r (median)', r=float(np.median(cm[np.triu_indices(4, 1)]))))
    lv = T[[f'g_{v}' for v in VNAMES]].corr().values
    reg.append(dict(region='between-level gap r (median)', r=float(np.median(lv[np.triu_indices(7, 1)]))))
    pd.DataFrame(reg).to_csv(TAB/'regional_performance.csv', index=False)
    print(pd.DataFrame(reg).round(3).to_string())

    D = G[G['role'] == 'dev']
    z_dev = (D['raw_spine'] - D['age']) / D['sm_spine']
    s = float(np.sqrt(np.mean(z_dev**2)))
    unc = []
    for nm, d in SETS.items():
        e = (d['raw_spine'] - d['age']).abs()
        unc.append(dict(set=nm, n=len(d), rho_sigma_abs_error=spearmanr(d['sm_spine'], e)[0],
                        p=spearmanr(d['sm_spine'], e)[1], coverage95_raw_sigma=float(np.mean(e <= 1.96*d['sm_spine'])),
                        coverage95_scaled_sigma=float(np.mean(e <= 1.96*s*d['sm_spine'])), scale=s,
                        sigma_median=float(d['sm_spine'].median())))
    pd.DataFrame(unc).to_csv(TAB/'uncertainty.csv', index=False)
    print(pd.DataFrame(unc).round(3).to_string())
