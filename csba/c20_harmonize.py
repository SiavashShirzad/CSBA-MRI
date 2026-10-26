import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
from neuroCombat import neuroCombat

FEAT = ['g_spine', 'g_cord', 'g_csf', 'g_disc', 'g_bone']
from batches import with_acquisition, batch_of
P = with_acquisition(PRIMARY)


def harmonize(D, level):
    D = D.copy(); D['batch'] = D.apply(batch_of, axis=1, level=level)
    hc = D['group'] == 'HC'
    nb = D[hc].groupby('batch').size()
    ok_b = set(nb[nb >= 3].index)
    D['harmonizable'] = D['batch'].isin(ok_b)
    H = D[hc & D['harmonizable']].copy()
    Y = H[FEAT].values.T
    o = neuroCombat(dat=Y, covars=H[['batch', 'age']].reset_index(drop=True), batch_col='batch', continuous_cols=['age'])
    est = o['estimates']; levels = list(est['batches'])
    # control-referenced application: refit the standardisation used by ComBat and apply EB location/scale to every participant
    Xb = pd.get_dummies(H['batch']).reindex(columns=levels).astype(float).values
    X = np.column_stack([Xb, H['age'].values])
    B = np.linalg.lstsq(X, Y.T, rcond=None)[0]
    nbv = Xb.sum(0); grand = (nbv/nbv.sum()) @ B[:len(levels)]
    var = ((Y.T - X @ B)**2).mean(0)
    A = D[D['harmonizable']].copy()
    stand = grand + np.outer(A['age'].values, B[-1])
    Z = (A[FEAT].values - stand)/np.sqrt(var)
    bi = A['batch'].map({b: i for i, b in enumerate(levels)}).values
    Zs = (Z - est['gamma.star'][bi])/np.sqrt(est['delta.star'][bi])
    A[[f'h_{f[2:]}' for f in FEAT]] = Zs*np.sqrt(var) + stand
    return A, D


rows, summ = [], {}
CON = {'Center 1 abnormal vs controls': (lambda d: (d['group'] == 'HC') & (d['dataset'] == 'JHU_clinical'), lambda d: d['group'] == 'JHU_abnormal', 'grp + age'),
       'DCM vs local controls (Centers 1-2)': (lambda d: (d['group'] == 'HC') & d['center'].isin(['Center1', 'Center2']), lambda d: d['group'] == 'DCM', 'grp + age + C(center)'),
       'DCM vs controls, Center 1': (lambda d: (d['group'] == 'HC') & (d['center'] == 'Center1'), lambda d: (d['group'] == 'DCM') & (d['center'] == 'Center1'), 'grp + age'),
       'DCM vs controls, Center 2': (lambda d: (d['group'] == 'HC') & (d['center'] == 'Center2'), lambda d: (d['group'] == 'DCM') & (d['center'] == 'Center2'), 'grp + age'),
       'Duke abnormal vs Duke controls': (lambda d: (d['group'] == 'HC') & (d['dataset'] == 'Duke'), lambda d: d['group'] == 'Duke_abnormal', 'grp + age'),
       'Spine Generic compression vs controls': (lambda d: (d['group'] == 'HC') & (d['dataset'] == 'SpineGeneric'), lambda d: d['group'] == 'SG_compression', 'grp + age')}
POOL = {'Center 1 abnormal vs all healthy': lambda d: d['group'] == 'JHU_abnormal', 'DCM vs all healthy': lambda d: d['group'] == 'DCM',
        'Duke abnormal vs all healthy': lambda d: d['group'] == 'Duke_abnormal', 'Spine Generic compression vs all healthy': lambda d: d['group'] == 'SG_compression'}


def contrasts(A, y, label):
    for nm, (fc, fk, rhs) in CON.items():
        d = pd.concat([A[fc(A)].assign(grp=0), A[fk(A)].assign(grp=1)])
        r = group_effect(d, y, rhs); r.update(contrast=nm, analysis=label, n_ctrl=int((d.grp == 0).sum()), n_case=int(d.grp.sum())); rows.append(r)
    for nm, fk in POOL.items():
        d = pd.concat([A[A['group'] == 'HC'].assign(grp=0), A[fk(A)].assign(grp=1)])
        r = group_effect(d, y, 'grp + age'); r.update(contrast=nm, analysis=label, n_ctrl=int((d.grp == 0).sum()), n_case=int(d.grp.sum())); rows.append(r)


# 0. residual batch structure in healthy participants before / after
def batch_var(A, y):
    h = A[A['group'] == 'HC']
    m = smf.mixedlm(f'{y} ~ age', h, groups=h['batch']).fit()
    v_b = float(m.cov_re.iloc[0, 0]); v_e = float(m.scale)
    return dict(feature=y, batch_sd=np.sqrt(v_b), resid_sd=np.sqrt(v_e), icc_batch=v_b/(v_b + v_e), n=len(h), n_batches=h['batch'].nunique())


for level in ['vendor_field', 'model']:
    A, D = harmonize(P, level)
    summ[level] = dict(dropped=D.loc[~D['harmonizable']].groupby('group').size().to_dict(), batches=int(A['batch'].nunique()))
    contrasts(A, 'g_spine', f'unharmonized, same participants ({level})')
    contrasts(A, 'h_spine', f'ComBat, batch = {level}')
    bv = pd.DataFrame([dict(**batch_var(A, f), stage='before') for f in ['g_spine']] + [dict(**batch_var(A, f), stage='after') for f in ['h_spine']])
    bv['level'] = level; summ[level]['batch_variance'] = bv.round(3).to_dict('records')
    if level == 'vendor_field':
        A.to_csv(ROOT/'gaps_harmonized.csv', index=False)
        for f in FEAT:
            for nm in ['DCM vs local controls (Centers 1-2)', 'Center 1 abnormal vs controls', 'Duke abnormal vs Duke controls', 'Spine Generic compression vs controls']:
                fc, fk, rhs = CON[nm]
                d = pd.concat([A[fc(A)].assign(grp=0), A[fk(A)].assign(grp=1)])
                r = group_effect(d, 'h_' + f[2:], rhs); r.update(contrast=nm, analysis='ComBat compartments', marker=MARKERS[f[2:]]); rows.append(r)

# 1. Center 1: scanner-matched and acquisition-adjusted DCM contrast (unharmonized gaps)
C1 = P[(P['center'] == 'Center1') & P['group'].isin(['HC', 'DCM'])].copy(); C1['grp'] = (C1['group'] == 'DCM').astype(int)
shared = set(C1.loc[C1.grp == 0, 'scanner']) & set(C1.loc[C1.grp == 1, 'scanner'])
d = C1[C1['scanner'].isin(shared)]
r = group_effect(d, 'g_spine', 'grp + age + C(scanner)'); r.update(contrast='DCM vs controls, Center 1', analysis=f"same scanner models only ({', '.join(sorted(shared))}), + model", n_ctrl=int((d.grp == 0).sum()), n_case=int(d.grp.sum())); rows.append(r)
d = C1.dropna(subset=['fieldT'])
r = group_effect(d, 'g_spine', 'grp + age + C(fieldT) + vox_min + vox_max'); r.update(contrast='DCM vs controls, Center 1', analysis='+ field strength, in-plane resolution, slice thickness', n_ctrl=int((d.grp == 0).sum()), n_case=int(d.grp.sum())); rows.append(r)
C1a = P[(P['dataset'] == 'JHU_clinical') & P['group'].isin(['HC', 'JHU_abnormal'])].copy(); C1a['grp'] = (C1a['group'] == 'JHU_abnormal').astype(int)
r = group_effect(C1a.dropna(subset=['fieldT']), 'g_spine', 'grp + age + C(scanner) + C(fieldT) + vox_min + vox_max')
r.update(contrast='Center 1 abnormal vs controls', analysis='+ scanner model, field strength, resolution'); rows.append(r)

R = pd.DataFrame(rows); R.to_csv(TAB/'harmonization.csv', index=False)
json.dump(summ, open(TAB/'harmonization_summary.json', 'w'), indent=1, default=str)
pd.set_option('display.width', 250)
print(R[['contrast', 'analysis', 'n_ctrl', 'n_case', 'beta', 'lo', 'hi', 'p']].round(3).to_string())
print(json.dumps(summ, indent=1, default=str))
