import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
G = load_gaps(); P = G[G['role'] != 'retest']
T = pd.read_csv(ROOT/'t1_marrow.csv')
J = P[P['dataset'] == 'JHU_clinical'].merge(T, on='img_id')
J = J[J['group'].isin(['HC', 'JHU_abnormal'])].copy()
J['grp'] = (J['group'] == 'JHU_abnormal').astype(int)
J['t1z'] = (J['t1_vert_disc'] - J['t1_vert_disc'].mean())/J['t1_vert_disc'].std()
rows = []
for y in ['g_spine', 'g_bone', 'g_cord', 'g_csf', 'g_disc']:
    m = ols(J, y, 't1z + age'); lo, hi = m.conf_int().loc['t1z']
    rows.append(dict(analysis=f'{y} ~ T1 marrow/disc ratio (per SD) + age', n=int(m.nobs), beta=m.params['t1z'], lo=lo, hi=hi, p=m.pvalues['t1z']))
for lab, rhs in [('abnormal vs controls, age', 'grp + age'), ('abnormal vs controls, age + T1 marrow ratio', 'grp + age + t1z')]:
    r = group_effect(J, 'g_spine', rhs); r['analysis'] = lab + ' (participants with T1)'; rows.append(r)
m = ols(J, 't1z', 'grp + age'); lo, hi = m.conf_int().loc['grp']
rows.append(dict(analysis='T1 marrow ratio: abnormal vs controls (SD units)', n=int(m.nobs), beta=m.params['grp'], lo=lo, hi=hi, p=m.pvalues['grp']))
H = P[(P['group'] == 'HC')].dropna(subset=['bmi']).copy(); H['bz'] = (H['bmi'] - H['bmi'].mean())/H['bmi'].std()
m = ols(H, 'g_spine', 'bz + age + C(dataset)'); lo, hi = m.conf_int().loc['bz']
rows.append(dict(analysis='healthy: gap ~ BMI (per SD) + age + dataset', n=int(m.nobs), beta=m.params['bz'], lo=lo, hi=hi, p=m.pvalues['bz']))
R = pd.DataFrame(rows); R.to_csv(TAB/'marrow.csv', index=False); print(R.round(3).to_string())
for k in range(1):
    F = pd.read_csv(ROOT/'preds'/f'{PRIMARY}_f0.csv')
    print('sigma (mean voxel) percentiles over spine:', F['sm_spine'].quantile([.05,.5,.95]).round(2).to_dict(), '| ceiling', round(float(np.exp(2.5)),2))
