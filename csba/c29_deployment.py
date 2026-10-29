import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
from sklearn.metrics import roc_auc_score

# Deployment-oriented analyses: automated processing success, discrimination of patient groups, use of model uncertainty
G = load_gaps(); P = G[G['role'] != 'retest']
HC = P['group'] == 'HC'
CON = {  # as in c07_contrasts
    'Center 1 abnormal vs controls': (P[HC & (P['dataset'] == 'JHU_clinical')], P[P['group'] == 'JHU_abnormal'], 'grp + age'),
    'DCM vs local controls (Centers 1-2)': (P[HC & P['center'].isin(['Center1', 'Center2'])], P[P['group'] == 'DCM'], 'grp + age + C(center)'),
    'Duke abnormal vs Duke controls': (P[HC & (P['dataset'] == 'Duke')], P[P['group'] == 'Duke_abnormal'], 'grp + age'),
    'Spine Generic compression vs controls': (P[HC & (P['dataset'] == 'SpineGeneric')], P[P['group'] == 'SG_compression'], 'grp + age + C(site)'),
}

# 1. automated processing success among images that entered the pipeline
Q = pd.read_csv(ROOT/'cohort_images_qc.csv')
FAIL = ('straightening failed', 'incomplete C2-T1 coverage', 'segmentation missing')
Q['proc_fail'] = Q['exclusion'].fillna('').str.startswith(FAIL)
E = Q[Q['exclusion'].isna() | Q['proc_fail']]
pr = []
for ds, d in list(E.groupby('dataset')) + [('All', E)]:
    ok = d.groupby('ID_s')['proc_fail'].apply(lambda s: (~s).any())
    pr.append(dict(dataset=ds, images=len(d), images_ok=int((~d['proc_fail']).sum()), participants=len(ok), participants_ok=int(ok.sum())))
PR = pd.DataFrame(pr)
PR['images_pct'] = 100*PR['images_ok']/PR['images']; PR['participants_pct'] = 100*PR['participants_ok']/PR['participants']
PR.to_csv(TAB/'processing_success.csv', index=False)
print(PR.round(1).to_string())

# 2. discrimination of each patient group from same-center controls by the whole-spine age gap,
#    unadjusted and age-adjusted (gap minus the value expected from the controls' regression on age and site)
def auc_adj(c, k, rh):
    m = smf.ols(f'g_spine ~ {rh}', c).fit()
    y = np.r_[np.zeros(len(c)), np.ones(len(k))]
    return roc_auc_score(y, np.r_[c['g_spine'] - m.predict(c), k['g_spine'] - m.predict(k)])


au = []
for nm, (c, k, rhs) in CON.items():
    rh = rhs.replace('grp + ', '')
    y = np.r_[np.zeros(len(c)), np.ones(len(k))]
    x = np.r_[c['g_spine'].values, k['g_spine'].values]
    lo, hi = boot_ci(lambda i: roc_auc_score(y[i], x[i]) if 0 < y[i].sum() < len(i) else np.nan, len(y))
    rng = np.random.default_rng(SEED); bs = []
    for _ in range(1000):
        cb, kb = c.sample(len(c), replace=True, random_state=rng), k.sample(len(k), replace=True, random_state=rng)
        try: bs.append(auc_adj(cb, kb, rh))
        except Exception: pass
    al, ah = np.nanpercentile(bs, [2.5, 97.5])
    au.append(dict(contrast=nm, n_ctrl=len(c), n_case=len(k), auc=roc_auc_score(y, x), lo=lo, hi=hi,
                   auc_age_adj=auc_adj(c, k, rh), adj_lo=al, adj_hi=ah))
AU = pd.DataFrame(au); AU.to_csv(TAB/'discrimination.csv', index=False)
print(AU.round(3).to_string())

# 3. model uncertainty: does it flag scans with large error, and is it higher in patients?
H = P[P['group'] == 'HC'].copy()
H['set'] = np.where(H['role'] == 'dev', 'Cross-validated development', np.where(H['role'] == 'test', 'Internal test', 'External controls'))
H['err'] = (H['raw_spine'] - H['age']).abs()
uf = []
for nm, d in H[H['role'] != 'dev'].groupby('set'):
    top = d['sm_spine'] >= d['sm_spine'].quantile(0.9)
    uf.append(dict(set=nm, n=len(d), mae_top10_sigma=d.loc[top, 'err'].mean(), mae_rest=d.loc[~top, 'err'].mean(),
                   auc_sigma_err_gt10=roc_auc_score(d['err'] > 10, d['sm_spine']) if (d['err'] > 10).sum() else np.nan,
                   n_err_gt10=int((d['err'] > 10).sum())))
for nm, (c, k, rhs) in CON.items():
    d = pd.concat([c.assign(grp=0), k.assign(grp=1)], ignore_index=True)
    r = group_effect(d, 'sm_spine', rhs)
    uf.append(dict(set=f'sigma, {nm}', n=r['n'], sigma_diff=r['beta'], lo=r['lo'], hi=r['hi'], p=r['p'],
                   sigma_ctrl=c['sm_spine'].median(), sigma_case=k['sm_spine'].median()))
UF = pd.DataFrame(uf); UF.to_csv(TAB/'uncertainty_flagging.csv', index=False)
pd.set_option('display.width', 250); print(UF.round(3).to_string())
