import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
import c05_calibrate as cal

TAG = sys.argv[1] if len(sys.argv) > 1 else PRIMARY
G = load_gaps(TAG)
G = G[G['role'] != 'retest'].copy()
HC = G['group'] == 'HC'
CON = {
    'Center 1 abnormal vs controls': (G[HC & (G['dataset'] == 'JHU_clinical')], G[G['group'] == 'JHU_abnormal'], 'grp + age'),
    'DCM vs local controls (Centers 1-2)': (G[HC & G['center'].isin(['Center1', 'Center2'])], G[G['group'] == 'DCM'], 'grp + age + C(center)'),
    'Duke abnormal vs Duke controls': (G[HC & (G['dataset'] == 'Duke')], G[G['group'] == 'Duke_abnormal'], 'grp + age'),
    'Spine Generic compression vs controls': (G[HC & (G['dataset'] == 'SpineGeneric')], G[G['group'] == 'SG_compression'], 'grp + age + C(site)'),
}


def frame(c, k):
    return pd.concat([c.assign(grp=0), k.assign(grp=1)], ignore_index=True)


def run(G_, tag):
    out = []
    for nm, (c, k, rhs) in CON.items():
        c = G_[G_['img_id'].isin(c['img_id'])]; k = G_[G_['img_id'].isin(k['img_id'])]
        d = frame(c, k)
        for mk in MARKERS:
            r = group_effect(d, f'g_{mk}', rhs)
            r.update(contrast=nm, marker=MARKERS[mk], n_ctrl=len(c), n_case=len(k),
                     age_ctrl=f"{c['age'].mean():.1f} ({c['age'].min():.0f}-{c['age'].max():.0f})",
                     age_case=f"{k['age'].mean():.1f} ({k['age'].min():.0f}-{k['age'].max():.0f})")
            out.append(r)
    R = pd.DataFrame(out)
    prim = R['marker'] == 'Whole spine'
    R.loc[prim, 'p_holm_primary'] = holm(R.loc[prim, 'p'])
    for nm in CON:
        s = (R['contrast'] == nm) & ~prim
        R.loc[s, 'p_holm_compartments'] = holm(R.loc[s, 'p'])
    R.to_csv(TAB/f'contrasts_{tag}.csv', index=False)
    return R


R = run(G, TAG)
print(R[['contrast', 'marker', 'n_ctrl', 'n_case', 'beta', 'lo', 'hi', 'p', 'p_holm_primary', 'p_holm_compartments']].round(3).to_string())
if TAG != PRIMARY:
    sys.exit()

sens = []
def add(nm, label, r): r.update(contrast=nm, analysis=label); sens.append(r)
for nm, (c, k, rhs) in CON.items():
    d = frame(c, k)
    add(nm, 'primary', group_effect(d, 'g_spine', rhs))
    add(nm, '+ sex', group_effect(d.dropna(subset=['sex']), 'g_spine', rhs + ' + C(sex)'))
    add(nm, 'quadratic age', group_effect(d, 'g_spine', rhs + ' + age_c2'))
    add(nm, 'unweighted pooling', group_effect(d, 'g_au_spine', rhs))
    if (d.loc[d.grp == 0, 'bmi'].notna().sum() >= 10) and (d.loc[d.grp == 1, 'bmi'].notna().sum() >= 10):
        add(nm, '+ BMI (where available)', group_effect(d.dropna(subset=['bmi']), 'g_spine', rhs + ' + bmi'))
    add(nm, '+ native curvature', group_effect(d.dropna(subset=['arc_chord']), 'g_spine', rhs + ' + arc_chord + c2c7_angle'))
    lo_, hi_ = c['age'].min(), c['age'].max()
    cs = d[d['age'].between(lo_, hi_)]
    r = group_effect(cs, 'g_spine', rhs); r['note'] = f'cases within control age range {lo_:.0f}-{hi_:.0f}: {int(cs.grp.sum())}/{len(k)}'
    add(nm, 'common age support', r)
    o = max(c['age'].min(), k['age'].min()), min(c['age'].max(), k['age'].max())
    ov = d[d['age'].between(*o)]
    r = group_effect(ov, 'g_spine', rhs); r['note'] = f'overlap {o[0]:.0f}-{o[1]:.0f}; controls {int((ov.grp==0).sum())}, cases {int(ov.grp.sum())}'
    add(nm, 'overlapping age range', r)
    dd = d.copy(); dd['age_c'] = dd['age'] - dd['age'].mean(); dd['gxa'] = dd['grp']*dd['age_c']
    m = ols(dd, 'g_spine', rhs + ' + gxa')
    lo, hi = m.conf_int().loc['gxa']
    add(nm, 'group x age interaction (per year)', dict(beta=m.params['gxa'], lo=lo, hi=hi, p=m.pvalues['gxa'], n=int(m.nobs)))
    if 'C(center)' in rhs:
        for ce, dc in d.groupby('center'):
            r = group_effect(dc, 'g_spine', 'grp + age'); r['note'] = f"controls {int((dc.grp==0).sum())}, cases {int(dc.grp.sum())}"
            add(nm, f'within {ce}', r)
        dd['gxc'] = dd['grp'] * (dd['center'] == 'Center2')
        m = ols(dd, 'g_spine', rhs + ' + gxc')
        lo, hi = m.conf_int().loc['gxc']
        add(nm, 'group x center interaction', dict(beta=m.params['gxc'], lo=lo, hi=hi, p=m.pvalues['gxc'], n=int(m.nobs)))
        pooled = G[HC & ~G['dataset'].isin(['JHU_clinical', 'MCW', 'JHU_DCM'])]
        dp = frame(pd.concat([c, pooled]), k)
        dp['site2'] = np.where(dp['center'].isin(['Center1', 'Center2']), dp['center'], dp['dataset'])
        add(nm, 'secondary: all healthy controls, + C(source)', group_effect(dp, 'g_spine', 'grp + age + C(site2)'))
        add(nm, 'secondary: all healthy controls, + protocol', group_effect(dp, 'g_spine', 'grp + age + C(protocol)'))
    else:
        dc = d
    pa, pb = [], []
    for ce, dc in (d.groupby('center') if 'C(center)' in rhs else [(None, d)]):
        a_, b_ = match_pairs(dc[dc.grp == 0], dc[dc.grp == 1]); pa.append(a_); pb.append(b_)
    a_, b_ = pd.concat(pa), pd.concat(pb)
    dif = b_['g_spine'].values - a_['g_spine'].values
    lo, hi = boot_ci(lambda i: dif[i].mean(), len(dif))
    add(nm, 'age-matched pairs (within center, no replacement, caliper 3 y)',
        dict(beta=dif.mean(), lo=lo, hi=hi, p=wilcoxon(dif).pvalue if len(dif) > 5 else np.nan, n=len(dif)))
    if nm.startswith('Center 1'):
        ok = d[(d['age_reader_range'].fillna(0) <= 1)]
        add(nm, 'excluding reader age disagreement > 1 y', group_effect(ok, 'g_spine', rhs))
        ps = frame(c, G[G['group'] == 'JHU_priorsurg'])
        add(nm, 'prior spine surgery vs controls (descriptive)', group_effect(ps, 'g_spine', rhs))
        d2 = frame(c, pd.concat([k, G[G['group'] == 'JHU_priorsurg']]))
        add(nm, 'cases including prior spine surgery', group_effect(d2, 'g_spine', rhs))

S = pd.DataFrame(sens)
S.to_csv(TAB/'contrast_sensitivity.csv', index=False)
print(S[['contrast', 'analysis', 'n', 'beta', 'lo', 'hi', 'p', 'note']].round(3).to_string() if 'note' in S else S.round(3).to_string())

# calibration-uncertainty bootstrap for the primary endpoint
B = 1000
Sp = pd.read_csv(ROOT/'splits.csv')
F = {k: pd.read_csv(ROOT/'preds'/f'{TAG}_f{k}.csv').set_index('img_id') for k in range(5)}
Sp = Sp[Sp['img_id'].isin(F[0].index) & (Sp['role'] != 'retest')].reset_index(drop=True)
P = np.stack([F[k].loc[Sp['img_id'], 'a_spine'].values for k in range(5)], 1)
isdev = (Sp['role'] == 'dev').values; fo = Sp['fold'].values.astype(int); age = Sp['age'].values
idx = {i: n for n, i in enumerate(Sp['img_id'])}
rng = np.random.default_rng(SEED)
boot = {nm: [] for nm in CON}
for b in range(B):
    samp = {k: rng.choice(np.where(isdev & (fo == k))[0], (isdev & (fo == k)).sum()) for k in range(5)}
    gap = np.full(len(Sp), np.nan)
    fk = {k: cal.fit(age[samp[k]], P[samp[k], k])[1] for k in range(5)}
    gap[~isdev] = np.mean([cal.invert(fk[k], P[~isdev, k]) for k in range(5)], 0) - age[~isdev]
    for j in range(5):
        oth = np.concatenate([samp[k] for k in range(5) if k != j])
        fg = cal.fit(age[oth], P[oth, fo[oth]])[1]
        m = isdev & (fo == j); gap[m] = cal.invert(fg, P[m, j]) - age[m]
    for nm, (c, k, rhs) in CON.items():
        d = frame(c, k)
        d = d.iloc[rng.integers(0, len(d), len(d))].copy()
        d['g_b'] = gap[[idx[i] for i in d['img_id']]]
        try:
            boot[nm].append(smf.ols(f'g_b ~ {rhs}', data=d).fit().params['grp'])
        except Exception:
            pass
cb = []
for nm in CON:
    v = np.array(boot[nm]); lo, hi = np.percentile(v, [2.5, 97.5])
    cb.append(dict(contrast=nm, beta_boot_median=np.median(v), lo=lo, hi=hi, p_two_sided=2*min((v <= 0).mean(), (v >= 0).mean()), B=len(v)))
pd.DataFrame(cb).to_csv(TAB/'contrast_calibration_bootstrap.csv', index=False)
print(pd.DataFrame(cb).round(3).to_string())
