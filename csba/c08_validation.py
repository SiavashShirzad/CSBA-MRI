import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import cohen_kappa_score

G = load_gaps()
P = G[G['role'] != 'retest'].copy()
OUT = {}
LV = ['C3', 'C4', 'C5', 'C6', 'C7']


def zs(x): return (x - x.mean()) / x.std()


def coef(m, t):
    lo, hi = m.conf_int().loc[t]; return dict(beta=m.params[t], lo=lo, hi=hi, p=m.pvalues[t], n=int(m.nobs))


# 1. reader-based reference standard, Center 1
JC = pd.read_csv(ROOT/'jhu_clinical.csv'); JC['ID_s'] = JC['mrn'].astype(str)
J = P[P['dataset'] == 'JHU_clinical'].merge(JC.drop(columns=['age', 'sex', 'height_m', 'weight_kg']), on='ID_s', how='left')
rows = []
for base, d in [('controls + abnormal', J[J['group'].isin(['HC', 'JHU_abnormal'])]), ('all incl. prior surgery', J)]:
    for x in ['n_spinalstenosis', 'n_foraminalstenosis', 'n_hyperintensity']:
        dd = d.dropna(subset=[x, 'g_spine']).copy(); dd['x'] = dd[x]
        m = ols(dd, 'g_spine', 'x + age'); r = coef(m, 'x'); r.update(sample=base, predictor=x, unit='per level')
        rr = spearmanr(sm.OLS(dd['g_spine'], sm.add_constant(dd['age'])).fit().resid, sm.OLS(dd[x], sm.add_constant(dd['age'])).fit().resid)
        r.update(partial_rho=rr[0], partial_rho_p=rr[1]); rows.append(r)
    dd = d.dropna(subset=['n_hyperintensity']).copy(); dd['x'] = (dd['n_hyperintensity'] > 0).astype(int)
    r = coef(ols(dd, 'g_spine', 'x + age'), 'x'); r.update(sample=base, predictor='any cord T2 hyperintensity', unit='yes vs no'); rows.append(r)
    for s in ['exam_umnsigns', 'exam_motordeficit', 'exam_gaitabnormal', 'sx_motor', 'sx_gaitabnormal', 'sx_neckpain']:
        dd = d.dropna(subset=[s]).copy(); dd['x'] = dd[s]
        if dd['x'].nunique() < 2 or dd['x'].sum() < 5: continue
        r = coef(ols(dd, 'g_spine', 'x + age'), 'x'); r.update(sample=base, predictor=s, unit='yes vs no', n_pos=int(dd['x'].sum())); rows.append(r)
R1 = pd.DataFrame(rows); R1.to_csv(TAB/'reader_reference.csv', index=False); OUT['reader'] = R1
print('READER REFERENCE\n', R1[['sample', 'predictor', 'n', 'beta', 'lo', 'hi', 'p', 'partial_rho']].round(3).to_string())

x = pd.read_excel(RAW/'JHU/SpineCervicalVariables.xlsx')
FT = {k: [f'{k}_{l}' for l in ['c2', 'c3', 'c4', 'c5', 'c6', 'c7', 't1']] for k in ['spinalstenosis', 'foraminalstenosis', 'hyperintensity']}
for cs in FT.values():
    for c in cs: x[c] = x[c].where(x[c].isin([0, 1]))
kap = []
pairs = []
for mrn, d in x.groupby('mrn'):
    d = d.drop_duplicates('student')
    if len(d) >= 2: pairs.append(d.iloc[:2])
for k, cs in FT.items():
    a = np.concatenate([p.iloc[0][cs].values for p in pairs]).astype(float); b = np.concatenate([p.iloc[1][cs].values for p in pairs]).astype(float)
    ok = np.isfinite(a) & np.isfinite(b)
    kap.append(dict(finding=k, level='per level', n_pairs=int(ok.sum()), kappa=cohen_kappa_score(a[ok], b[ok]), agreement=float(np.mean(a[ok] == b[ok]))))
allc = sum(FT.values(), [])
a = np.array([float(np.nansum(p.iloc[0][allc].values) > 0) if p.iloc[0][allc].notna().all() else np.nan for p in pairs])
b = np.array([float(np.nansum(p.iloc[1][allc].values) > 0) if p.iloc[1][allc].notna().all() else np.nan for p in pairs])
ok = np.isfinite(a) & np.isfinite(b)
kap.append(dict(finding='any finding (control vs abnormal assignment)', level='participant', n_pairs=int(ok.sum()),
                kappa=cohen_kappa_score(a[ok], b[ok]), agreement=float(np.mean(a[ok] == b[ok]))))
K = pd.DataFrame(kap); K.to_csv(TAB/'reader_agreement.csv', index=False)
print('\nREADER AGREEMENT\n', K.round(3).to_string())


# 2. focal (level-specific) analyses
def parse_levels(s):
    s = str(s).upper().replace(' ', '')
    out = set()
    for tok in re.split(r'[,;]', s):
        m = re.match(r'^C(\d)/?(\d|T1)$', tok) or re.match(r'^C(\d)(\d|T1)$', tok)
        if m:
            a = int(m.group(1)); out |= {f'C{a}', 'T1' if m.group(2) == 'T1' else f'C{m.group(2)}'}; continue
        m = re.match(r'^C(\d)-C?(\d)$', tok)
        if m: out |= {f'C{i}' for i in range(int(m.group(1)), int(m.group(2))+1)}
    return out


def focal(df, aff, label):
    L = []
    for _, r in df.iterrows():
        A = aff(r)
        if A is None: continue
        A = A & set(LV)
        if not A or A == set(LV): continue
        for lv in LV:
            if pd.notna(r[f'g_{lv}']): L.append(dict(pid=r['img_id'], level=lv, gap=r[f'g_{lv}'], affected=int(lv in A)))
    L = pd.DataFrame(L)
    if L.empty or L['pid'].nunique() < 8: return None
    m = smf.mixedlm('gap ~ affected + C(level)', L, groups=L['pid']).fit()
    lo, hi = m.conf_int().loc['affected']
    L['gc'] = L['gap'] - L.groupby('pid')['gap'].transform('mean')
    w = L.groupby(['pid', 'affected'])['gc'].mean().unstack().dropna()
    return dict(analysis=label, participants=int(L['pid'].nunique()), beta=m.params['affected'], lo=lo, hi=hi,
                p=m.pvalues['affected'], wilcoxon_p=wilcoxon(w[1] - w[0]).pvalue, mean_diff=float((w[1] - w[0]).mean()))


Dc = pd.read_csv(ROOT/'dcm_clinical_raw.csv'); Dc['ID_s'] = Dc['ID'].astype(str).str.strip()
D = P[P['group'] == 'DCM'].merge(Dc.drop(columns=['Age', 'Sex']), on='ID_s', how='left')
fo = []
for lab, d, f in [
    ('Center 1 abnormal: reader spinal stenosis levels', J[J['group'] == 'JHU_abnormal'],
     lambda r: {lv for lv in LV if r.get(f'spinalstenosis_{lv.lower()}', np.nan) >= 0.5} if pd.notna(r.get('spinalstenosis_c5')) else None),
    ('Center 1 abnormal: reader foraminal stenosis levels', J[J['group'] == 'JHU_abnormal'],
     lambda r: {lv for lv in LV if r.get(f'foraminalstenosis_{lv.lower()}', np.nan) >= 0.5} if pd.notna(r.get('foraminalstenosis_c5')) else None),
    ('DCM: level of maximum compression', D, lambda r: parse_levels(r['Level of Max Compression']) if pd.notna(r['Level of Max Compression']) else None),
    ('Spine Generic compression: reported levels', P[P['group'] == 'SG_compression'], lambda r: parse_levels(r['sg_level']) if pd.notna(r['sg_level']) else None)]:
    r = focal(d, f, lab)
    if r: fo.append(r)
FO = pd.DataFrame(fo); FO.to_csv(TAB/'focal_levels.csv', index=False)
print('\nFOCAL\n', FO.round(3).to_string())


# 3. clinical endpoints in DCM
D['gap_z'] = zs(D['g_spine'])
D['sev'] = D['Severity'].map({'Mild': 0, 'Moderate': 1, 'Severe': 2})
D['t2hi'] = D['Presence of T2 hyperintensity'].dropna().astype(str).str.upper().str[0].map({'Y': 1, 'N': 0}).reindex(D.index)
D['dur'] = D['duration of symptoms (months)']
D['mfi'] = pd.to_numeric(D['mFI-5'], errors='coerce')
cl = []
cov = 'age + C(sex) + C(center)'
specs = [('Baseline mJOA', 'mJOA', f'gap_z + {cov}', 'ols'),
         ('Frailty (mFI-5)', 'mfi', f'gap_z + {cov}', 'ols'),
         ('MSCC (AP diameter)', 'MSCC_diameter_AP', f'gap_z + {cov}', 'ols'),
         ('Cord T2 hyperintensity', 't2hi', f'gap_z + {cov}', 'logit'),
         ('Symptom duration (months)', 'dur', f'gap_z + {cov}', 'ols')]
for nm, y, rhs, kind in specs:
    d = D.dropna(subset=[y, 'gap_z', 'age', 'sex'] + (['mJOA'] if 'mJOA +' in rhs else []) + (['mfi'] if 'mfi' in rhs else []))
    if kind == 'ols': m = smf.ols(f'{y} ~ {rhs}', d).fit(cov_type='HC3')
    else: m = smf.logit(f'{y} ~ {rhs}', d).fit(disp=0)
    r = coef(m, 'gap_z'); r.update(endpoint=nm, model=f'{y} ~ {rhs}', scale='OR' if kind == 'logit' else 'per 1 SD gap')
    if kind == 'logit': r.update(beta=np.exp(r['beta']), lo=np.exp(r['lo']), hi=np.exp(r['hi']))
    r['centers'] = d['center'].value_counts().to_dict(); cl.append(r)
d = D.dropna(subset=['sev', 'gap_z', 'age', 'sex'])
X = pd.get_dummies(d[['gap_z', 'age', 'sex', 'center']], columns=['sex', 'center'], drop_first=True).astype(float)
om = OrderedModel(d['sev'].astype(int), X, distr='logit').fit(method='bfgs', disp=0)
lo, hi = om.conf_int().loc['gap_z']
cl.append(dict(endpoint='Severity (mild/moderate/severe, ordinal)', beta=np.exp(om.params['gap_z']), lo=np.exp(lo), hi=np.exp(hi),
               p=om.pvalues['gap_z'], n=len(d), model='proportional-odds logit, gap_z + age + sex + center', scale='OR per 1 SD gap',
               centers=d['center'].value_counts().to_dict()))
for ce, dc in D.groupby('center'):
    for nm, y, rhs in [('Baseline mJOA', 'mJOA', 'gap_z + age + C(sex)')]:
        d = dc.dropna(subset=[y, 'gap_z', 'age', 'sex'] + (['mJOA'] if 'mJOA +' in rhs else []))
        m = smf.ols(f'{y} ~ {rhs}', d).fit(cov_type='HC3'); r = coef(m, 'gap_z')
        r.update(endpoint=f'{nm}, {ce.replace("Center", "Center ")} only', model=f'{y} ~ {rhs}', scale='per 1 SD gap', centers={ce: len(d)}); cl.append(r)
CL = pd.DataFrame(cl); CL.to_csv(TAB/'clinical_endpoints.csv', index=False)
D[['ID_s', 'center', 'age', 'sex', 'g_spine', 'mJOA', 'sev', 'mfi', 't2hi']].assign(ID_s=lambda d: d['ID_s'].map(pseudo_id)).to_csv(ROOT/'dcm_clinical_frame.csv', index=False)
print('\nCLINICAL\n', CL[['endpoint', 'n', 'beta', 'lo', 'hi', 'p', 'scale']].round(3).to_string())
print('DCM clinical availability by center:', D.groupby('center')[['mJOA', 'Severity', 'MSCC_diameter_AP']].count().to_dict())


# 4. proteomics (exploratory)
clin = set(['Severity', 'improved', 'mJOA_Delta', 'ID', 'Anesthesia', 'mFI-5', 'mJOA_postop3mo', 'duration of symptoms (months)',
            'Level of Max Compression', 'Presence of T2 hyperintensity', 'area', 'diameter_AP', 'diameter_RL', 'eccentricity', 'solidity',
            'MSCC_diameter_AP', 'Batch', 'Pathology', 'mJOA', 'Age', 'Sex', 'ID_s',
            'APOE4'])  # >20% missing before imputation; excluded as in the source proteomic pipeline
prot = [c for c in Dc.columns if c not in clin and pd.api.types.is_numeric_dtype(Dc[c])]
Q = D.dropna(subset=['Batch', 'sex', 'GFAP']).copy()
Zc = pd.get_dummies(Q[['age', 'sex', 'Batch', 'center']].astype({'Batch': str}), columns=['sex', 'Batch', 'center'], drop_first=True).astype(float)
Zc = sm.add_constant(Zc)
def resid(v): return sm.OLS(v, Zc).fit().resid
rg = resid(Q['g_spine'].rank().values.astype(float))
pr = []
RES = {}
for c in prot:
    v = Q[c]
    if v.notna().sum() < 0.8*len(Q) or v.nunique() < 5: continue
    k = v.notna().values
    rv = sm.OLS(v[k].rank().values.astype(float), Zc[k]).fit().resid
    rgk = sm.OLS(Q['g_spine'][k].rank().values.astype(float), Zc[k]).fit().resid
    rho, p = spearmanr(rgk, rv); RES[c] = (np.asarray(rgk), np.asarray(rv))
    pr.append(dict(protein=c, n=int(k.sum()), rho=rho, p=p))
PR = pd.DataFrame(pr)
from statsmodels.stats.multitest import multipletests
PR['q'] = multipletests(PR['p'], method='fdr_bh')[1]
rng = np.random.default_rng(SEED)
null = []
for b in range(1000):
    perm = rng.permutation(len(Q)); cnt = 0
    gperm = rg[perm] if isinstance(rg, np.ndarray) else rg.values[perm]
    for c in PR['protein']:
        v = Q[c]; k = v.notna().values
        if k.all():
            cnt += spearmanr(gperm, resid(v.rank().values.astype(float)))[1] < .05
        else:
            cnt += spearmanr(RES[c][0][rng.permutation(k.sum())], RES[c][1])[1] < .05
    null.append(cnt)
null = np.array(null); obs = int((PR['p'] < .05).sum())
PR = PR.sort_values('p'); PR.to_csv(TAB/'proteomics.csv', index=False)
psum = dict(n_participants=len(Q), n_proteins=len(PR), n_nominal=obs, expected=0.05*len(PR), perm_p=float((null >= obs).mean()),
            min_q=float(PR['q'].min()), max_abs_rho=float(PR['rho'].abs().max()), centers=Q['center'].value_counts().to_dict())
json.dump(psum, open(TAB/'proteomics_summary.json', 'w'), indent=1, default=str)
np.save(TAB/'proteomics_null.npy', null)
import pickle; pickle.dump(dict(RES=RES, rg=np.asarray(rg), complete={c: bool(Q[c].notna().all()) for c in PR['protein']}), open(ROOT/'proteomics_resid.pkl', 'wb'))
top = PR['protein'].iloc[0]; np.savez(ROOT/'proteomics_top.npz', protein=top, g=RES[top][0], v=RES[top][1])
print('\nPROTEOMICS', psum); print(PR.head(8).round(3).to_string())


# 5. acquisition and source
ac = []
H = P[(P['group'] == 'HC') & P['role'].isin(['dev', 'test'])]
m = ols(H, 'g_spine', 'age + C(dataset, Treatment("Duke"))')
for t in m.params.index:
    if 'dataset' in t: r = coef(m, t); r.update(analysis='healthy development+test: dataset vs Duke', term=t); ac.append(r)
Hall = P[P['group'] == 'HC']
m = ols(Hall, 'g_spine', 'age + C(protocol)'); r = coef(m, 'C(protocol)[T.3D]'); r.update(analysis='all healthy: 3D vs 2D (confounded with source)'); ac.append(r)
sg = H[H['dataset'] == 'SpineGeneric'].dropna(subset=['vendor'])
m = ols(sg, 'g_spine', 'age + C(vendor)'); ft = m.wald_test_terms(skip_single=False)
for t in m.params.index:
    if 'vendor' in t: r = coef(m, t); r.update(analysis='Spine Generic healthy: vendor vs GE', term=t); ac.append(r)
du = H[H['dataset'] == 'Duke']
r = coef(ols(du, 'g_spine', 'age + vox_max'), 'vox_max'); r.update(analysis='Duke healthy: slice thickness (per mm)'); ac.append(r)
AQ = pd.read_csv(ROOT/'jhu_dcm_acquisition.csv'); AQ['vendor'] = AQ['vendor'].str.upper()
jd = P[P['dataset'] == 'JHU_DCM'].merge(AQ[['ID_s', 'field_T']], on='ID_s', how='left')
jd['f3'] = (jd['field_T'] >= 2.9).astype(int)
r = coef(ols(jd.dropna(subset=['field_T']), 'g_spine', 'f3 + age'), 'f3'); r.update(analysis='Center 1 DCM: 3 T vs <3 T'); ac.append(r)
jd = jd.merge(Dc[['ID_s', 'mJOA']], on='ID_s', how='left')
AC = pd.DataFrame(ac); AC.to_csv(TAB/'acquisition.csv', index=False)
print('\nACQUISITION\n', AC[['analysis', 'term', 'n', 'beta', 'lo', 'hi', 'p']].round(3).to_string() if 'term' in AC else AC.round(3).to_string())
print('Center 1 DCM field strength', jd['field_T'].value_counts().to_dict(), '| protocol by group/center:')
print(P.groupby(['center', 'group'])['vox_max'].describe()[['count', '50%', 'min', 'max']].round(2).to_string())


# 6. reliability: repeat sagittal T2 series from the same examination
Rt = G[G['role'] == 'retest'][['ID_s', 'dataset', 'g_spine', 'raw_spine', 'marrow_csf_ratio', 'vox_max', 'img_id']]
Pm = P[['ID_s', 'dataset', 'g_spine', 'raw_spine', 'marrow_csf_ratio', 'vox_max', 'role']]
pairs = Rt.merge(Pm, on=['ID_s', 'dataset'], suffixes=('_b', '_a'))
pairs = pairs[pairs['role'] != 'dev']
pairs['lr'] = np.abs(np.log(pairs['marrow_csf_ratio_b'] / pairs['marrow_csf_ratio_a']))


def icc21(a, b):
    X = np.c_[a, b]; n, k = X.shape; gm = X.mean()
    msr = k*((X.mean(1) - gm)**2).sum()/(n-1); msc = n*((X.mean(0) - gm)**2).sum()/(k-1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm)**2).sum(); mse = sse/((n-1)*(k-1))
    return (msr - mse)/(msr + (k-1)*mse + k*(msc - mse)/n)


rel = []
for lab, d in [('all repeat series', pairs), ('same contrast (|log marrow/CSF ratio| < 0.2)', pairs[pairs['lr'] < 0.2])]:
    d = d.drop_duplicates('ID_s')
    if len(d) < 10: continue
    dif = d['g_spine_b'] - d['g_spine_a']
    rel.append(dict(set=lab, n=len(d), icc_2_1=icc21(d['g_spine_a'].values, d['g_spine_b'].values),
                    bias=dif.mean(), loa_lo=dif.mean()-1.96*dif.std(), loa_hi=dif.mean()+1.96*dif.std(),
                    within_sd=float(np.sqrt(np.mean(dif**2)/2)), datasets=d['dataset'].value_counts().to_dict()))
RL = pd.DataFrame(rel); RL.to_csv(TAB/'reliability.csv', index=False)
print('\nRELIABILITY\n', RL.round(3).to_string())


# 7. posture / native alignment
po = []
for x in ['arc_chord', 'c2c7_angle']:
    d = H.dropna(subset=[x]).copy(); d['xz'] = zs(d[x])
    r = coef(ols(d, 'g_spine', 'xz + age + C(dataset)'), 'xz'); r.update(analysis=f'healthy: {x} (per SD)'); po.append(r)
    d = P[P['group'].isin(['HC', 'JHU_abnormal', 'DCM', 'Duke_abnormal', 'SG_compression'])].dropna(subset=[x]).copy()
    po.append(dict(analysis=f'{x} mean by group', **{g: float(v) for g, v in d.groupby('group')[x].mean().items()}))
if (ROOT/'preds'/f'{PRIMARY}_jitter_f0.csv').exists():
    Gj = pd.read_csv(ROOT/f'gaps_{PRIMARY}_jitter.csv')
    j = P.merge(Gj[['img_id', 'g_spine']], on='img_id', suffixes=('', '_j'))
    j = j[j['role'] != 'dev']
    dif = j['g_spine_j'] - j['g_spine']
    po.append(dict(analysis=f'vertebral anchor jitter (SD {JITTER_MM} mm): ICC original vs perturbed', n=len(j),
                   beta=icc21(j['g_spine'].values, j['g_spine_j'].values), lo=dif.mean(), hi=dif.std()))
PO = pd.DataFrame(po); PO.to_csv(TAB/'posture.csv', index=False)
print('\nPOSTURE\n', PO.round(3).to_string())

json.dump({k: v.to_dict('records') for k, v in OUT.items()}, open(TAB/'validation_bundle.json', 'w'), default=str)
