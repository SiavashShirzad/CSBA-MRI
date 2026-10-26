import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Age gap versus conventional DCM imaging measures as correlates of mJOA, frailty and severity
G = load_gaps(); P = G[G['role'] != 'retest']
Dc = pd.read_csv(ROOT/'dcm_clinical_raw.csv'); Dc['ID_s'] = Dc['ID'].astype(str).str.strip()
D = P[P['group'] == 'DCM'].merge(Dc.drop(columns=['Age', 'Sex']), on='ID_s', how='left')
D['sev'] = D['Severity'].map({'Mild': 0, 'Moderate': 1, 'Severe': 2})
D['t2hi'] = D['Presence of T2 hyperintensity'].dropna().astype(str).str.upper().str[0].map({'Y': 1, 'N': 0}).reindex(D.index)
D['mfi'] = pd.to_numeric(D['mFI-5'], errors='coerce')
# sign so that higher = worse (more compression / smaller cord)
PRED = [('Age gap', 'g_spine', 1), ('MSCC', 'MSCC_diameter_AP', 1), ('Cord T2 hyperintensity', 't2hi', None),
        ('Cord cross-sectional area (smaller)', 'area', -1), ('Cord AP diameter (smaller)', 'diameter_AP', -1),
        ('Cord flattening (lower eccentricity)', 'eccentricity', -1), ('Cord solidity (lower)', 'solidity', -1)]
OUTC = [('Baseline mJOA', 'mJOA', 'ols'), ('Frailty (mFI-5)', 'mfi', 'ols'), ('Severity category', 'sev', 'ord')]
COV = ['age', 'sex', 'center']


def fit(d, y, x, kind):
    if kind == 'ols':
        m = smf.ols(f'{y} ~ x + age + C(sex) + C(center)', d).fit(cov_type='HC3'); lo, hi = m.conf_int().loc['x']
        return m.params['x'], lo, hi, m.pvalues['x']
    X = pd.get_dummies(d[['x'] + COV], columns=['sex', 'center'], drop_first=True).astype(float)
    m = OrderedModel(d[y].astype(int), X, distr='logit').fit(method='bfgs', disp=0); lo, hi = m.conf_int().loc['x']
    return np.exp(m.params['x']), np.exp(lo), np.exp(hi), m.pvalues['x']


rows = []
for sample in ['available', 'common']:
    for on, y, kind in OUTC:
        base = D.dropna(subset=[y] + COV)
        if sample == 'common': base = base.dropna(subset=[c for _, c, _ in PRED])
        for pn, c, sgn in PRED:
            d = base.dropna(subset=[c]).copy()
            d['x'] = d[c] if sgn is None else sgn*(d[c] - D[c].mean())/D[c].std()  # SD of all DCM patients
            b, lo, hi, p = fit(d, y, 'x', kind)
            rows.append(dict(sample=sample, outcome=on, predictor=pn, n=len(d), est=b, lo=lo, hi=hi, p=p,
                             scale=('OR' if kind == 'ord' else 'points') + (' (present vs absent)' if sgn is None else ' per SD')))
R = pd.DataFrame(rows); R.to_csv(TAB/'imaging_vs_gap_clinical.csv', index=False)

# joint model: does the gap add to conventional imaging?
J = []
for on, y, kind in OUTC[:2]:
    d = D.dropna(subset=[y, 'g_spine', 'MSCC_diameter_AP', 't2hi', 'area'] + COV).copy()
    for c in ['g_spine', 'MSCC_diameter_AP', 'area']: d[c + '_z'] = (d[c] - D[c].mean())/D[c].std()
    base = smf.ols(f'{y} ~ MSCC_diameter_AP_z + t2hi + area_z + age + C(sex) + C(center)', d).fit(cov_type='HC3')
    full = smf.ols(f'{y} ~ g_spine_z + MSCC_diameter_AP_z + t2hi + area_z + age + C(sex) + C(center)', d).fit(cov_type='HC3')
    lo, hi = full.conf_int().loc['g_spine_z']
    J.append(dict(outcome=on, n=len(d), gap_est=full.params['g_spine_z'], lo=lo, hi=hi, p=full.pvalues['g_spine_z'],
                  r2_imaging=base.rsquared, r2_imaging_gap=full.rsquared))
J = pd.DataFrame(J); J.to_csv(TAB/'imaging_vs_gap_joint.csv', index=False)
# how much of the gap is explained by conventional measures
C = D.dropna(subset=['g_spine', 'MSCC_diameter_AP', 't2hi', 'area', 'diameter_AP'] + COV)
cr = {c: spearmanr(C['g_spine'], C[c])[0] for c in ['MSCC_diameter_AP', 't2hi', 'area', 'diameter_AP']}
mg = smf.ols('g_spine ~ MSCC_diameter_AP + t2hi + area + diameter_AP + eccentricity + solidity', C).fit()
json.dump(dict(spearman=cr, r2_gap_on_imaging=mg.rsquared, n=len(C)), open(TAB/'imaging_vs_gap_overlap.json', 'w'), indent=1)
print(cr, 'R2 gap~imaging', round(mg.rsquared, 3), len(C))
pd.set_option('display.width', 200)
print(R.round(3).to_string()); print(J.round(3).to_string())
