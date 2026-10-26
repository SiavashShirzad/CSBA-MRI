import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr, wilcoxon
from scipy.optimize import linear_sum_assignment

PRIMARY = json.load(open(ROOT/'tables'/'harm_decision.json'))['rule']['primary'] if (ROOT/'tables'/'harm_decision.json').exists() else 'main'
N_BOOT = 4000
SEED = 42
TAB = ROOT/'tables'; TAB.mkdir(exist_ok=True)
FIG = ROOT/'logs'/'qc_figs'; FIG.mkdir(parents=True, exist_ok=True)
BONE = ['C3', 'C4', 'C5', 'C6', 'C7']
MARKERS = {'spine': 'Whole spine', 'cord': 'Cord', 'csf': 'CSF', 'disc': 'Disc', 'bone': 'Vertebral bone'}


def fmt_p(p):
    if not np.isfinite(p): return 'NA'
    return '<.001' if p < .001 else f'{p:.3f}'.lstrip('0')


def load_gaps(tag=None):
    tag = tag or PRIMARY
    G = pd.read_csv(ROOT/f'gaps_{tag}.csv')
    for pre in ['g_', 'raw_']:
        cols = [f'{pre}{v}' for v in BONE]
        if all(c in G for c in cols): G[f'{pre}bone'] = G[cols].mean(1)
    Q = pd.read_csv(ROOT/'cohort_images_qc.csv')
    keep = ['img_id', 'vox_max', 'vox_min', 'arc_chord', 'c2c7_angle', 'marrow_csf_ratio', 'disc_csf_ratio',
            'height_m', 'weight_kg', 'vendor', 'model', 'sg_problem', 'sg_level', 'age_reader_range', 'img']
    G = G.merge(Q[[c for c in keep if c in Q]], on='img_id', how='left')
    G['protocol'] = np.where(G['vox_max'] < 2.0, '3D', '2D')
    G['bmi'] = G['weight_kg'] / G['height_m']**2
    G.loc[(G['bmi'] < 12) | (G['bmi'] > 70), 'bmi'] = np.nan
    return G


def boot_ci(f, n, B=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.array([f(rng.integers(0, n, n)) for _ in range(B)])
    v = v[np.isfinite(v)]
    return np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)


def accuracy(d, pred, label):
    a, p = d['age'].values, d[pred].values
    k = np.isfinite(p); a, p = a[k], p[k]
    e = p - a
    mae = np.mean(np.abs(e)); r = pearsonr(p, a)[0]
    lo, hi = boot_ci(lambda i: np.mean(np.abs(e[i])), len(e))
    rl, rh = boot_ci(lambda i: pearsonr(p[i], a[i])[0], len(e))
    sl = np.polyfit(a, e, 1)[0]
    sll, slh = boot_ci(lambda i: np.polyfit(a[i], e[i], 1)[0], len(e))
    return dict(set=label, n=len(e), mae=mae, mae_lo=lo, mae_hi=hi, r=r, r_lo=rl, r_hi=rh,
                bias=e.mean(), slope=sl, slope_lo=sll, slope_hi=slh, age_min=a.min(), age_max=a.max())


def ols(d, y, rhs, hc='HC3'):
    f = f'{y} ~ {rhs}'
    m = smf.ols(f, data=d).fit(cov_type=hc)
    return m


def group_effect(d, y, rhs='grp + age', term='grp'):
    d = d.dropna(subset=[y, 'grp', 'age']).copy()
    d['age_c'] = d['age'] - d['age'].mean(); d['age_c2'] = d['age_c']**2
    m = ols(d.dropna(subset=[c for c in d.columns if c in rhs]), y, rhs)
    b = m.params[term]; lo, hi = m.conf_int().loc[term]
    return dict(beta=b, lo=lo, hi=hi, p=m.pvalues[term], n=int(m.nobs),
                n_case=int(d.loc[m.model.data.row_labels, 'grp'].sum()), formula=f'{y} ~ {rhs}')


def match_pairs(a, b, cal=3.0):
    if not len(a) or not len(b): return a.iloc[:0], b.iloc[:0]
    C = np.abs(a['age'].values[:, None] - b['age'].values[None, :])
    r, c = linear_sum_assignment(np.where(C <= cal, C, 1e6))
    ok = C[r, c] <= cal
    return a.iloc[r[ok]], b.iloc[c[ok]]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj
