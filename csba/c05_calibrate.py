import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

GRID = np.linspace(10, 100, 9001)
DEG = 2


def fit(age, pred, deg=DEG):
    k = np.isfinite(age) & np.isfinite(pred)
    co = np.polyfit(age[k], pred[k], deg)
    fg = np.poly1d(co)(GRID)
    lo, hi = 18, 90
    w = (GRID >= lo) & (GRID <= hi)
    if not np.all(np.diff(fg[w]) > 0):
        co = np.polyfit(age[k], pred[k], 1); fg = np.poly1d(co)(GRID)
    return co, fg


def invert(fg, pred):
    return np.interp(pred, fg, GRID)


def calibrate(variant, tag=None, cols=None):
    tag = tag or variant
    S = pd.read_csv(ROOT/'splits.csv')
    F = {k: pd.read_csv(ROOT/'preds'/f'{tag}_f{k}.csv').set_index('img_id') for k in range(5)}
    cols = cols or [c for c in F[0].columns if c.startswith(('a_', 'au_'))]
    S = S[S['img_id'].isin(F[0].index)].reset_index(drop=True)
    dev = S[S['role'] == 'dev']
    out = S[['img_id', 'ID_s', 'dataset', 'center', 'site', 'group', 'role', 'fold', 'age', 'sex']].copy()
    for c in cols:
        P = np.stack([F[k].loc[S['img_id'], c].values for k in range(5)], 1)
        isdev = (S['role'] == 'dev').values; fo = S['fold'].values.astype(int)
        raw = np.where(isdev, P[np.arange(len(S)), np.clip(fo, 0, 4)], np.nanmean(P, 1))
        own = {}
        for k in range(5):
            d = dev[dev['fold'] == k]
            own[k] = fit(d['age'].values, F[k].loc[d['img_id'], c].values)
        gap = np.full(len(S), np.nan)
        nd = ~isdev
        gap[nd] = np.mean([invert(own[k][1], P[nd, k]) for k in range(5)], 0) - S.loc[nd, 'age'].values
        for j in range(5):
            d = dev[dev['fold'] != j]
            oof = np.array([F[int(f)].loc[i, c] for i, f in zip(d['img_id'], d['fold'])])
            _, fg = fit(d['age'].values, oof)
            m = isdev & (fo == j)
            gap[m] = invert(fg, P[m, j]) - S.loc[m, 'age'].values
        out[f'raw_{c[2:] if c.startswith("a_") else c}'] = raw
        out[f'g_{c[2:] if c.startswith("a_") else c}'] = gap
    for c in [c for c in F[0].columns if c.startswith(('se_', 'sm_'))]:
        P = np.stack([F[k].loc[S['img_id'], c].values for k in range(5)], 1)
        isdev = (S['role'] == 'dev').values; fo = S['fold'].values.astype(int)
        out[c] = np.where(isdev, P[np.arange(len(S)), np.clip(fo, 0, 4)], np.nanmean(P, 1))
    out.to_csv(ROOT/f'gaps_{tag}.csv', index=False)
    return out


if __name__ == '__main__':
    for v in sys.argv[1:]:
        o = calibrate(v)
        print(v, len(o), o.groupby(['role'])['g_spine'].agg(['size', 'mean', 'std']).round(2).to_dict('index'))
