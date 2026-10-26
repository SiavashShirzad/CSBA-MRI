import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.model_selection import train_test_split, StratifiedKFold

C = pd.read_csv(ROOT/'cohort_images.csv')
E = pd.read_csv(ROOT/'extraction.csv')
EU = pd.read_csv(ROOT/'extraction_uniform.csv') if (ROOT/'extraction_uniform.csv').exists() else None
C = C.merge(E, on='img_id', how='left')
C['ok'] = C['ok'].fillna(False).astype(bool)
C['usable'] = C['ok'] & (C['n_verts_vol'] >= 6) & (C['n_discs_vol'] >= 5) & (C['cord_vox'] > 500)
el = C['exclusion'].isna()
C.loc[el & ~C['ok'], 'exclusion'] = 'straightening failed: ' + C.loc[el & ~C['ok'], 'reason'].astype(str)
el = C['exclusion'].isna()
C.loc[el & ~C['usable'], 'exclusion'] = 'incomplete C2-T1 coverage after straightening'


def ratio(iid):
    d = np.load(NPZ/f'{iid}.npz'); img, lab = d['image'], d['compartments']
    v, c, k = img[np.isin(lab, VERTS)], img[lab == CSF], img[np.isin(lab, DISCS)]
    return (np.median(v)/max(np.median(c), 1e-6), np.median(k)/max(np.median(c), 1e-6)) if v.size and c.size else (np.nan, np.nan)


ok = C['usable'] & C['exclusion'].isna()
rv = {i: ratio(i) for i in C.loc[ok, 'img_id']}
C['marrow_csf_ratio'] = C['img_id'].map(lambda i: rv.get(i, (np.nan, np.nan))[0])
C['disc_csf_ratio'] = C['img_id'].map(lambda i: rv.get(i, (np.nan, np.nan))[1])

C['rank_key'] = list(zip(C['n_verts_vol'].fillna(0), C['n_discs_vol'].fillna(0), C['cord_vox'].fillna(0)))
C = C.sort_values(['ID_s', 'img'])
C['primary'] = False
for pid, d in C[ok].groupby(['dataset', 'ID_s']):
    best = d.sort_values(['n_verts_vol', 'n_discs_vol', 'cord_vox'], ascending=False).index[0]
    C.loc[best, 'primary'] = True

C['role'] = None
C.loc[ok & ~C['primary'], 'role'] = 'retest'
P = C[ok & C['primary']].copy()
P['role'] = 'apply'
dev_pool = P[(P['group'] == 'HC') & P['dataset'].isin(['Duke', 'SpineGeneric', 'OpenNeuro'])].copy()
dev_pool['band'] = pd.cut(dev_pool['age'], [17, 30, 45, 60, 75, 90]).astype(str)
dev_pool['strat'] = dev_pool['dataset'] + '_' + dev_pool['band']
vc = dev_pool['strat'].value_counts()
dev_pool.loc[dev_pool['strat'].map(vc) < 10, 'strat'] = dev_pool['dataset']
P['fold'] = -1
FROZEN = ROOT/'partition_frozen.csv'
if FROZEN.exists():
    Z = pd.read_csv(FROZEN).set_index('img_id')
    hit = P['img_id'].isin(Z.index)
    P.loc[hit, 'role'] = P.loc[hit, 'img_id'].map(Z['role']).values
    P.loc[hit, 'fold'] = P.loc[hit, 'img_id'].map(Z['fold']).values
    new_dev = dev_pool[~dev_pool['img_id'].isin(Z.index)]
    assert len(new_dev) == 0, f'{len(new_dev)} new healthy development images not in frozen partition'
else:
    dev_pool = dev_pool.sort_values('img_id')
    tr_idx, te_idx = train_test_split(dev_pool.index, test_size=0.15, random_state=42, stratify=dev_pool['strat'])
    P.loc[te_idx, 'role'] = 'test'
    P.loc[tr_idx, 'role'] = 'dev'
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    d = dev_pool.loc[tr_idx]
    for k, (_, ix) in enumerate(skf.split(d, d['strat'])):
        P.loc[d.index[ix], 'fold'] = k
    P.loc[P['role'].isin(['dev', 'test']), ['img_id', 'ID_s', 'dataset', 'role', 'fold']].to_csv(FROZEN, index=False)

S = pd.concat([P, C[C['role'] == 'retest'].assign(fold=-1)])
S = S.drop(columns=['rank_key'])
S.to_csv(ROOT/'splits.csv', index=False)
C.drop(columns=['rank_key']).to_csv(ROOT/'cohort_images_qc.csv', index=False)

print('participants (primary images) by role/dataset/group:')
print(P.groupby(['role', 'dataset', 'group']).size().to_string())
print('\nretest images', int((S['role'] == 'retest').sum()), S[S['role'] == 'retest'].groupby('dataset').size().to_dict())
print('dev folds', P[P['role'] == 'dev']['fold'].value_counts().sort_index().to_dict())
print('\nexclusions after QC (images):')
print(C.groupby(['dataset', 'exclusion']).size().to_string())
