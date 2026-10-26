import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from c11_figures import panel, save, MM, GC, GL, OI

S = pd.read_csv(ROOT/'splits.csv')
S['npz'] = [str(ROOT/'npz'/f'{i}.npz') for i in S['img_id']]
S = S[[os.path.exists(p) for p in S['npz']]].reset_index(drop=True)
isdev = (S['role'] == 'dev').values; fo = S['fold'].values.astype(int)
M = None
for k in range(5):
    X = np.load(ROOT/'preds'/f'{PRIMARY}_f{k}_maps.npy', mmap_mode='r')
    assert len(X) == len(S)
    if M is None: M = np.zeros(X.shape, np.float32); W = np.zeros(len(S), np.float32)
    use = ~isdev | (fo == k)
    M[use] += np.asarray(X[use], np.float32); W[use] += 1
M /= W[:, None, None, None]
keep = (S['role'] != 'retest').values
S, M = S[keep].reset_index(drop=True), M[keep]
age = S['age'].values
ref = (S['role'] == 'dev').values
R = M[ref]; a = age[ref]
fin = np.isfinite(R)
n = fin.sum(0); Rz = np.where(fin, R, 0)
sx = (fin*a[:, None, None, None]).sum(0); sy = Rz.sum(0); sxx = (fin*a[:, None, None, None]**2).sum(0); sxy = (Rz*a[:, None, None, None]).sum(0)
beta = (n*sxy - sx*sy)/np.maximum(n*sxx - sx**2, 1e-6); alpha = (sy - beta*sx)/np.maximum(n, 1)
anat = n > 0.5*n.max()
np.savez_compressed(ROOT/'normative_voxel.npz', alpha=alpha, beta=beta, n=n, anat=anat)
use = anat & (beta > 0.3)
gap = (M - (alpha + beta*age[:, None, None, None])) / np.where(use, beta, np.nan)
def dcm_vox_beta():
    sel = (S['center'].isin(['Center1', 'Center2']) & S['group'].isin(['HC', 'DCM'])).values
    d = S[sel]
    X = np.column_stack([np.ones(len(d)), (d['group'] == 'DCM').astype(float), d['age'] - d['age'].mean(), (d['center'] == 'Center2').astype(float)])
    Y = gap[sel].reshape(sel.sum(), -1)
    ok = np.isfinite(Y).all(0)
    B = np.full(Y.shape[1], np.nan, np.float32)
    B[ok] = np.linalg.lstsq(X, Y[:, ok], rcond=None)[0][1]
    return B.reshape(gap.shape[1:])


np.savez_compressed(ROOT/'slice_profiles.npz', img_id=S['img_id'].values, prof=np.nanmean(np.where(use[None], gap, np.nan), axis=(2, 3)),
                    alpha=alpha, beta=beta, anat=anat, use=use,
                    dcm_map=np.nanmean(gap[(S['group'] == 'DCM').values], 0), dcm_beta=dcm_vox_beta())
REF_AGE = 45
mid = M.shape[3]//2
fig, ax = plt.subplots(1, 4, figsize=(180*MM, 85*MM), layout='constrained', gridspec_kw=dict(width_ratios=[1, 1, 1, 1.4]))
show = lambda A: np.where(anat[:, :, mid], A[:, :, mid], np.nan)
im = ax[0].imshow(show(alpha + beta*REF_AGE), cmap='RdBu_r', vmin=REF_AGE-15, vmax=REF_AGE+15, aspect=0.623/0.6)
plt.colorbar(im, ax=ax[0], fraction=.05, label=f'Estimated age at {REF_AGE} y'); ax[0].set_title('Normative estimate'); panel(ax[0], 'a')
im = ax[1].imshow(show(beta), cmap='viridis', vmin=0, vmax=1, aspect=0.623/0.6)
plt.colorbar(im, ax=ax[1], fraction=.05, label='Slope (y per y)'); ax[1].set_title('Cross-sectional slope'); panel(ax[1], 'b')
dc = (S['group'] == 'DCM').values
im = ax[2].imshow(np.where(use[:, :, mid], np.nanmean(gap[dc][:, :, :, mid], 0), np.nan), cmap='RdBu_r', vmin=-10, vmax=10, aspect=0.623/0.6)
plt.colorbar(im, ax=ax[2], fraction=.05, label='Gap (y)'); ax[2].set_title('DCM, unadjusted'); panel(ax[2], 'c')
for a_ in ax[:3]:
    a_.set_xticks([])
    a_.set_yticks([ANCHORS[i]//2 for i in range(7)]); a_.set_yticklabels(VNAMES)
a_ = ax[3]
for g in ['HC', 'JHU_abnormal', 'DCM', 'Duke_abnormal', 'SG_compression']:
    k = (S['group'] == g).values & ~((S['role'] == 'dev').values & (g == 'HC'))
    prof = np.nanmean(np.where(use[None], gap[k], np.nan), axis=(2, 3))
    mu = np.nanmean(prof, 0); se = np.nanstd(prof, 0)/np.sqrt(np.isfinite(prof).sum(0))
    z = np.arange(len(mu))
    a_.plot(mu, z, color=GC[g], label=GL[g]); a_.fill_betweenx(z, mu-1.96*se, mu+1.96*se, color=GC[g], alpha=.15)
a_.invert_yaxis(); a_.axvline(0, color='0.5', ls='--', lw=.8)
a_.set_yticks([ANCHORS[i]//2 for i in range(7)]); a_.set_yticklabels(VNAMES)
a_.set(xlabel='Slice-averaged gap (y)'); a_.legend(fontsize=5.5, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, handlelength=1.2); panel(a_, 'd')
save(fig, 'Figure3_atlas')
print('anatomical voxels', int(anat.sum()), '| beta>0.3 fraction', float((beta[anat] > 0.3).mean()), '| beta median', float(np.median(beta[anat])))
json.dump(dict(anat_vox=int(anat.sum()), frac_beta_gt03=float((beta[anat] > 0.3).mean()), beta_median=float(np.median(beta[anat])),
               beta_p5_p95=np.percentile(beta[anat], [5, 95]).tolist()), open(TAB/'atlas_summary.json', 'w'))
for thr in [0.2, 0.3, 0.4]:
    u = anat & (beta > thr)
    gg = (M - (alpha + beta*age[:, None, None, None])) / np.where(u, beta, np.nan)
    pv = np.nanmean(gg.reshape(len(S), -1), 1)
    d = S.assign(vg=pv)
    for nm, c_, k_ in [('DCM', 'Center', 'DCM')]:
        dd = d[d['group'].isin(['HC', 'DCM']) & d['center'].isin(['Center1', 'Center2'])].assign(grp=lambda x: (x['group'] == 'DCM').astype(int))
        m = smf.ols('vg ~ grp + age + C(center)', dd).fit(cov_type='HC3')
        print(f'beta threshold {thr}: coverage {u.sum()/anat.sum():.2f} | DCM voxel-mean gap difference {m.params["grp"]:.2f} ({m.conf_int().loc["grp",0]:.2f}, {m.conf_int().loc["grp",1]:.2f})')
