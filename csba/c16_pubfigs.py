import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
from figstyle import *
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap
import nibabel as nib
import statsmodels.api as sm

OUT = ROOT/'logs'/'qc_figs'; OUT.mkdir(parents=True, exist_ok=True)
G = load_gaps(); P = G[G['role'] != 'retest'].copy(); P['cal'] = P['age'] + P['g_spine']
A = pd.read_csv(TAB/f'accuracy_{PRIMARY}.csv')
S = pd.read_csv(ROOT/'splits.csv')
S = S[[os.path.exists(NPZ/f'{i}.npz') for i in S['img_id']]].reset_index(drop=True)
SIDX = {i: n for n, i in enumerate(S['img_id'])}
CTX = {'HC': 'Healthy'}
YR = 'years'


def acc(nm, kind): return A[(A['set'] == nm) & (A['kind'] == kind)].iloc[0]


def forest(ax, rows, xlabel, ref=0, log=False, colors=None, ms=3.2):
    for i, r in enumerate(rows):
        c = (colors or {}).get(i, 'k')
        ax.plot([r['lo'], r['hi']], [i, i], color=c, lw=0.9, solid_capstyle='butt')
        ax.plot(r['beta'], i, marker=r.get('marker', 's'), color=c, mfc=c if r.get('filled', True) else 'white', ms=ms, mew=0.8)
    ax.axvline(ref, color='0.6', lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r['label'] for r in rows]); ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.tick_params(axis='y', length=0)
    if log: ax.set_xscale('log')
    ax.set_xlabel(xlabel)


# ---------------------------------------------------------------- Figure 1: design
def figure1():
    fig = plt.figure(figsize=(W2, 150*MM), layout='constrained')
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05])
    top = gs[0].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.05])
    bot = gs[1].subgridspec(1, 3, width_ratios=[1.25, 1, 1])

    # example: locked internal-test Spine Generic participant (open data)
    cand = S[(S['role'] == 'test') & (S['dataset'] == 'SpineGeneric') & (S['vox_max'] < 1.2)].sort_values('img_id')
    r = cand.iloc[(cand['age'] - 45).abs().argsort().iloc[0]]
    im = nib.as_closest_canonical(nib.load(r['img'])); sg = nib.as_closest_canonical(nib.load(r['seg_file']))
    I, L = np.asarray(im.dataobj, np.float32), np.asarray(sg.dataobj).astype(int)
    x = int(np.argmax((L == CORD).sum((1, 2))))
    zs = im.header.get_zooms()
    sl, ll = I[x].T, L[x].T
    rows_ = np.where((np.isin(ll, VERTS + DISCS + [CORD])).any(1))[0]
    z0, z1 = max(rows_.min() - 15, 0), min(rows_.max() + 15, sl.shape[0])
    cols_ = np.where((np.isin(ll, VERTS + DISCS + [CORD])).any(0))[0]
    c0, c1 = max(cols_.min() - 25, 0), min(cols_.max() + 25, sl.shape[1])
    sl, ll = sl[z0:z1, c0:c1], ll[z0:z1, c0:c1]
    vmax = np.percentile(sl, 99.5)
    d = np.load(NPZ/f"{r['img_id']}.npz"); st, sc = d['image'], d['compartments']
    k = SIDX[r['img_id']]
    mp = np.mean([np.load(ROOT/'preds'/f'main_f{f}_maps.npy', mmap_mode='r')[k] for f in range(5)], 0)
    mid, midm = st.shape[2]//2, mp.shape[2]//2
    asp = 0.623/0.6
    titles = ['Native T2-weighted', 'Segmentation', 'Straightened, level-normalized', 'Voxel-wise estimate', 'Calibrated compartment ages']
    axs = [fig.add_subplot(top[i]) for i in range(5)]
    axs[0].imshow(sl, cmap='gray', origin='lower', vmax=vmax, aspect=zs[2]/zs[1])
    lab = np.zeros(ll.shape); lab[np.isin(ll, VERTS)] = 1; lab[np.isin(ll, DISCS)] = 2; lab[ll == CORD] = 3; lab[ll == CSF] = 4
    cm = ListedColormap([OI['sky'], OI['orange'], OI['verm'], OI['green']])
    axs[1].imshow(sl, cmap='gray', origin='lower', vmax=vmax, aspect=zs[2]/zs[1])
    axs[1].imshow(np.ma.masked_where(lab == 0, lab), cmap=cm, vmin=1, vmax=4, origin='lower', alpha=.55, aspect=zs[2]/zs[1], interpolation='nearest')
    axs[2].imshow(st[:, :, mid], cmap='gray', aspect=asp, vmin=np.percentile(st, 1), vmax=np.percentile(st, 99.5))
    for a_, n_ in zip(ANCHORS, VNAMES):
        axs[2].axhline(a_, color=OI['sky'], lw=0.5, alpha=.9); axs[2].text(st.shape[1] + 2, a_, n_, va='center', fontsize=5.5, clip_on=False)
    msk = np.isfinite(mp[:, :, midm]) & (sc[::2, ::2, ::2][:, :, midm] > 0) if sc.shape[0] == 2*mp.shape[0] else np.isfinite(mp[:, :, midm])
    mm = np.where(msk, mp[:, :, midm], np.nan); lo_, hi_ = np.nanpercentile(mm, [2, 98])
    h = axs[3].imshow(mm, cmap='viridis', vmin=lo_, vmax=hi_, aspect=asp)
    cb = fig.colorbar(h, ax=axs[3], fraction=.06, pad=.02, shrink=.6); cb.set_label('Voxel estimate (years)', fontsize=6); cb.ax.tick_params(labelsize=5.5)
    for a_ in axs[:4]: a_.set_xticks([]); a_.set_yticks([]); [s_.set_visible(False) for s_ in a_.spines.values()]
    for a_, t in zip(axs, titles): a_.set_title(t, fontsize=6.5)
    a_ = axs[4]
    gr = G[G['img_id'] == r['img_id']].iloc[0]
    items = [('Whole spine', 'g_spine', OI['black']), ('Cord', 'g_cord', OI['verm']), ('CSF', 'g_csf', OI['green']), ('Discs', 'g_disc', OI['orange']), ('Vertebral bone', 'g_bone', OI['sky'])]
    for i, (nm, c, col) in enumerate(items):
        if c in gr and pd.notna(gr[c]): a_.plot(gr['age'] + gr[c], i, 's' if i == 0 else 'o', color=col, ms=4)
    a_.set_xlim(gr['age'] - 15, gr['age'] + 15)
    a_.axvline(gr['age'], color='0.5', lw=.6, ls=(0, (3, 2)))
    a_.set_yticks(range(len(items))); a_.set_yticklabels([i[0] for i in items]); a_.set_ylim(len(items) - .5, -.5)
    a_.set_xlabel('Estimated age (years)'); a_.tick_params(axis='y', length=0)
    a_.text(gr['age'] + .5, len(items) - .55, 'chronological age', fontsize=5.5, ha='left', va='bottom', color='0.4')
    fig.text(0.004, 0.995, 'a', fontsize=9, fontweight='bold', va='top')
    for i in range(4):
        if i < 3: fig.add_artist(FancyArrowPatch((0, 0), (1, 1), transform=fig.transFigure, alpha=0))

    # b: cohort composition
    a_ = fig.add_subplot(bot[0])
    src = [('CSpineSeg (Center 3)', P['dataset'] == 'Duke'), ('Spine Generic (43 sites)', P['dataset'] == 'SpineGeneric'),
           ('OpenNeuro', P['dataset'] == 'OpenNeuro'), ('Center 1, clinical', P['dataset'] == 'JHU_clinical'),
           ('Center 1, DCM study', P['dataset'] == 'JHU_DCM'), ('Center 2, DCM study', P['dataset'] == 'MCW')]
    cats = [('Development', lambda d: d['role'] == 'dev', OI['blue']), ('Internal test', lambda d: d['role'] == 'test', OI['sky']),
            ('External healthy', lambda d: (d['role'] == 'apply') & (d['group'] == 'HC'), OI['green']),
            ('Abnormal / compression', lambda d: d['group'].isin(['JHU_abnormal', 'Duke_abnormal', 'SG_compression', 'ON_stenosis']), OI['orange']),
            ('DCM', lambda d: d['group'].isin(['DCM', 'SG_DCM']), OI['verm']), ('Prior surgery', lambda d: d['group'] == 'JHU_priorsurg', OI['grey'])]
    for i, (nm, m) in enumerate(src):
        d = P[m]; left = 0
        for cn, f, col in cats:
            v = int(f(d).sum())
            if v: a_.barh(i, v, left=left, color=col, height=.62, edgecolor='white', lw=.4); left += v
        a_.text(left + 8, i, str(len(d)), va='center', fontsize=6, color='0.25')
    a_.set_yticks(range(len(src))); a_.set_yticklabels([s_[0] for s_ in src]); a_.invert_yaxis(); a_.tick_params(axis='y', length=0)
    a_.set_xlabel('Participants'); a_.set_xlim(0, 980)
    from matplotlib.patches import Patch
    a_.legend(handles=[Patch(color=c, label=n) for n, _, c in cats], loc='lower right', fontsize=5.8, handlelength=1, ncol=1)
    panel(a_, 'b', x=-0.42)

    # c: level alignment
    a_ = fig.add_subplot(bot[1])
    rng = np.random.default_rng(SEED)
    ids = rng.choice(S.loc[S['role'] == 'dev', 'img_id'].values, 300, replace=False)
    occ = np.zeros((len(VERTS), 192))
    for i in ids:
        c = np.load(NPZ/f'{i}.npz')['compartments']
        for j, v in enumerate(VERTS): occ[j] += (c == v).sum((1, 2)) > 0
    occ /= len(ids)
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(VERTS)))
    z = np.arange(192)*0.623
    for j, n_ in enumerate(VNAMES):
        a_.fill_betweenx(z, 0, occ[j], color=cols[j], alpha=.55, lw=0); a_.plot(occ[j], z, color=cols[j], lw=.6)
        a_.axhline(ANCHORS[j]*0.623, color='0.3', lw=.4, ls=(0, (2, 2)), zorder=3)
        a_.text(1.03, ANCHORS[j]*0.623, n_, va='center', fontsize=5.8, transform=a_.get_yaxis_transform())
    a_.set_ylim(z.max(), 0); a_.set_xlim(0, 1)
    a_.set_xlabel('Fraction of participants with level present'); a_.set_ylabel('Position along canal (mm)')
    a_.set_title('Vertebral level alignment', fontsize=7)
    panel(a_, 'c')

    # d: age distributions
    a_ = fig.add_subplot(bot[2])
    grps = [('Development', P[P['role'] == 'dev'], OI['blue']), ('Internal test', P[P['role'] == 'test'], OI['sky']),
            ('External healthy', P[(P['role'] == 'apply') & (P['group'] == 'HC')], OI['green']),
            ('Abnormal / compression', P[P['group'].isin(['JHU_abnormal', 'Duke_abnormal', 'SG_compression'])], OI['orange']),
            ('DCM', P[P['group'] == 'DCM'], OI['verm']), ('Prior surgery', P[P['group'] == 'JHU_priorsurg'], OI['grey'])]
    rg = np.random.default_rng(3)
    for i, (nm, d, col) in enumerate(grps):
        a_.scatter(d['age'], i + rg.uniform(-.22, .22, len(d)), s=1.5, color=col, alpha=.35, lw=0)
        q = np.percentile(d['age'], [25, 50, 75])
        a_.plot([q[0], q[2]], [i, i], color='k', lw=2.2, solid_capstyle='butt'); a_.plot(q[1], i, '|', color='white', ms=5, mew=1.2)
    a_.set_yticks(range(len(grps))); a_.set_yticklabels([g_[0] for g_ in grps]); a_.invert_yaxis(); a_.tick_params(axis='y', length=0)
    a_.set_xlabel('Chronological age (years)'); a_.set_xlim(15, 92)
    a_.set_title('Age distribution', fontsize=7)
    panel(a_, 'd', x=-0.5)
    save(fig, OUT/'Figure1')


# ---------------------------------------------------------------- Figure 2: performance
def figure2():
    fig = plt.figure(figsize=(W2, 118*MM), layout='constrained')
    gs = fig.add_gridspec(2, 3)
    sets = [('Internal test', P[P['role'] == 'test'], OI['blue'], 'o'),
            ('External Center 1 controls', P[(P['group'] == 'HC') & (P['center'] == 'Center1')], OI['green'], 's'),
            ('External Center 2 controls', P[(P['group'] == 'HC') & (P['center'] == 'Center2')], OI['orange'], 'D')]
    for i, (nm, d, col, mk) in enumerate(sets):
        a = fig.add_subplot(gs[0, i])
        a.plot([15, 92], [15, 92], color='0.6', lw=.6, ls=(0, (3, 2)), zorder=0)
        a.scatter(d['age'], d['raw_spine'], s=9, color=col, alpha=.75, lw=0, marker=mk)
        b = np.polyfit(d['age'], d['raw_spine'], 1); xs = np.array([d['age'].min(), d['age'].max()])
        a.plot(xs, np.polyval(b, xs), color='k', lw=.9)
        rr, rc = acc(nm, 'raw'), acc(nm, 'calibrated')
        a.text(.04, .97, f"n = {int(rr['n'])}\nMAE {rr['mae']:.1f} years ({rr['mae_lo']:.1f}–{rr['mae_hi']:.1f})\nr = {rr['r']:.2f}",
               transform=a.transAxes, va='top', fontsize=6.3, linespacing=1.35)
        a.set(xlim=(15, 92), ylim=(15, 92), xlabel='Chronological age (years)', ylabel='Estimated spine age (years)' if i == 0 else '')
        a.set_xticks([20, 40, 60, 80]); a.set_yticks([20, 40, 60, 80]); a.set_aspect('equal')
        a.set_title(nm.replace(' controls', ', healthy').replace('External ', 'External, '))
        panel(a, 'abc'[i])
    a = fig.add_subplot(gs[1, 0])
    for nm, d, col, mk in [('Cross-validated development', P[P['role'] == 'dev'], OI['blue'], 'o'), ('Internal test', P[P['role'] == 'test'], OI['sky'], 'o')]:
        a.scatter(d['age'], d['g_spine'], s=4, color=col, alpha=.4, lw=0, label=nm.replace('Cross-validated development', 'Development (out-of-fold)'))
        b = np.polyfit(d['age'], d['g_spine'], 1); xs = np.linspace(d['age'].min(), d['age'].max(), 2)
        a.plot(xs, np.polyval(b, xs), color=col if col != OI['sky'] else '#2a7fb8', lw=1.1)
    a.axhline(0, color='0.6', lw=.6, ls=(0, (3, 2)))
    a.set(xlabel='Chronological age (years)', ylabel='Calibrated age gap (years)', ylim=(-25, 25)); a.legend(loc='lower left', fontsize=6, markerscale=2)
    a.set_title('Gap independent of age'); panel(a, 'd')
    a = fig.add_subplot(gs[1, 1])
    R = pd.read_csv(TAB/'regional_performance.csv').set_index('region').loc[['cord', 'csf', 'disc', 'bone'] + VNAMES].reset_index()
    lab = {'cord': 'Cord', 'csf': 'CSF', 'disc': 'Discs', 'bone': 'Vertebral bone'}
    for i, rr in R.iterrows():
        tissue = rr['region'] in lab
        c = OI['verm'] if tissue else '0.25'
        a.plot([rr['r_lo'], rr['r_hi']], [i, i], color=c, lw=.9); a.plot(rr['r'], i, 'o', color=c, ms=3.2)
    a.set_yticks(range(len(R))); a.set_yticklabels([lab.get(x, x) for x in R['region']]); a.set_ylim(len(R) - .5, -.5); a.tick_params(axis='y', length=0)
    a.axhline(3.5, color='0.85', lw=.5)
    a.set(xlabel='Correlation with age, internal test (r)', xlim=(0.75, 0.97)); a.set_title('Regional estimates'); panel(a, 'e')
    a = fig.add_subplot(gs[1, 2])
    B = pd.read_csv(TAB/'ablation_summary.csv')
    short = {'Full model': 'Full model', 'No precision weighting': 'Unweighted pooling', 'Global-age supervision only': 'Global target only',
             'No level normalization': 'No level normalization', 'No marrow/CSF intensity anchor': 'No intensity anchor', 'Marrow removed from input': 'Marrow masked'}
    y = np.arange(len(B))
    a.barh(y, B['test_mae'], color=[OI['blue']] + ['0.65']*(len(B)-1), height=.6)
    a.errorbar(B['test_mae'], y, xerr=[B['test_mae'] - B['test_mae_lo'], B['test_mae_hi'] - B['test_mae']], fmt='none', ecolor='k', elinewidth=.6, capsize=0)
    for yy, v in zip(y, B['test_mae']): a.text(v + .25 + (B['test_mae_hi'] - B['test_mae']).iloc[yy], yy, f'{v:.1f}', va='center', fontsize=6)
    a.set_yticks(y); a.set_yticklabels([short.get(x, x) for x in B['label']]); a.invert_yaxis(); a.tick_params(axis='y', length=0)
    a.set(xlabel='MAE, internal test (years)', xlim=(0, 14.5)); a.set_title('Ablation'); panel(a, 'f')
    save(fig, OUT/'Figure2')


# ---------------------------------------------------------------- Figure 3: atlas
def figure3():
    Z = np.load(ROOT/'slice_profiles.npz', allow_pickle=True)
    alpha, beta, anat, use = Z['alpha'], Z['beta'], Z['anat'], Z['use']
    prof = pd.DataFrame(Z['prof'], index=Z['img_id'])
    fig = plt.figure(figsize=(W2, 92*MM), layout='constrained')
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.6])
    mid = alpha.shape[2]//2; asp = 0.623/0.6
    show = lambda X, m: np.where(m[:, :, mid], X[:, :, mid], np.nan)
    specs = [(alpha + beta*45, anat, 'RdBu_r', (30, 60), 'Normative age at 45 years', 'Estimated age (years)'),
             (beta, anat, 'viridis', (0, 1), 'Slope with chronological age', 'Years per year'),
             (Z['dcm_beta'], use, 'RdBu_r', (-10, 10), 'DCM vs local controls', 'Adjusted difference (years)')]
    for i, (X, m, cmap, (lo, hi), t, cl) in enumerate(specs):
        a = fig.add_subplot(gs[i])
        h = a.imshow(show(X, m), cmap=cmap, vmin=lo, vmax=hi, aspect=asp, interpolation='nearest')
        a.set_xticks([]); a.set_yticks([ANCHORS[j]//2 for j in range(7)]); a.set_yticklabels(VNAMES); a.tick_params(axis='y', length=0)
        [s_.set_visible(False) for s_ in a.spines.values()]
        cb = fig.colorbar(h, ax=a, orientation='horizontal', fraction=.05, pad=.02, aspect=14); cb.set_label(cl, fontsize=6); cb.ax.tick_params(labelsize=5.5)
        a.set_title(t); panel(a, 'abc'[i], x=-0.25)
    a = fig.add_subplot(gs[3])
    CON = {'Center 1 abnormal vs controls': (P[(P['group'] == 'HC') & (P['dataset'] == 'JHU_clinical')], P[P['group'] == 'JHU_abnormal'], ''),
           'DCM vs local controls (Centers 1-2)': (P[(P['group'] == 'HC') & P['center'].isin(['Center1', 'Center2'])], P[P['group'] == 'DCM'], 'center'),
           'Duke abnormal vs Duke controls': (P[(P['group'] == 'HC') & (P['dataset'] == 'Duke')], P[P['group'] == 'Duke_abnormal'], ''),
           'Spine Generic compression vs controls': (P[(P['group'] == 'HC') & (P['dataset'] == 'SpineGeneric')], P[P['group'] == 'SG_compression'], 'site')}
    zz = np.arange(prof.shape[1])
    out = []
    for nm, (c, k, adj) in CON.items():
        d = pd.concat([c.assign(grp=0), k.assign(grp=1)])
        d = d[d['img_id'].isin(prof.index)]
        X = np.column_stack([np.ones(len(d)), d['grp'], d['age'] - d['age'].mean()] +
                            ([pd.get_dummies(d[adj], drop_first=True).astype(float).values] if adj else []))
        Y = prof.loc[d['img_id']].values
        b, lo_, hi_ = [], [], []
        for j in zz:
            y = Y[:, j]; ok = np.isfinite(y)
            if ok.sum() < 30: b.append(np.nan); lo_.append(np.nan); hi_.append(np.nan); continue
            m = sm.OLS(y[ok], X[ok]).fit(cov_type='HC3'); ci = m.conf_int()[1]
            b.append(m.params[1]); lo_.append(ci[0]); hi_.append(ci[1])
        col = CONTRAST_COLOR[nm]
        a.plot(b, zz, color=col, lw=1.1, label=CONTRAST_SHORT[nm]); a.fill_betweenx(zz, lo_, hi_, color=col, alpha=.10, lw=0)
        out.append(pd.DataFrame(dict(contrast=nm, slice=zz, beta=b, lo=lo_, hi=hi_)))
    pd.concat(out).to_csv(TAB/'slice_profile_contrasts.csv', index=False)
    a.axvline(0, color='0.6', lw=.6, ls=(0, (3, 2)))
    a.set_yticks([ANCHORS[j]//2 for j in range(7)]); a.set_yticklabels(VNAMES); a.set_ylim(zz.max(), 0); a.tick_params(axis='y', length=0)
    a.set_xlabel('Adjusted difference in slice gap (years)'); a.set_title('Level-resolved group differences')
    a.legend(loc='lower center', bbox_to_anchor=(0.5, 1.07), ncol=2, fontsize=6, handlelength=1.4, columnspacing=1)
    panel(a, 'd', x=-0.16)
    save(fig, OUT/'Figure3')


# ---------------------------------------------------------------- Figure 4: group differences
def figure4():
    C = pd.read_csv(TAB/f'contrasts_{PRIMARY}.csv'); SS = pd.read_csv(TAB/'contrast_sensitivity.csv')
    fig = plt.figure(figsize=(W2, 120*MM), layout='constrained')
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], height_ratios=[1.15, 1])
    a = fig.add_subplot(gs[:, 0])
    rows, colors = [], {}
    for cn in C['contrast'].unique():
        d = C[C['contrast'] == cn].set_index('marker')
        for mk in ['Whole spine', 'Cord', 'CSF', 'Disc', 'Vertebral bone']:
            r = d.loc[mk]; prim = mk == 'Whole spine'
            p = r['p_holm_primary'] if prim else r['p_holm_compartments']
            rows.append(dict(beta=r['beta'], lo=r['lo'], hi=r['hi'], label=(CONTRAST_SHORT[cn] if prim else '   ' + mk),
                             marker='s' if prim else 'o', filled=p < .05, p=p))
            colors[len(rows) - 1] = CONTRAST_COLOR[cn] if prim else '0.45'
        rows.append(dict(beta=np.nan, lo=np.nan, hi=np.nan, label=''));
    rows = rows[:-1]
    forest(a, rows, 'Adjusted difference in age gap (years)', colors=colors)
    for i, r in enumerate(rows):
        if r['label'] and r['marker'] == 's':
            a.get_yticklabels()[i].set_fontweight('bold')
            a.text(1.0, i, fmt_p(r['p']), transform=a.get_yaxis_transform(), fontsize=5.8, va='center', ha='left')
    a.set_title('Pre-specified contrasts (filled: Holm P < .05)'); panel(a, 'a', x=-0.5)

    a = fig.add_subplot(gs[0, 1])
    nm = 'DCM vs local controls (Centers 1-2)'; d = SS[SS['contrast'] == nm].set_index('analysis')
    rr = [('Pooled', d.loc['primary']), ('Center 1', d.loc['within Center1']), ('Center 2', d.loc['within Center2']),
          ('Age-matched pairs', d.loc['age-matched pairs (within center, no replacement, caliper 3 y)']),
          ('All healthy controls', d.loc['secondary: all healthy controls, + C(source)'])]
    rows = [dict(beta=v['beta'], lo=v['lo'], hi=v['hi'], label=l, filled=v['p'] < .05, marker='s' if l == 'Pooled' else 'o') for l, v in rr]
    forest(a, rows, 'DCM vs controls (years)', colors={i: OI['verm'] for i in range(len(rows))})
    gi = d.loc['group x center interaction']
    a.text(.98, .03, f"Center interaction {fmt_p(gi['p'])}", transform=a.transAxes, ha='right', fontsize=5.8)
    a.set_title('DCM effect by center'); panel(a, 'b', x=-0.55)

    a = fig.add_subplot(gs[0, 2])
    J = P[P['dataset'] == 'JHU_clinical'].merge(pd.read_csv(ROOT/'jhu_clinical.csv').assign(ID_s=lambda d: d['mrn'].astype(str))[['ID_s', 'n_spinalstenosis']], on='ID_s')
    J = J[J['group'].isin(['HC', 'JHU_abnormal'])].copy()
    J['bin'] = pd.cut(J['n_spinalstenosis'], [-0.1, 0.25, 1.5, 3.5, 8], labels=['0', '1', '2–3', '≥4'])
    J['res'] = smf.ols('g_spine ~ age', J).fit().resid
    s = J.groupby('bin', observed=False)['res'].agg(['mean', 'sem', 'size'])
    a.scatter(J['bin'].cat.codes + np.random.default_rng(1).uniform(-.15, .15, len(J)), J['res'], s=3, color=OI['orange'], alpha=.35, lw=0)
    a.errorbar(range(len(s)), s['mean'], yerr=1.96*s['sem'], fmt='o', color='k', ms=3.5, capsize=2, lw=.9)
    Rr = pd.read_csv(TAB/'reader_reference.csv'); rr = Rr[(Rr['predictor'] == 'n_spinalstenosis') & (Rr['sample'] == 'controls + abnormal')].iloc[0]
    a.text(.03, .97, f"{rr['beta']:+.1f} years per level\n{fmt_p(rr['p'])}", transform=a.transAxes, va='top', fontsize=6)
    a.set_xticks(range(len(s))); a.set_xticklabels(s.index.astype(str)); a.axhline(0, color='0.6', lw=.6, ls=(0, (3, 2)))
    a.set(xlabel='Levels with canal stenosis (readers)', ylabel='Age-adjusted gap (years)', ylim=(-22, 28))
    a.set_title('Dose–response, Center 1'); panel(a, 'c', x=-0.3)

    a = fig.add_subplot(gs[1, 1:])
    lab = {'any cord T2 hyperintensity': 'Cord T2 hyperintensity', 'sx_gaitabnormal': 'Gait symptoms', 'exam_gaitabnormal': 'Gait abnormality (exam)',
           'sx_motor': 'Motor symptoms', 'exam_motordeficit': 'Motor deficit (exam)', 'exam_umnsigns': 'Upper motor neuron signs',
           'sx_neckpain': 'Neck pain'}
    R1 = Rr[(Rr['sample'] == 'controls + abnormal') & Rr['predictor'].isin(lab)].copy()
    R1['order'] = R1['predictor'].map({k: i for i, k in enumerate(lab)}); R1 = R1.sort_values('order')
    rows = [dict(beta=v['beta'], lo=v['lo'], hi=v['hi'], label=lab[v['predictor']], filled=v['p'] < .05) for _, v in R1.iterrows()]
    forest(a, rows, 'Difference in age gap, present vs absent (years)', colors={i: OI['orange'] for i in range(len(rows))})
    for i, (_, v) in enumerate(R1.iterrows()): a.text(1.0, i, fmt_p(v['p']), transform=a.get_yaxis_transform(), fontsize=5.8, va='center')
    a.set_title('Reader findings and clinical features, Center 1'); panel(a, 'd', x=-0.3)
    save(fig, OUT/'Figure4')


# ---------------------------------------------------------------- Figure 5: focal and clinical
def figure5():
    FO = pd.read_csv(TAB/'focal_levels.csv'); CL = pd.read_csv(TAB/'clinical_endpoints.csv')
    D = pd.read_csv(ROOT/'dcm_clinical_frame.csv')
    fig = plt.figure(figsize=(W2, 66*MM), layout='constrained')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.25])
    a = fig.add_subplot(gs[0])
    lab = {'DCM: level of maximum compression': 'DCM: maximal compression',
           'Center 1 abnormal: reader spinal stenosis levels': 'Center 1: canal stenosis',
           'Center 1 abnormal: reader foraminal stenosis levels': 'Center 1: foraminal stenosis',
           'Spine Generic compression: reported levels': 'Asymptomatic compression'}
    col = {'DCM: level of maximum compression': OI['verm'], 'Center 1 abnormal: reader spinal stenosis levels': OI['orange'],
           'Center 1 abnormal: reader foraminal stenosis levels': OI['orange'], 'Spine Generic compression: reported levels': OI['green']}
    F = FO.set_index('analysis').loc[list(lab)].reset_index()
    rows = [dict(beta=v['beta'], lo=v['lo'], hi=v['hi'], label=lab[v['analysis']], filled=v['p'] < .05) for _, v in F.iterrows()]
    forest(a, rows, 'Affected vs other levels, same person (years)', colors={i: col[v] for i, v in enumerate(F['analysis'])})
    for i, (_, v) in enumerate(F.iterrows()): a.text(1.0, i, fmt_p(v['p']), transform=a.get_yaxis_transform(), fontsize=5.8, va='center')
    a.set_title('Within-person focal excess'); panel(a, 'a', x=-0.62)

    a = fig.add_subplot(gs[1])
    d = D.dropna(subset=['mJOA', 'g_spine', 'age', 'sex']).copy()
    rx = smf.ols('g_spine ~ age + C(sex) + C(center)', d).fit().resid + d['g_spine'].mean()
    ry = smf.ols('mJOA ~ age + C(sex) + C(center)', d).fit().resid + d['mJOA'].mean()
    for ce, mk, c in [('Center1', 'o', OI['blue']), ('Center2', 'D', OI['orange'])]:
        k = d['center'] == ce
        a.scatter(rx[k], ry[k] + np.random.default_rng(2).uniform(-.12, .12, k.sum()), s=8, marker=mk, color=c, alpha=.7, lw=0, label=ce.replace('Center', 'Center '))
    b = np.polyfit(rx, ry, 1); xs = np.linspace(rx.min(), rx.max(), 2); a.plot(xs, np.polyval(b, xs), color='k', lw=1)
    r0 = CL[CL['endpoint'] == 'Baseline mJOA'].iloc[0]
    a.set_title(f"Baseline function, DCM\n{r0['beta']:.2f} points per SD, {fmt_p(r0['p'])}", fontsize=7)
    a.set(xlabel='Adjusted age gap (years)', ylabel='Adjusted baseline mJOA')
    a.legend(loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2, fontsize=6, handletextpad=.2)
    panel(a, 'b', x=-0.25)

    a = fig.add_subplot(gs[2])
    lab = {'Baseline mJOA': 'Baseline mJOA (points)', 'Baseline mJOA, + mFI-5': '  + frailty (mFI-5)',
           'Change in mJOA (3 months)': 'ΔmJOA at 3 months (points)', 'Change in mJOA, + mFI-5': '  + frailty (mFI-5)',
           'Frailty (mFI-5)': 'Frailty index (mFI-5)', 'MSCC (AP diameter)': 'MSCC (%)'}
    C2 = CL.set_index('endpoint').loc[[k for k in lab if k in set(CL['endpoint'])]].reset_index()
    rows = [dict(beta=v['beta'], lo=v['lo'], hi=v['hi'], label=lab[v['endpoint']], filled=v['p'] < .05) for _, v in C2.iterrows()]
    forest(a, rows, 'Effect per 1 SD of age gap', colors={i: OI['verm'] for i in range(len(rows))})
    for i, (_, v) in enumerate(C2.iterrows()): a.text(1.0, i, fmt_p(v['p']), transform=a.get_yaxis_transform(), fontsize=5.8, va='center')
    a.set_title('Clinical measures, DCM'); panel(a, 'c', x=-0.62)
    save(fig, OUT/'Figure5')


# ---------------------------------------------------------------- Supplementary figures
def figS_reliability():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 62*MM), layout='constrained', gridspec_kw=dict(width_ratios=[.8, 1.1, 1]))
    RL = pd.read_csv(TAB/'reliability.csv')
    a = ax[0]
    UN = pd.read_csv(TAB/'uncertainty.csv') if (TAB/'uncertainty.csv').exists() else None
    PO = pd.read_csv(TAB/'posture.csv')
    rows = [dict(beta=v['icc_2_1'], lo=v['icc_2_1'], hi=v['icc_2_1'], label=v['set'][:38]) for _, v in RL.iterrows()]
    for i, v in enumerate(RL.itertuples()):
        a.barh(i, v.icc_2_1, color=OI['blue'] if v.icc_2_1 > .8 else '0.6', height=.6); a.text(v.icc_2_1 + .01, i, f'{v.icc_2_1:.2f}', va='center', fontsize=6)
    a.set_yticks(range(len(RL))); a.set_yticklabels(['All repeat series', 'Same-contrast repeats']); a.invert_yaxis(); a.set_xlim(0, 1.1)
    a.set_xlabel('ICC(2,1)'); a.set_title('Repeat-series agreement'); panel(a, 'a', x=-0.55)
    AC = pd.read_csv(TAB/'acquisition.csv')
    a = ax[1]
    albl = ['OpenNeuro vs CSpineSeg', 'Spine Generic vs CSpineSeg', '3D vs 2D (all healthy)', 'Philips vs GE (Spine Generic)',
            'Siemens vs GE (Spine Generic)', 'Slice thickness, per mm (CSpineSeg)', '3 T vs 1.5 T (Center 1 DCM)']
    rows = [dict(beta=v['beta'], lo=v['lo'], hi=v['hi'], label=albl[i], filled=v['p'] < .05) for i, (_, v) in enumerate(AC.iterrows())]
    forest(a, rows, 'Difference in age gap (years)', colors={i: '0.3' for i in range(len(rows))})
    a.set_title('Acquisition and source'); panel(a, 'b', x=-0.8)
    a = ax[2]
    Ab = pd.read_csv(TAB/'ablation_summary.csv')
    for j, (cn, key) in enumerate([('C1 abnormal', 'C1 abnormal'), ('DCM', 'DCM'), ('CSpineSeg abnormal', 'Duke abnormal'), ('Asympt. compression', 'SG compression')]):
        for i, v in Ab.iterrows():
            yy = j*(len(Ab)+1) + i
            a.plot([v[f'{key} lo'], v[f'{key} hi']], [yy, yy], color=OI['blue'] if i == 0 else '0.55', lw=.8)
            a.plot(v[f'{key} beta'], yy, 's' if i == 0 else 'o', color=OI['blue'] if i == 0 else '0.55', ms=2.6)
    a.set_yticks([j*(len(Ab)+1) + (len(Ab)-1)/2 for j in range(4)]); a.set_yticklabels(['C1 abnormal', 'DCM', 'CSpineSeg abn.', 'Asympt. compr.'])
    a.invert_yaxis(); a.axvline(0, color='0.6', lw=.6, ls=(0, (3, 2))); a.tick_params(axis='y', length=0)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], marker='s', color=OI['blue'], lw=.8, ms=3, label='Full model'), Line2D([], [], marker='o', color='0.55', lw=.8, ms=3, label='Ablations')],
             loc='lower right', fontsize=5.8)
    a.set_xlabel('Adjusted difference (years)'); a.set_title('Contrasts across ablations'); panel(a, 'c', x=-0.45)
    save(fig, OUT/'FigureS1_robustness')


def figS_proteomics():
    PR = pd.read_csv(TAB/'proteomics.csv'); ps = json.load(open(TAB/'proteomics_summary.json'))
    fig, ax = plt.subplots(1, 2, figsize=(W2*0.75, 62*MM), layout='constrained')
    a = ax[0]
    sig = PR['q'] < .05
    a.scatter(PR['rho'], -np.log10(PR['p']), s=7, color=np.where(sig, OI['verm'], '0.6'), lw=0)
    a.axhline(-np.log10(.05), color='0.6', lw=.6, ls=(0, (3, 2)))
    for j, (_, v) in enumerate(PR.head(6).iterrows()): a.annotate(v['protein'], (v['rho'], -np.log10(v['p'])), xytext=(5, 5 if j % 2 else -5), textcoords='offset points', fontsize=5.8, va='center')
    a.set(xlabel='Partial Spearman ρ with age gap', ylabel='−log$_{10}$ P'); a.set_title(f"{ps['n_nominal']} of {ps['n_proteins']} proteins at P < .05 (expected {ps['expected']:.1f})")
    panel(a, 'a')
    a = ax[1]
    top = PR.head(12).iloc[::-1]
    a.barh(range(len(top)), top['rho'], color=np.where(top['q'] < .05, OI['verm'], '0.6'), height=.6)
    a.set_yticks(range(len(top))); a.set_yticklabels(top['protein']); a.axvline(0, color='0.4', lw=.6); a.tick_params(axis='y', length=0)
    a.set_xlabel('Partial Spearman ρ'); a.set_title(f"Set-level permutation {fmt_p(ps['perm_p'])}"); panel(a, 'b', x=-0.35)
    save(fig, OUT/'FigureS2_proteomics')


if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', 'S']
    if '1' in which: figure1()
    if '2' in which: figure2()
    if '3' in which: figure3()
    if '4' in which: figure4()
    if '5' in which: figure5()
    if 'S' in which: figS_reliability(); figS_proteomics()
    print('saved to', OUT)
