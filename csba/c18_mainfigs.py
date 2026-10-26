import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c16_pubfigs import *
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'legend.fontsize': 6.5, 'axes.titlepad': 5,
                     'figure.constrained_layout.h_pad': 0.12, 'figure.constrained_layout.w_pad': 0.12,
                     'figure.constrained_layout.hspace': 0.06, 'figure.constrained_layout.wspace': 0.06})
OUT = ROOT/'manuscript'/'figures'; OUT.mkdir(parents=True, exist_ok=True)
NPZ_P = ROOT/('npz_n4' if PRIMARY == 'harm' else 'npz')
CI_C = '0.45'


def mn(s): return s.replace('-', '−') if s else s


def letter(ax, s, x=-0.16): ax.text(x, 1.06, s, transform=ax.transAxes, fontsize=7, fontweight='bold', va='bottom')


def dashed0(ax, v=0, axis='x'): (ax.axvline if axis == 'x' else ax.axhline)(v, color='0.55', lw=.7, ls=(0, (3, 2)), zorder=0)


def ci_point(ax, x, y, lo, hi, color, filled=True, marker='o', ms=4.5, horizontal=True):
    kw = dict(fmt='none', ecolor=CI_C, elinewidth=.9, capsize=2, capthick=.9, zorder=2)
    if horizontal: ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], **kw)
    else: ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], **kw)
    ax.plot(x, y, marker, mfc=color if filled else 'white', mec='k', mew=.6, ms=ms, zorder=3)


def fplot(ax, rows, xlabel, color, ptext=True):
    for i, r in enumerate(rows):
        ci_point(ax, r['beta'], i, r['lo'], r['hi'], r.get('color', color), r['p'] < .05, r.get('marker', 'o'), 5 if r.get('marker') == 's' else 4.5)
        if ptext: ax.text(1.02, i, fmt_p(r['p']), transform=ax.get_yaxis_transform(), va='center', fontsize=6.5, color='k')
    dashed0(ax)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r['label'] for r in rows]); ax.set_ylim(len(rows) - .5, -.6)
    ax.tick_params(axis='y', length=0); ax.set_xlabel(xlabel)


def boxstrip(ax, groups, seed=SEED):
    rng = np.random.default_rng(seed)
    bp = ax.boxplot([g[1] for g in groups], positions=range(len(groups)), widths=.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color='k', lw=1.1), boxprops=dict(facecolor='none', edgecolor='k', lw=.7),
                    whiskerprops=dict(color='k', lw=.7), capprops=dict(color='k', lw=.7), zorder=2)
    for i, (_, v, c) in enumerate(groups):
        v = np.asarray(v); ax.scatter(i + rng.uniform(-.2, .2, len(v)), v, s=3.5, color=c, alpha=.5, lw=0, zorder=1)
    ax.set_xticks(range(len(groups))); ax.set_xlim(-.6, len(groups) - .4)
    return bp


# ------------------------------------------------------------ Figure 1
def fig1():
    fig = plt.figure(figsize=(W2, 150*MM), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1.15, .85])
    sub = gs[0, 0].subgridspec(1, 3)
    cand = S[(S['role'] == 'test') & (S['dataset'] == 'SpineGeneric') & (S['vox_max'] < 1.2)]
    r = cand.iloc[(cand['age'] - 45).abs().argsort().iloc[0]]
    im = nib.as_closest_canonical(nib.load(r['img'])); sg = nib.as_closest_canonical(nib.load(r['seg_file']))
    I, L = np.asarray(im.dataobj, np.float32), np.asarray(sg.dataobj).astype(int)
    x = int(np.argmax((L == CORD).sum((1, 2)))); zs = im.header.get_zooms()
    sl, ll = I[x].T, L[x].T
    rr_ = np.where(np.isin(ll, VERTS + DISCS + [CORD]).any(1))[0]; cc_ = np.where(np.isin(ll, VERTS + DISCS + [CORD]).any(0))[0]
    sl = sl[max(rr_.min() - 12, 0):rr_.max() + 12, max(cc_.min() - 20, 0):cc_.max() + 20]
    ll = ll[max(rr_.min() - 12, 0):rr_.max() + 12, max(cc_.min() - 20, 0):cc_.max() + 20]
    d = np.load(NPZ_P/f"{r['img_id']}.npz"); st = d['image']
    mp = np.mean([np.load(ROOT/'preds'/f'{PRIMARY}_f{f}_maps.npy', mmap_mode='r')[SIDX[r['img_id']]] for f in range(5)], 0)
    sc = d['compartments'][::2, ::2, ::2]
    asp = 0.623/0.6
    ax = [fig.add_subplot(sub[i]) for i in range(3)]
    ax[0].imshow(sl, cmap='gray', origin='lower', vmax=np.percentile(sl, 99.5), aspect=zs[2]/zs[1])
    lab = np.zeros(ll.shape); lab[np.isin(ll, VERTS)] = 1; lab[np.isin(ll, DISCS)] = 2; lab[ll == CORD] = 3
    ax[0].imshow(np.ma.masked_where(lab == 0, lab), cmap=ListedColormap([OI['sky'], OI['orange'], OI['verm']]), vmin=1, vmax=3,
                 origin='lower', alpha=.45, aspect=zs[2]/zs[1], interpolation='nearest')
    ax[0].set_title('T2-weighted MRI\nwith segmentation')
    ax[1].imshow(st[:, :, st.shape[2]//2], cmap='gray', aspect=asp, vmin=np.percentile(st, 1), vmax=np.percentile(st, 99.5))
    for a_, n_ in zip(ANCHORS, VNAMES):
        ax[1].axhline(a_, color=OI['sky'], lw=.6); ax[1].text(st.shape[1] + 3, a_, n_, va='center', fontsize=6.5, clip_on=False)
    ax[1].set_title('Straightened,\nlevel-normalized')
    m = mp[:, :, mp.shape[2]//2]; m = np.where(sc[:, :, sc.shape[2]//2] > 0, m, np.nan); lo, hi = np.nanpercentile(m, [2, 98])
    h = ax[2].imshow(m, cmap='viridis', vmin=lo, vmax=hi, aspect=asp)
    cb = fig.colorbar(h, ax=ax[2], fraction=.07, pad=.03, shrink=.55); cb.set_label('Years', fontsize=7); cb.ax.tick_params(labelsize=6.5)
    ax[2].set_title('Voxel-wise\nage estimate')
    for a_ in ax: a_.set_xticks([]); a_.set_yticks([]); [s_.set_visible(False) for s_ in a_.spines.values()]
    fig.text(0.005, 0.995, 'a', fontsize=7, fontweight='bold', va='top')
    a = fig.add_subplot(gs[0, 1])
    src = [('CSpineSeg', P['dataset'] == 'Duke'), ('Spine Generic', P['dataset'] == 'SpineGeneric'), ('OpenNeuro', P['dataset'] == 'OpenNeuro'),
           ('Center 1', P['dataset'].isin(['JHU_clinical', 'JHU_DCM'])), ('Center 2', P['dataset'] == 'MCW')]
    cats = [('Development', lambda d: d['role'] == 'dev', OI['blue']), ('Internal test', lambda d: d['role'] == 'test', OI['sky']),
            ('External healthy', lambda d: (d['role'] == 'apply') & (d['group'] == 'HC'), OI['green']),
            ('Abnormal MRI', lambda d: d['group'].isin(['JHU_abnormal', 'Duke_abnormal', 'SG_compression', 'ON_stenosis', 'JHU_priorsurg']), OI['orange']),
            ('DCM', lambda d: d['group'].isin(['DCM', 'SG_DCM']), OI['verm'])]
    for i, (nm, msk) in enumerate(src):
        dd = P[msk]; left = 0
        for cn, f, col in cats:
            v = int(f(dd).sum())
            if v: a.barh(i, v, left=left, color=col, height=.6, edgecolor='white', lw=.5); left += v
        a.text(left + 10, i, str(len(dd)), va='center', fontsize=6.5)
    a.set_yticks(range(len(src))); a.set_yticklabels([s_[0] for s_ in src]); a.invert_yaxis(); a.tick_params(axis='y', length=0)
    a.set_xlabel('Participants'); a.set_xlim(0, 960); a.set_title('Datasets and analysis sets')
    a.legend(handles=[Patch(color=c, label=n) for n, _, c in cats], loc='lower right', handlelength=1, frameon=False)
    letter(a, 'b', x=-0.3)
    # c: level alignment after straightening (peak slice of each vertebra vs its target anchor)
    a = fig.add_subplot(gs[1, 0])
    LA = pd.read_csv(TAB/'level_alignment.csv')
    for k, (geo, lab, col, mk, dx) in enumerate([('npz_uniform', 'Uniform length scaling', '0.75', 's', -.12), ('npz', 'Level-normalized (used)', OI['blue'], 'o', .12)]):
        d = LA[LA['geometry'] == geo].reset_index(drop=True)
        for t, v in d.iterrows():
            m, sd = v['mean_dev']*0.623, v['sd']*0.623
            ci_point(a, t + dx, m, m - sd, m + sd, col, marker=mk, horizontal=False)
        a.plot([], [], mk, mfc=col, mec='k', mew=.6, ms=4.5, label=lab)
    dashed0(a, axis='y')
    a.set_xticks(range(len(VNAMES))); a.set_xticklabels(VNAMES); a.set_xlim(-.5, len(VNAMES) - .5)
    a.set(xlabel='Vertebral level', ylabel='Offset from target position (mm)', title='Vertebral level alignment (mean ± SD)')
    a.legend(loc='lower left'); letter(a, 'c', x=-0.1)
    # d: age distribution by analysis set
    a = fig.add_subplot(gs[1, 1])
    grps = [('Development', P[P['role'] == 'dev'], OI['blue']), ('Internal test', P[P['role'] == 'test'], OI['sky']),
            ('External healthy', P[(P['role'] == 'apply') & (P['group'] == 'HC')], OI['green']),
            ('Abnormal MRI', P[P['group'].isin(['JHU_abnormal', 'Duke_abnormal', 'SG_compression'])], OI['orange']), ('DCM', P[P['group'] == 'DCM'], OI['verm'])]
    bp = a.boxplot([g_[1]['age'] for g_ in grps], vert=False, widths=.55, patch_artist=True, showfliers=False,
                   medianprops=dict(color='k', lw=1.1), whiskerprops=dict(lw=.7), capprops=dict(lw=.7), boxprops=dict(lw=.7))
    for b_, g_ in zip(bp['boxes'], grps): b_.set_facecolor(g_[2]); b_.set_alpha(.8)
    a.set_yticks(range(1, len(grps) + 1)); a.set_yticklabels([g_[0] for g_ in grps]); a.invert_yaxis(); a.tick_params(axis='y', length=0)
    a.set(xlabel='Chronological age (years)', title='Age by analysis set'); letter(a, 'd', x=-0.3)
    save(fig, OUT/'Figure1')


# ------------------------------------------------------------ Figure 2
def fig2():
    fig, ax = plt.subplots(2, 2, figsize=(W2, 150*MM), layout='constrained'); ax = ax.ravel()
    a = ax[0]; d = P[P['role'] == 'test']; rr = acc('Internal test', 'raw')
    a.plot([15, 92], [15, 92], color='0.55', lw=.7, ls=(0, (3, 2)))
    a.scatter(d['age'], d['raw_spine'], s=8, color=OI['blue'], alpha=.65, lw=0)
    b = np.polyfit(d['age'], d['raw_spine'], 1); xs = np.array([18, 88]); a.plot(xs, np.polyval(b, xs), color='k', lw=1.2)
    a.text(.05, .95, f"$\\it{{n}}$ = {int(rr['n'])}\nMAE = {rr['mae']:.1f} years\n$\\it{{r}}$ = {rr['r']:.2f}", transform=a.transAxes, va='top', linespacing=1.5)
    a.set(xlim=(15, 92), ylim=(15, 92), xlabel='Chronological age (years)', ylabel='Estimated spine age (years)', title='Internal test')
    letter(a, 'a')
    a = ax[1]
    a.plot([15, 92], [15, 92], color='0.55', lw=.7, ls=(0, (3, 2)))
    txt = []
    for nm, ce, col, mk in [('Center 1', 'Center1', OI['green'], 's'), ('Center 2', 'Center2', OI['orange'], 'D')]:
        d = P[(P['group'] == 'HC') & (P['center'] == ce)]; rr = acc(f'External {ce.replace("Center", "Center ")} controls', 'raw')
        a.scatter(d['age'], d['raw_spine'], s=9, color=col, marker=mk, alpha=.8, lw=0, label=nm)
        txt.append(f"{nm}: $\\it{{n}}$ = {int(rr['n'])}, MAE = {rr['mae']:.1f} years, $\\it{{r}}$ = {rr['r']:.2f}")
    a.text(.05, .95, '\n'.join(txt), transform=a.transAxes, va='top', linespacing=1.5)
    a.legend(loc='lower right')
    a.set(xlim=(15, 92), ylim=(15, 92), xlabel='Chronological age (years)', ylabel='Estimated spine age (years)', title='External test (healthy)')
    letter(a, 'b')
    a = ax[2]
    for nm, d, col in [('Development (out-of-fold)', P[P['role'] == 'dev'], '0.6'), ('Internal test', P[P['role'] == 'test'], OI['blue'])]:
        a.scatter(d['age'], d['g_spine'], s=6, color=col, alpha=.6, lw=0)
        b = np.polyfit(d['age'], d['g_spine'], 1); xs = np.array([d['age'].min(), d['age'].max()])
        a.plot(xs, np.polyval(b, xs), color=col, lw=1.5, label=f'{nm}, slope ' + mn(f'{b[0]:+.3f}'))
    dashed0(a, axis='y')
    a.set(xlabel='Chronological age (years)', ylabel='Age gap (years)', ylim=(-25, 25), title='Calibrated age gap (healthy)')
    a.legend(loc='lower left'); letter(a, 'c')
    a = ax[3]
    R = pd.read_csv(TAB/'regional_performance.csv').set_index('region')
    A_ = acc('Internal test', 'raw')
    items = [('Whole spine', A_['r'], A_['r_lo'], A_['r_hi'])] + [(lab, R.loc[k, 'r'], R.loc[k, 'r_lo'], R.loc[k, 'r_hi']) for k, lab in
             [('cord', 'Cord'), ('csf', 'CSF'), ('disc', 'Intervertebral discs'), ('bone', 'Vertebral bone')]]
    for i, (lab, r_, lo, hi) in enumerate(items):
        ci_point(a, r_, i, lo, hi, OI['blue'], marker='s' if i == 0 else 'o')
    a.set_yticks(range(len(items))); a.set_yticklabels([i[0] for i in items]); a.set_ylim(len(items) - .5, -.5); a.tick_params(axis='y', length=0)
    a.set(xlim=(0.7, 1.0), xlabel='Correlation with chronological age ($\\it{r}$, 95% CI)', title='Age signal by tissue (internal test)')
    letter(a, 'd', x=-0.3)
    save(fig, OUT/'Figure2')


# ------------------------------------------------------------ Figure 3
def fig3():
    Z = np.load(ROOT/'slice_profiles.npz', allow_pickle=True)
    beta, anat, use = Z['beta'], Z['anat'], Z['use']
    fig = plt.figure(figsize=(W2, 100*MM), layout='constrained')
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    mid = beta.shape[2]//2; asp = 0.623/0.6
    for i, (X, m, cmap, (lo, hi), t, cl) in enumerate([(beta, anat, 'viridis', (0, 1), 'Aging rate, healthy', 'Years per year'),
                                                     (Z['dcm_beta'], use, 'RdBu_r', (-10, 10), 'DCM vs local controls', 'Adjusted difference (years)')]):
        a = fig.add_subplot(gs[i])
        h = a.imshow(np.where(m[:, :, mid], X[:, :, mid], np.nan), cmap=cmap, vmin=lo, vmax=hi, aspect=asp, interpolation='nearest')
        a.set_xticks([]); a.set_yticks([ANCHORS[j]//2 for j in range(7)]); a.set_yticklabels(VNAMES); a.tick_params(axis='y', length=0)
        [s_.set_visible(False) for s_ in a.spines.values()]
        cb = fig.colorbar(h, ax=a, orientation='horizontal', fraction=.045, pad=.03, aspect=16); cb.set_label(cl); cb.ax.tick_params(labelsize=7)
        a.set_title(t); letter(a, 'ab'[i], x=-0.2)
    a = fig.add_subplot(gs[2])
    SP = pd.read_csv(TAB/'slice_profile_contrasts.csv')
    for cn in ['DCM vs local controls (Centers 1-2)', 'Center 1 abnormal vs controls', 'Duke abnormal vs Duke controls']:
        d = SP[SP['contrast'] == cn]; col = CONTRAST_COLOR[cn]
        if cn.startswith('DCM'): a.fill_betweenx(d['slice'], d['lo'], d['hi'], color=col, alpha=.12, lw=0, label='95% CI (DCM)')
        a.plot(d['beta'], d['slice'], color=col, lw=1.6, label=CONTRAST_SHORT[cn].replace('DCM (Centers 1-2)', 'DCM'))
    dashed0(a)
    a.set_yticks([ANCHORS[j]//2 for j in range(7)]); a.set_yticklabels(VNAMES); a.set_ylim(SP['slice'].max(), 0); a.tick_params(axis='y', length=0)
    a.set(xlabel='Adjusted difference in age gap (years)', title='Age gap by vertebral level')
    a.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2); letter(a, 'c', x=-0.12)
    save(fig, OUT/'Figure3')


# ------------------------------------------------------------ Figure 4
def fig4():
    C = pd.read_csv(TAB/f'contrasts_{PRIMARY}.csv'); C = C[C['marker'] == 'Whole spine'].set_index('contrast')
    fig, ax = plt.subplots(2, 2, figsize=(W2, 150*MM), layout='constrained'); ax = ax.ravel()
    a = ax[0]
    groups = [('Healthy, internal test', P.loc[(P['group'] == 'HC') & (P['role'] == 'test'), 'g_spine'], OI['blue']),
              ('Healthy, external', P.loc[(P['group'] == 'HC') & (P['role'] == 'apply'), 'g_spine'], OI['sky']),
              ('Asymptomatic compression', P.loc[P['group'] == 'SG_compression', 'g_spine'], OI['green']),
              ('CSpineSeg abnormal', P.loc[P['group'] == 'Duke_abnormal', 'g_spine'], OI['purple']),
              ('Center 1 abnormal', P.loc[P['group'] == 'JHU_abnormal', 'g_spine'], OI['orange']),
              ('DCM', P.loc[P['group'] == 'DCM', 'g_spine'], OI['verm'])]
    boxstrip(a, groups); dashed0(a, axis='y')
    a.set_xticklabels([g[0] for g in groups], rotation=35, ha='right', rotation_mode='anchor')
    a.set(ylabel='Whole-spine age gap (years)', ylim=(-32, 42), title='Age gap by group'); letter(a, 'a', x=-0.2)
    a = ax[1]
    rows = []
    for cn, lab in [('Spine Generic compression vs controls', 'Asymptomatic compression'), ('Duke abnormal vs Duke controls', 'CSpineSeg abnormal'),
                    ('Center 1 abnormal vs controls', 'Center 1 abnormal'), ('DCM vs local controls (Centers 1-2)', 'DCM')]:
        r = C.loc[cn]; rows.append(dict(beta=r['beta'], lo=r['lo'], hi=r['hi'], p=r['p_holm_primary'], label=lab, marker='s', color=CONTRAST_COLOR[cn]))
    fplot(a, rows, 'Age-adjusted difference (years, 95% CI)', 'k')
    a.set_title('Patients vs same-center controls'); letter(a, 'b', x=-0.5)
    a = ax[2]
    J = P[P['dataset'] == 'JHU_clinical'].merge(pd.read_csv(ROOT/'jhu_clinical.csv').assign(ID_s=lambda d: d['mrn'].astype(str))[['ID_s', 'n_spinalstenosis']], on='ID_s')
    J = J[J['group'].isin(['HC', 'JHU_abnormal'])].copy()
    J['bin'] = pd.cut(J['n_spinalstenosis'], [-0.1, 0.25, 1.5, 3.5, 8], labels=['0', '1', '2–3', '≥4'])
    J['res'] = smf.ols('g_spine ~ age', J).fit().resid
    s = J.groupby('bin', observed=False)['res'].agg(['mean', 'sem'])
    for i, (_, v) in enumerate(s.iterrows()):
        ci_point(a, i, v['mean'], v['mean'] - 1.96*v['sem'], v['mean'] + 1.96*v['sem'], OI['orange'], horizontal=False, ms=6.5)
    dashed0(a, axis='y')
    Rr = pd.read_csv(TAB/'reader_reference.csv'); rr = Rr[(Rr['predictor'] == 'n_spinalstenosis') & (Rr['sample'] == 'controls + abnormal')].iloc[0]
    a.text(.04, .96, f"+{rr['beta']:.1f} years per stenotic level\n({rr['lo']:.1f} to {rr['hi']:.1f}), {fmt_p(rr['p'])}", transform=a.transAxes, va='top')
    a.set_xticks(range(len(s))); a.set_xticklabels(s.index.astype(str)); a.set_xlim(-.5, len(s) - .5)
    a.set(xlabel='Levels with canal stenosis', ylabel='Age-adjusted gap (years, 95% CI)', title='Dose–response, Center 1'); letter(a, 'c', x=-0.2)
    a = ax[3]
    FO = pd.read_csv(TAB/'focal_levels.csv').set_index('analysis')
    rows = [dict(**FO.loc[k][['beta', 'lo', 'hi', 'p']].to_dict(), label=lab, color=c) for k, lab, c in
            [('Spine Generic compression: reported levels', 'Asymptomatic:\ncompressed levels', OI['green']),
             ('Center 1 abnormal: reader spinal stenosis levels', 'Center 1:\nstenotic levels', OI['orange']),
             ('DCM: level of maximum compression', 'DCM: level of\nmaximal compression', OI['verm'])]]
    fplot(a, rows, 'Affected vs unaffected levels, same person (years)', 'k')
    a.set_title('Within-person focal excess'); letter(a, 'd', x=-0.5)
    save(fig, OUT/'Figure4')


# ------------------------------------------------------------ Figure 5
def fig5():
    D = pd.read_csv(ROOT/'dcm_clinical_frame.csv'); CL = pd.read_csv(TAB/'clinical_endpoints.csv').set_index('endpoint')
    IC = pd.read_csv(TAB/'imaging_vs_gap_clinical.csv'); IC = IC[IC['sample'] == 'available']
    JT = pd.read_csv(TAB/'imaging_vs_gap_joint.csv').set_index('outcome')
    fig = plt.figure(figsize=(W2, 165*MM), layout='constrained'); top, bot = fig.subfigures(2, 1, height_ratios=[1, 1.05])
    ax = list(top.subplots(1, 3, width_ratios=[1.25, 1, 0.62])) + list(bot.subplots(1, 2))
    # a: baseline mJOA vs age gap, adjusted for age, sex and center (component-plus-residual)
    a = ax[0]; d = D.dropna(subset=['mJOA', 'g_spine', 'age', 'sex']).copy()
    m = smf.ols('mJOA ~ g_spine + age + C(sex) + C(center)', d).fit(cov_type='HC3'); b = m.params['g_spine']
    other = m.fittedvalues - b*d['g_spine']; d['adj'] = d['mJOA'] - other + other.mean()
    a.scatter(d['g_spine'], d['adj'], s=11, color=OI['verm'], edgecolor='k', lw=.4, alpha=.85, zorder=2)
    xs = np.linspace(d['g_spine'].min(), d['g_spine'].max(), 100); xb = d['g_spine'].mean(); yb = d['adj'].mean()
    se = np.sqrt(m.scale/len(d) + (xs - xb)**2*m.cov_params().loc['g_spine', 'g_spine'])
    a.fill_between(xs, yb + b*(xs - xb) - 1.96*se, yb + b*(xs - xb) + 1.96*se, color='0.5', alpha=.18, lw=0)
    a.plot(xs, yb + b*(xs - xb), color='k', lw=1.4)
    r = CL.loc['Baseline mJOA']
    a.text(.97, .97, mn(f"{r['beta']:.2f} points per SD\n(95% CI {r['lo']:.2f} to {r['hi']:.2f})\n") + fmt_p(r['p']), transform=a.transAxes, ha='right', va='top')
    a.set_ylim(6, 21); a.set_yticks(range(6, 20, 2))
    a.set(xlabel='Whole-spine age gap (years)', ylabel='Baseline mJOA, adjusted (points)', title='Age gap and myelopathy severity'); letter(a, 'a', x=-0.2)
    # b, c: age gap versus conventional MRI measures
    ROWS = [('Age gap', 'Age gap', OI['verm']), (None, 'Age gap, adjusted for\nMSCC, T2 signal, cord area', OI['verm']),
            ('MSCC', 'MSCC', OI['grey']), ('Cord cross-sectional area (smaller)', 'Smaller cord area', OI['grey']),
            ('Cord AP diameter (smaller)', 'Smaller cord AP diameter', OI['grey']), ('Cord T2 hyperintensity', 'Cord T2 hyperintensity*', OI['grey'])]
    for k, (a, oc, lab, xl) in enumerate([(ax[1], 'Baseline mJOA', 'mJOA', 'Difference in mJOA (points)'), (ax[2], 'Frailty (mFI-5)', 'mFI-5', 'Difference in mFI-5')]):
        rows = []
        for key, nm, col in ROWS:
            if key is None:
                j = JT.loc[oc]; rows.append(dict(label=nm, beta=j['gap_est'], lo=j['lo'], hi=j['hi'], p=j['p'], color=col, marker='s'))
            else:
                r = IC[(IC['outcome'] == oc) & (IC['predictor'] == key)].iloc[0]
                rows.append(dict(label=nm, beta=r['est'], lo=r['lo'], hi=r['hi'], p=r['p'], color=col))
        fplot(a, rows, xl, OI['verm'], ptext=False)
        if k: a.set_yticklabels([])
        a.set_title(lab); letter(a, 'bc'[k], x=-0.62 if k == 0 else -0.12)
    # d: proteomics volcano
    PR = pd.read_csv(TAB/'proteomics.csv'); PS = json.load(open(TAB/'proteomics_summary.json'))
    a = ax[3]; hit = PR['p'] < .05
    a.scatter(PR.loc[~hit, 'rho'], -np.log10(PR.loc[~hit, 'p']), s=8, color='0.65', lw=0, zorder=1)
    a.scatter(PR.loc[hit, 'rho'], -np.log10(PR.loc[hit, 'p']), s=16, color=OI['verm'], edgecolor='k', lw=.5, zorder=3)
    a.axhline(-np.log10(.05), color='0.45', lw=.8, ls=(0, (4, 2)))
    for sign, xt, ha in [(-1, -.56, 'right'), (1, .4, 'left')]:
        h = PR[hit & (np.sign(PR['rho']) == sign)].assign(y=lambda d: -np.log10(d['p'])).sort_values('y', ascending=False)
        ys = list(h['y'])
        for i in range(1, len(ys)): ys[i] = min(ys[i], ys[i - 1] - .2)
        for (_, v), y in zip(h.iterrows(), ys):
            lab = v['protein'] + (f" ($\\it{{q}}$ = {v['q']:.2f})" if v['q'] < .05 else '')
            a.annotate(lab, (v['rho'], v['y']), xytext=(xt, y), textcoords='data', fontsize=6, color='k', va='center', ha=ha,
                       arrowprops=dict(arrowstyle='-', color='0.65', lw=.6, shrinkA=1, shrinkB=3))
    a.set(xlim=(-1.4, .85), ylim=(0, None), xlabel='Partial Spearman ρ with age gap', ylabel='−log$_{10}$ $\\it{P}$',
          title=f"{PS['n_nominal']} of {PS['n_proteins']} proteins at $\\it{{P}}$ < 0.05 (permutation {fmt_p(PS['perm_p'])})")
    a.set_xticks([-.4, -.2, 0, .2, .4]); letter(a, 'd', x=-0.2)
    # e: leading protein, partial rank association
    a = ax[4]; T = np.load(ROOT/'proteomics_top.npz'); g_, v_ = T['g'], T['v']; t = PR.iloc[0]
    a.scatter(g_, v_, s=11, color=OI['verm'], edgecolor='k', lw=.4, alpha=.85)
    k = np.polyfit(g_, v_, 1); xs = np.array([g_.min(), g_.max()]); a.plot(xs, np.polyval(k, xs), color='k', lw=1.4)
    a.set(xlabel='Age gap rank (adjusted for age, sex, batch, center)', ylabel=f"{t['protein']} rank (adjusted)",
          title=f"{t['protein']}: partial ρ = " + mn(f"{t['rho']:.2f}") + f", {fmt_p(t['p'])}, FDR $\\it{{q}}$ = {t['q']:.3f}"); letter(a, 'e', x=-0.2)
    save(fig, OUT/'Figure5')


# ------------------------------------------------------------ Supplementary
def figS():
    # S2: model variants
    fig, ax = plt.subplots(1, 2, figsize=(W2, 75*MM), layout='constrained')
    B = pd.read_csv(TAB/'ablation_summary.csv')
    short = {'Primary (N4 + acquisition augmentation)': 'Primary (N4 + acq. aug.)', 'Without N4 / acquisition augmentation': 'Base model (no N4/aug.)',
             'Full model': 'Full model', 'No precision weighting': 'Unweighted pooling', 'Global-age supervision only': 'Global target only',
             'No level normalization': 'No level normalization', 'No marrow/CSF intensity anchor': 'No intensity anchor', 'Marrow removed from input': 'Marrow masked'}
    a = ax[0]; y = np.arange(len(B))
    a.barh(y, B['test_mae'], color=[OI['blue']] + ['0.75']*(len(B) - 1), height=.6, edgecolor='k', lw=.6)
    a.errorbar(B['test_mae'], y, xerr=[B['test_mae'] - B['test_mae_lo'], B['test_mae_hi'] - B['test_mae']], fmt='none', ecolor='k', elinewidth=.8, capsize=2)
    a.set_yticks(y); a.set_yticklabels([short[x] for x in B['label']]); a.invert_yaxis(); a.tick_params(axis='y', length=0)
    a.set(xlabel='MAE, internal test (years, 95% CI)', title='Accuracy'); letter(a, 'a', x=-0.42)
    a = ax[1]
    fplot(a, [dict(beta=v['DCM beta'], lo=v['DCM lo'], hi=v['DCM hi'], p=v['DCM p'], label=short[v['label']], color=OI['blue'] if i == 0 else '0.75')
              for i, v in B.iterrows()], 'DCM vs local controls (years, 95% CI)', 'k', ptext=False)
    a.set_title('DCM contrast'); letter(a, 'b', x=-0.42)
    save(fig, OUT/'FigureS4_model_variants')
    # S3: compartments
    C = pd.read_csv(TAB/f'contrasts_{PRIMARY}.csv')
    fig, ax = plt.subplots(1, 4, figsize=(W2, 62*MM), layout='constrained', sharey=True)
    for a, cn in zip(ax, CONTRAST_SHORT):
        d = C[C['contrast'] == cn].set_index('marker')
        rows = [dict(beta=d.loc[m, 'beta'], lo=d.loc[m, 'lo'], hi=d.loc[m, 'hi'], p=d.loc[m, 'p_holm_primary' if m == 'Whole spine' else 'p_holm_compartments'],
                     label=m.replace('Vertebral bone', 'Bone'), marker='s' if m == 'Whole spine' else 'o') for m in ['Whole spine', 'Cord', 'CSF', 'Disc', 'Vertebral bone']]
        fplot(a, rows, 'Difference (years)', CONTRAST_COLOR[cn], ptext=False); a.set_title(CONTRAST_SHORT[cn].replace(' (Centers 1-2)', ''))
    save(fig, OUT/'FigureS1_compartments')
    # S4: reliability and acquisition
    fig, ax = plt.subplots(1, 2, figsize=(W2, 70*MM), layout='constrained', gridspec_kw=dict(width_ratios=[.8, 1.2]))
    RL = pd.read_csv(TAB/'reliability.csv'); a = ax[0]
    a.barh([0, 1], RL['icc_2_1'], color=['0.75', OI['blue']], height=.55, edgecolor='k', lw=.6)
    for i, v in enumerate(RL['icc_2_1']): a.text(v + .02, i, f'{v:.2f}', va='center')
    a.set_yticks([0, 1]); a.set_yticklabels(['All repeat series', 'Same-contrast repeats']); a.invert_yaxis(); a.set_xlim(0, 1.1); a.tick_params(axis='y', length=0)
    a.set(xlabel='ICC(2,1)', title='Repeat-scan agreement'); letter(a, 'a', x=-0.5)
    AC = pd.read_csv(TAB/'acquisition.csv'); a = ax[1]
    albl = ['OpenNeuro vs CSpineSeg', 'Spine Generic vs CSpineSeg', '3D vs 2D', 'Philips vs GE', 'Siemens vs GE', 'Slice thickness (per mm)', '3 T vs 1.5 T']
    fplot(a, [dict(beta=v['beta'], lo=v['lo'], hi=v['hi'], p=v['p'], label=albl[i]) for i, (_, v) in enumerate(AC.iterrows())], 'Difference in age gap (years, 95% CI)', '0.5')
    a.set_title('Acquisition and data source, healthy'); letter(a, 'b', x=-0.5)
    save(fig, OUT/'FigureS5_reliability_acquisition')
    # S6: baseline mJOA sensitivity and proteomics permutation null
    CL = pd.read_csv(TAB/'clinical_endpoints.csv').set_index('endpoint')
    fig, ax = plt.subplots(1, 2, figsize=(W2, 55*MM), layout='constrained', gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = ax[0]; D = pd.read_csv(ROOT/'dcm_clinical_frame.csv').dropna(subset=['sev'])
    groups = [(lab, D.loc[D['sev'] == k, 'g_spine'], OI['verm']) for k, lab in enumerate(['Mild\n(mJOA ≥ 15)', 'Moderate\n(12–14)', 'Severe\n(≤ 11)'])]
    boxstrip(a, groups); dashed0(a, axis='y'); a.set_xticklabels([g[0] for g in groups])
    r = CL.loc['Severity (mild/moderate/severe, ordinal)']
    a.text(.03, .97, f"OR {r['beta']:.2f} per SD, more severe category\n(95% CI {r['lo']:.2f} to {r['hi']:.2f}), " + fmt_p(r['p']), transform=a.transAxes, va='top')
    a.set(ylabel='Whole-spine age gap (years)', ylim=(-25, 34), title='Age gap by DCM severity category'); letter(a, 'a', x=-0.2)
    a = ax[1]; null = np.load(TAB/'proteomics_null.npy'); PS = json.load(open(TAB/'proteomics_summary.json'))
    a.hist(null, bins=np.arange(-.5, null.max() + 1.5), color='0.78', edgecolor='white', lw=.4)
    a.axvline(PS['n_nominal'], color=OI['verm'], lw=1.8, label=f"Observed ({PS['n_nominal']})")
    a.axvline(np.median(null), color='0.4', lw=1, ls=(0, (3, 2)), label=f"Null median ({np.median(null):.0f})")
    a.legend(loc='upper right'); a.set(xlabel='Proteins at $\\it{P}$ < 0.05 (count)', ylabel='Permutations', title=f"Hits vs chance (permutation {fmt_p(PS['perm_p'])})")
    letter(a, 'b', x=-0.25)
    save(fig, OUT/'FigureS2_severity_proteomics')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); figS()
    print('saved', OUT)
