import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 600, 'pdf.fonttype': 42,
                     'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.frameon': False, 'legend.fontsize': 7.5})
OI = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73', yellow='#F0E442', blue='#0072B2',
          verm='#D55E00', purple='#CC79A7', grey='#999999')
GC = {'HC': OI['blue'], 'JHU_abnormal': OI['orange'], 'DCM': OI['verm'], 'Duke_abnormal': OI['purple'],
      'SG_compression': OI['green'], 'JHU_priorsurg': OI['grey']}
GL = {'HC': 'Healthy', 'JHU_abnormal': 'Center 1 abnormal', 'DCM': 'DCM', 'Duke_abnormal': 'CSpineSeg abnormal',
      'SG_compression': 'Asymptomatic compression', 'JHU_priorsurg': 'Prior surgery'}
MM = 1/25.4


def panel(ax, s): ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')


def save(fig, name):
    for e in ('png', 'pdf'): fig.savefig(FIG/f'{name}.{e}', bbox_inches='tight')
    plt.close(fig)


G = load_gaps(); P = G[G['role'] != 'retest'].copy(); P['cal'] = P['age'] + P['g_spine']
A = pd.read_csv(TAB/f'accuracy_{PRIMARY}.csv')

# Figure 2: performance
fig, ax = plt.subplots(2, 3, figsize=(180*MM, 120*MM), layout='constrained'); ax = ax.ravel()
for i, (nm, d, col) in enumerate([('Internal test', P[P['role'] == 'test'], OI['blue']),
                                  ('External Center 1 controls', P[(P['group'] == 'HC') & (P['center'] == 'Center1')], OI['green']),
                                  ('External Center 2 controls', P[(P['group'] == 'HC') & (P['center'] == 'Center2')], OI['orange'])]):
    a = ax[i]
    a.scatter(d['age'], d['raw_spine'], s=7, color=col, alpha=.6, lw=0, label='Network output')
    a.scatter(d['age'], d['cal'], s=7, facecolor='none', edgecolor='0.35', lw=.5, label='Calibrated')
    a.plot([15, 92], [15, 92], ls='--', color='0.5', lw=.8)
    rr = A[(A['set'] == nm) & (A['kind'] == 'raw')].iloc[0]; rc = A[(A['set'] == nm) & (A['kind'] == 'calibrated')].iloc[0]
    a.text(.04, .97, f"{nm}\nn = {int(rr['n'])}\nMAE {rr['mae']:.2f} (raw), {rc['mae']:.2f} (cal.)\nr = {rr['r']:.3f}",
           transform=a.transAxes, va='top', fontsize=6.5)
    a.set(xlim=(15, 92), ylim=(15, 92), xlabel='Chronological age (y)', ylabel='Estimated age (y)')
    if i == 0: a.legend(loc='lower right', fontsize=6, handletextpad=.2, borderaxespad=.2)
    panel(a, 'abc'[i])
a = ax[3]
d = P[P['role'].isin(['dev', 'test'])]
a.scatter(d['age'], d['g_spine'], s=4, color=OI['blue'], alpha=.35, lw=0)
a.axhline(0, color='0.5', ls='--', lw=.8)
xs = np.linspace(d['age'].min(), d['age'].max(), 50); a.plot(xs, np.poly1d(np.polyfit(d['age'], d['g_spine'], 1))(xs), color='k', lw=1)
a.set(xlabel='Chronological age (y)', ylabel='Age gap (y)', ylim=(-25, 25)); panel(a, 'd')
R = pd.read_csv(TAB/'regional_performance.csv'); R = R[R['region'].isin(['cord', 'csf', 'disc', 'bone'] + VNAMES)]
a = ax[4]
lab = {'cord': 'Cord', 'csf': 'CSF', 'disc': 'Disc', 'bone': 'Bone (C3-C7)'}
a.barh(range(len(R)), R['r'], xerr=[R['r'] - R['r_lo'], R['r_hi'] - R['r']], color='0.55', height=.65, error_kw=dict(lw=.6))
a.set_yticks(range(len(R))); a.set_yticklabels([lab.get(x, x) for x in R['region']]); a.invert_yaxis()
a.set(xlim=(0.5, 1), xlabel='r with age (internal test)'); panel(a, 'e')
a = ax[5]
if (TAB/'ablation_summary.csv').exists():
    B = pd.read_csv(TAB/'ablation_summary.csv')
    a.barh(range(len(B)), B['test_mae'], color='0.55', height=.65)
    short = {'Full model': 'Full', 'No precision weighting': 'No precision wt.', 'Global-age supervision only': 'Global target only',
             'No level normalization': 'No level norm.', 'No marrow/CSF intensity anchor': 'No intensity anchor', 'Marrow removed from input': 'Marrow masked'}
    a.set_yticks(range(len(B))); a.set_yticklabels([short.get(x, x) for x in B['label']]); a.invert_yaxis()
    a.set(xlabel='MAE, internal test (y)'); panel(a, 'f')
save(fig, 'Figure2_performance')

# Figure 4: group differences
C = pd.read_csv(TAB/f'contrasts_{PRIMARY}.csv')
fig, ax = plt.subplots(1, 3, figsize=(180*MM, 70*MM), gridspec_kw=dict(width_ratios=[1.1, 1.3, 1]))
a = ax[0]
order = ['HC', 'Duke_abnormal', 'SG_compression', 'JHU_abnormal', 'DCM']
data = [P.loc[(P['group'] == g) & ~((g == 'HC') & P['role'].isin(['dev'])), 'g_spine'].dropna() for g in order]
bp = a.boxplot(data, widths=.6, patch_artist=True, showfliers=False, medianprops=dict(color='k'))
for b, g in zip(bp['boxes'], order): b.set_facecolor(GC[g]); b.set_alpha(.7)
a.set_xticks(range(1, len(order)+1)); a.set_xticklabels([GL[g] for g in order], rotation=35, ha='right')
a.axhline(0, color='0.5', ls='--', lw=.8); a.set(ylabel='Age gap (y), unadjusted'); panel(a, 'a')
a = ax[1]
y = 0; ticks = []
for cn in C['contrast'].unique():
    d = C[C['contrast'] == cn]
    for _, r in d.iterrows():
        prim = r['marker'] == 'Whole spine'
        a.errorbar(r['beta'], y, xerr=[[r['beta'] - r['lo']], [r['hi'] - r['beta']]], fmt='s' if prim else 'o',
                   color='k' if prim else '0.5', ms=4 if prim else 3, lw=.8)
        ticks.append((y, f"{cn.split(' vs')[0]} - {r['marker']}" if prim else r['marker'])); y -= 1
    y -= .6
a.axvline(0, color='0.5', ls='--', lw=.8)
a.set_yticks([t[0] for t in ticks]); a.set_yticklabels([t[1] for t in ticks], fontsize=6)
a.set(xlabel='Adjusted difference in age gap (y)'); panel(a, 'b')
a = ax[2]
Rr = pd.read_csv(TAB/'reader_reference.csv')
J = P[P['dataset'] == 'JHU_clinical'].merge(pd.read_csv(ROOT/'jhu_clinical.csv').assign(ID_s=lambda d: d['mrn'].astype(str))[['ID_s', 'n_spinalstenosis']], on='ID_s')
J = J[J['group'].isin(['HC', 'JHU_abnormal'])]
J['bin'] = pd.cut(J['n_spinalstenosis'], [-0.1, 0.25, 1.5, 3.5, 8], labels=['0', '1', '2-3', '>=4'])
m = smf.ols('g_spine ~ age', J).fit(); J['res'] = m.resid
s = J.groupby('bin')['res'].agg(['mean', 'sem', 'size'])
a.errorbar(range(len(s)), s['mean'], yerr=1.96*s['sem'], fmt='o', color='k', capsize=2, lw=.8)
a.set_xticks(range(len(s))); a.set_xticklabels(s.index.astype(str))
a.set(xlabel='Levels with spinal stenosis (readers)', ylabel='Age-adjusted gap (y)'); a.axhline(0, color='0.5', ls='--', lw=.8)
panel(a, 'c')
fig.tight_layout(); save(fig, 'Figure4_groups')
print('figures saved')
