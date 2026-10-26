import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c18_mainfigs import *

H = pd.read_csv(TAB/'harmonization.csv'); HS = json.load(open(TAB/'harmonization_summary.json'))
rows = []
TAGS = [(PRIMARY, 'Primary'), (f'{PRIMARY}_thick', '3D converted to 3.3-mm 2D-like slices')] + ([('main', 'Base model (no N4, no acquisition augmentation)')] if PRIMARY != 'main' else [])
HD = json.load(open(TAB/'harm_decision.json'))
for tag, lab in TAGS:
    G_ = load_gaps(tag); Q = G_[G_['role'] != 'retest']
    for nm, c, k, rhs in [('DCM vs controls, Center 1', Q[(Q.group == 'HC') & (Q.center == 'Center1')], Q[(Q.group == 'DCM') & (Q.center == 'Center1')], 'grp + age'),
                          ('DCM vs controls, Center 2', Q[(Q.group == 'HC') & (Q.center == 'Center2')], Q[(Q.group == 'DCM') & (Q.center == 'Center2')], 'grp + age'),
                          ('Spine Generic compression vs controls', Q[(Q.group == 'HC') & (Q.dataset == 'SpineGeneric')], Q[Q.group == 'SG_compression'], 'grp + age + C(site)')]:
        r = group_effect(pd.concat([c.assign(grp=0), k.assign(grp=1)]), 'g_spine', rhs); r.update(contrast=nm, analysis=lab); rows.append(r)
    t = Q[Q.role == 'test']; rows.append(dict(contrast='Internal test MAE (calibrated)', analysis=lab, beta=t['g_spine'].abs().mean(), n=len(t)))
    if tag in HD: rows.append(dict(contrast='Batch ICC, healthy (vendor x field)', analysis=lab, beta=HD[tag]['batch_icc'], n=HD[tag]['n_healthy']))
T = pd.concat([H, pd.DataFrame(rows)], ignore_index=True); T.to_csv(TAB/'harmonization_all.csv', index=False)

fig, ax = plt.subplots(1, 2, figsize=(W2, 78*MM), layout='constrained', gridspec_kw=dict(width_ratios=[1.5, 1]))
a = ax[0]
specs = [('Center 1 abnormal vs controls', 'Center 1 abnormal'), ('DCM vs controls, Center 1', 'DCM, Center 1'), ('DCM vs controls, Center 2', 'DCM, Center 2'),
         ('Duke abnormal vs Duke controls', 'CSpineSeg abnormal'), ('Spine Generic compression vs controls', 'Asymptomatic compression')]
var = [('unharmonized, same participants (vendor_field)', 'Unharmonized', '0.55', 'o'), ('ComBat, batch = vendor_field', 'ComBat (vendor × field)', OI['blue'], 's'),
       ('ComBat, batch = model', 'ComBat (scanner model)', OI['sky'], 'D')]
for i, (cn, lab) in enumerate(specs):
    for j, (an, vl, col, mk) in enumerate(var):
        r = H[(H['contrast'] == cn) & (H['analysis'] == an)]
        if r.empty: continue
        r = r.iloc[0]; y = i + (j - 1)*.22
        a.plot([r['lo'], r['hi']], [y, y], color=col, lw=1.1); a.plot(r['beta'], y, mk, color=col, ms=4.5, label=vl if i == 0 else None)
dashed0(a); a.set_yticks(range(len(specs))); a.set_yticklabels([s[1] for s in specs]); a.set_ylim(len(specs) - .5, -.6); a.tick_params(axis='y', length=0)
a.set(xlabel='Difference in age gap vs same-center controls (years)', title='Contrasts before and after harmonization'); a.legend(loc='lower right')
letter(a, 'a', x=-0.33)
a = ax[1]
bv = HS['vendor_field']['batch_variance']
ic = ([HD['main']['batch_icc']] if PRIMARY != 'main' else []) + [bv[0]['icc_batch'], bv[1]['icc_batch']]
xl = (['Base model'] if PRIMARY != 'main' else []) + (['N4 + acq. aug.'] if PRIMARY != 'main' else ['Before']) + ['+ ComBat']
a.bar(range(len(ic)), [100*v for v in ic], color=['0.75', '0.5', OI['blue']][-len(ic):], width=.55)
for i, v in enumerate(ic): a.text(i, 100*v + .15, f'{100*v:.1f}%', ha='center')
a.set_xticks(range(len(ic))); a.set_xticklabels(xl); a.set_ylim(0, 6.5)
a.set(ylabel='Variance due to scanner/site (%)', title='Batch variance, healthy participants'); letter(a, 'b', x=-0.3)
save(fig, OUT/'FigureS3_harmonization')
print(pd.DataFrame(rows).round(3).to_string())
