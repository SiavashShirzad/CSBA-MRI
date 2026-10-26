import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
G = load_gaps(); P = G[G['role'] != 'retest'].copy()
def grp(r):
    if r['group'] == 'HC':
        return {'dev': 'Healthy, development (cross-validated)', 'test': 'Healthy, internal test (locked)'}.get(r['role'], f"Healthy, external {r['center'].replace('Center', 'Center ')}")
    return {'JHU_abnormal': 'Center 1 abnormal', 'JHU_priorsurg': 'Center 1 prior spine surgery', 'DCM': f"DCM, {r['center'].replace('Center', 'Center ')}",
            'Duke_abnormal': 'CSpineSeg abnormal', 'SG_compression': 'Spine Generic asymptomatic compression', 'SG_DCM': 'Spine Generic DCM', 'ON_stenosis': 'OpenNeuro stenosis'}[r['group']]
P['tgroup'] = P.apply(grp, axis=1)
T = P.groupby('tgroup').apply(lambda d: pd.Series({'n': len(d), 'Age, mean (SD)': f"{d.age.mean():.1f} ({d.age.std():.1f})",
    'Age range': f"{d.age.min():.0f}-{d.age.max():.0f}", 'Female, n (%)': f"{(d.sex=='F').sum()} ({100*(d.sex=='F').mean():.0f})",
    'Sex missing': int(d.sex.isna().sum()), 'Datasets': ', '.join(sorted(d.dataset.unique())), 'Slice thickness, median (mm)': f"{d.vox_max.median():.2f}"}))
T.to_csv(TAB/'Table1_cohort.csv'); print(T.to_string())
print('\nTOTAL participants', len(P), '| female', int((P.sex=='F').sum()), f"({100*(P.sex=='F').mean():.1f}%)", '| sex missing', int(P.sex.isna().sum()),
      '| age mean', round(P.age.mean(),1), 'SD', round(P.age.std(),1), '| SG institutions', P.loc[P.dataset=='SpineGeneric','site'].nunique(), '| 2D/3D', P.protocol.value_counts().to_dict())
Q = pd.read_csv(ROOT/'cohort_images_qc.csv')
fl = Q.groupby(['dataset', Q['exclusion'].fillna('included')]).agg(images=('img_id', 'size'), participants=('ID_s', 'nunique')).reset_index()
fl.to_csv(TAB/'flow_exclusions.csv', index=False); print(fl.to_string())
