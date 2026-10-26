import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

FIND = [f'{k}_{l}' for k in ['spinalstenosis','foraminalstenosis','hyperintensity']
        for l in ['c1','c2','c3','c4','c5','c6','c7','t1']]
rows = []

def add(**k): rows.append(k)

EXTRA = ROOT/'extra_raw'

def seg_dir(img):
    d = str(img).replace('T2_Nifti', 'T2_totalspineseg').replace('.nii.gz', '')
    if resolve_seg_file(d) is None:
        alt = d.replace(str(RAW), str(EXTRA))
        if resolve_seg_file(alt): return alt
    return d

# Duke / CSpineSeg
case = pd.read_csv(RAW/'Duke/case_RSNA_20250321.tsv', sep='\t')
study = pd.read_csv(DATA/'cspineseg_imaging_study.csv', sep='\t')
nst = study.groupby('cases.submitter_id').size()
st1 = study[study['cases.submitter_id'].map(nst) == 1].set_index('cases.submitter_id')
lab = pd.read_csv(RAW/'Duke/CSBA_Duke.csv')
lab['key'] = lab['File System Source'].str.extract(r'/(\d{6}-\d{6})')[0]
lab = lab.groupby('key')['Description'].agg(lambda x: '|'.join(sorted(set(x))))
case = case.set_index('submitter_id')
for f in sorted(glob.glob(str(RAW/'Duke/T2_Nifti/*.nii.gz'))):
    k = os.path.basename(f)[:13]
    if k not in case.index: add(img=f, seg=seg_dir(f), ID_s=k, dataset='Duke', exclusion='no demographic record'); continue
    c = case.loc[k]
    if k in st1.index:
        age = np.nan if st1.loc[k, 'age_at_imaging_gt89'] == 'Yes' else st1.loc[k, 'age_at_imaging']
    else:
        age = np.nan if c['age_at_index_gt89'] == 'Yes' else c['age_at_index']
    d = lab.get(k, 'not labelled')
    grp = {'healthy': 'HC', 'abnormal': 'Duke_abnormal'}.get(d)
    add(img=f, seg=seg_dir(f), ID_s=k, dataset='Duke', center='Duke', site='Duke', age=age,
        sex=norm_sex(c['sex']), group=grp, label=d,
        exclusion=None if grp else f'image label {d}')

# Spine Generic
sg = pd.read_csv(RAW/'generic_data/participants.tsv', sep='\t').set_index('participant_id')
for f in sorted(glob.glob(str(RAW/'generic_data/T2_Nifti/*.nii.gz'))):
    k = os.path.basename(f).split('_')[0]
    if k not in sg.index: add(img=f, seg=seg_dir(f), ID_s=k, dataset='SpineGeneric', exclusion='no participant record'); continue
    r = sg.loc[k]
    grp = {'HC': 'HC', 'MildCompression': 'SG_compression', 'DCM': 'SG_DCM'}.get(r['pathology'])
    add(img=f, seg=seg_dir(f), ID_s=k, dataset='SpineGeneric', center='SpineGeneric',
        site=f"SG_{r['institution_id']}", age=r['age'], sex=norm_sex(r['sex']), group=grp,
        label=r['pathology'], vendor=r['manufacturer'], model=r['manufacturers_model_name'],
        height_m=r['height']/100 if pd.notna(r['height']) else np.nan, weight_kg=r['weight'],
        sg_problem=r['problem_classification'], sg_level=r['problem_localization'],
        exclusion=None if grp else f"dataset label {r['pathology']}")

# OpenNeuro ds005616
on = pd.read_csv(RAW/'OpenNeuro/participants.tsv', sep='\t').set_index('participant_id')
for f in sorted(glob.glob(str(RAW/'OpenNeuro/T2_Nifti/*.nii.gz'))):
    k = os.path.basename(f).split('_')[0]
    r = on.loc[k] if k in on.index else None
    if r is None: add(img=f, seg=seg_dir(f), ID_s=k, dataset='OpenNeuro', exclusion='no participant record'); continue
    sten = str(r['stenosis']) if pd.notna(r['stenosis']) else ''
    exc = None
    if pd.isna(r['age']): exc = 'age missing'
    add(img=f, seg=seg_dir(f), ID_s=k, dataset='OpenNeuro', center='OpenNeuro', site=f"ON_{r['institution']}",
        age=r['age'], sex=norm_sex(r['sex']), group='ON_stenosis' if 'C' in sten else 'HC', label=r['pathology'],
        height_m=r['height (cm)']/100 if pd.notna(r['height (cm)']) else np.nan, weight_kg=r['weight (kg)'],
        exclusion=exc)

# JHU clinical (Center 1)
x = pd.read_excel(RAW/'JHU/SpineCervicalVariables.xlsx')
for c in FIND: x[c] = x[c].where(x[c].isin([0, 1]))
x['sex_n'] = x['sex'].map(norm_sex)
x['weight_kg'] = pd.to_numeric(x['weight_kg'], errors='coerce')
x['height_m'] = pd.to_numeric(x['height_m'], errors='coerce')
x['complete_read'] = x[FIND].notna().all(1)
agg = []
for mrn, d in x.groupby('mrn'):
    r = dict(mrn=mrn, n_readers=d['student'].nunique(), n_complete=int(d['complete_read'].sum()),
             age=float(d['age'].median()), age_range=float(d['age'].max()-d['age'].min()),
             sex=d['sex_n'].mode().iloc[0] if d['sex_n'].nunique() == 1 else np.nan,
             sex_conflict=d['sex_n'].nunique() > 1,
             prior_surg=float(d['prior_spine_surg'].max()), surgery=float(d['surgery'].max()),
             height_m=d['height_m'].median(), weight_kg=d['weight_kg'].median(),
             date_year=pd.to_datetime(d['date_mri'], errors='coerce').dt.year.median())
    dc = d[d['complete_read']]
    for k in ['spinalstenosis','foraminalstenosis','hyperintensity']:
        cc = [f'{k}_{l}' for l in ['c2','c3','c4','c5','c6','c7','t1']]
        r[f'n_{k}'] = float(dc[cc].sum(1).mean()) if len(dc) else np.nan
        for l in ['c2','c3','c4','c5','c6','c7','t1']:
            r[f'{k}_{l}'] = float(dc[f'{k}_{l}'].mean()) if len(dc) else np.nan
    anyf = dc[FIND].sum(1) > 0
    r['any_finding_any'] = bool(anyf.any()) if len(dc) else np.nan
    r['any_finding_all'] = bool(anyf.all()) if len(dc) else np.nan
    for s in ['sx_neckpain','sx_sensory','sx_motor','sx_bowelbladder','sx_gaitabnormal',
              'exam_motordeficit','exam_sensorydeficit','exam_umnsigns','exam_gaitabnormal']:
        r[s] = float(d[s].max()) if d[s].notna().any() else np.nan
    agg.append(r)
JC = pd.DataFrame(agg)
JC.to_csv(ROOT/'jhu_clinical.csv', index=False)
xw = pd.read_excel(RAW/'JHU/Davidar_Spine_Patient_Crosswalk11102023.xlsx')
xw['stem'] = xw['PathName'].astype(str).str.split('^').str[0]
xw = xw.drop_duplicates(['stem','mrn'])
stem2mrn = xw.groupby('stem')['mrn'].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
JCi = JC.set_index('mrn')
for f in sorted(glob.glob(str(RAW/'JHU/T2_Nifti/*.nii.gz'))):
    fx = EXTRA/'JHU/T2_Nifti'/os.path.basename(f)
    if fx.exists(): f = str(fx)
    stem = os.path.basename(f).split('_')[0]
    mrn = stem2mrn.get(stem)
    if mrn is None or mrn not in JCi.index:
        add(img=f, seg=seg_dir(f), ID_s=stem, dataset='JHU_clinical', exclusion='no clinical record'); continue
    r = JCi.loc[mrn]
    exc = 'reader findings incomplete' if r['n_complete'] == 0 else None
    grp = None if exc else ('JHU_priorsurg' if r['prior_surg'] == 1 else ('JHU_abnormal' if r['any_finding_any'] else 'HC'))
    add(img=f, seg=seg_dir(f), ID_s=str(mrn), dataset='JHU_clinical', center='Center1', site='JHU',
        age=r['age'], sex=r['sex'], group=grp, label='finding' if r['any_finding_any'] else 'no finding',
        height_m=r['height_m'], weight_kg=r['weight_kg'], age_reader_range=r['age_range'], exclusion=exc)

# DCM studies: JHU (Center 1) and MCW (Center 2)
P = pd.read_csv(RAW/'JHU_DCM/preprocessed.csv')
P['ID'] = P['ID'].astype(str).str.strip()
P['clin_source'] = 'study file'
# imaged DCM participants absent from the study file: records from the reviewed-mJOA, merged clinical and master files
DDIR = DATA/'dcm'
RV = pd.read_csv(DDIR/'Data_with_reviewed_mJOA.csv', encoding='cp1252'); RV['ID'] = RV['ID'].astype(str).str.strip()
A8 = pd.read_excel(DDIR/'AllData_merged_8.18_MDB.xlsx'); A8['ID'] = A8['ID'].astype(str).str.strip()
XW = pd.read_excel(DATA/'mcw_master.xlsx', sheet_name='id_crosswalk')
XM = pd.read_excel(DATA/'mcw_master.xlsx', sheet_name='master').drop(columns=['id_raw_mri'], errors='ignore').merge(XW[['patient_id', 'id_raw_mri']], on='patient_id')
XM['ID'] = XM['id_raw_mri'].astype(str).str.strip()
AM = pd.read_excel(DDIR/'AllData_with_FollowupMCW.xlsx').dropna(subset=['ID_extracted'])
AM['ID'] = AM['ID_extracted'].astype(str).str.strip().str.replace('vDCM_c', 'vDCM_', regex=False)
AM = AM.drop_duplicates('ID')
sev_of = lambda m: 'Mild' if m >= 15 else ('Moderate' if m >= 12 else 'Severe') if pd.notna(m) else np.nan
sup = []
for f in glob.glob(str(RAW/'JHU_DCM/T2_Nifti/*.nii.gz')) + glob.glob(str(EXTRA/'JHU_DCM/T2_Nifti/*.nii.gz')):
    k = os.path.basename(f).split('_')[0].strip()
    if k in set(P['ID']) or k in {r['ID'] for r in sup} or k not in set(XM['ID']): continue
    xm = XM[XM['ID'] == k].iloc[0]; rv = RV[RV['ID'] == k]; a8 = A8[A8['ID'] == k]
    path = str(xm['pathology_std'])
    age = rv['ageAtSurgery'].iloc[0] if len(rv) and pd.notna(rv['ageAtSurgery'].iloc[0]) else (a8['Age'].iloc[0] if len(a8) else xm['age_best'])
    sex = {'male': 1, 'female': 0}.get(str(rv['Sex'].iloc[0]).lower()) if len(rv) else (a8['Sex'].iloc[0] if len(a8) else np.nan)
    m0 = rv['reconciled mJOA'].iloc[0] if len(rv) else (a8['mJOA'].iloc[0] if len(a8) else np.nan)
    m3 = rv['reconciled \n3m mJOA'].iloc[0] if len(rv) else np.nan
    r = dict(ID=k, Pathology='DCM' if path in ('DCM', 'CSM') else path, Age=age, Sex=sex, mJOA=m0, mJOA_postop3mo=m3,
             mJOA_Delta=m3 - m0 if pd.notna(m3) and pd.notna(m0) else np.nan,
             improved=rv['improved (1/0) at 6w or 3m postop'].iloc[0] if len(rv) else np.nan,
             Severity=sev_of(m0), clin_source='reviewed/merged/master files')
    if len(rv): r['mFI-5'] = rv['mFI-5'].iloc[0]
    if len(a8):
        for c in ['Level of Max Compression', 'Presence of T2 hyperintensity', 'MSCC_diameter_AP']: r[c] = a8[c].iloc[0]
    sup.append(r)
for k, a in AM.set_index('ID').iterrows():
    if k in set(P['ID']): continue
    sup.append(dict(ID=k, Pathology=a['Pathology'], Age=a['Age'], Sex=a['Sex'], mJOA=a['mJOA'], mJOA_postop3mo=a['mjoa_3months'],
                    mJOA_Delta=a['mjoa_3months'] - a['mJOA'] if pd.notna(a['mjoa_3months']) else np.nan,
                    improved=a['mjoa_MCID3months'], Severity=sev_of(a['mJOA']), Batch=a['Batch'],
                    **{c: a[c] for c in ['Level of Max Compression', 'Presence of T2 hyperintensity', 'MSCC_diameter_AP']},
                    clin_source='MCW follow-up file'))
P = pd.concat([P, pd.DataFrame(sup)], ignore_index=True)
print('supplementary DCM-study records', pd.DataFrame(sup).groupby(['clin_source', 'Pathology']).size().to_dict())
P.to_csv(ROOT/'dcm_clinical_raw.csv', index=False)
Pi = P.drop_duplicates('ID').set_index('ID')
for f in sorted(glob.glob(str(RAW/'JHU_DCM/T2_Nifti/*.nii.gz')) + glob.glob(str(EXTRA/'JHU_DCM/T2_Nifti/*.nii.gz'))):
    k = os.path.basename(f).split('_')[0].strip()
    if k not in Pi.index: add(img=f, seg=seg_dir(f), ID_s=k, dataset='JHU_DCM', exclusion='no clinical record'); continue
    r = Pi.loc[k]
    if r['Pathology'] not in ('DCM', 'healthy'):
        add(img=f, seg=seg_dir(f), ID_s=k, dataset='JHU_DCM', exclusion=f"other pathology ({r['Pathology']})"); continue
    grp = 'DCM' if r['Pathology'] == 'DCM' else 'HC'
    add(img=f, seg=seg_dir(f), ID_s=k, dataset='JHU_DCM', center='Center1', site='JHU', age=r['Age'],
        sex='M' if r['Sex'] == 1 else 'F', group=grp, label=r['Pathology'])
sl = pd.concat([pd.read_csv(p) for p in glob.glob(str(DATA/'mcw'/'**'/'scanlog_*.csv'), recursive=True)])
sl = sl.drop_duplicates('subject').set_index('subject')
for f in sorted(glob.glob(str(RAW/'MCW_DCM/T2_Nifti/*.nii.gz'))):
    k = os.path.basename(f).split('.')[0]
    if k.startswith('vDCM'):
        k = 'vDCM_' + re.sub(r'\D', '', k[4:])
    if k.startswith('SF_'):
        s = sl.loc[k] if k in sl.index else None
        if s is None: add(img=f, seg=seg_dir(f), ID_s=k, dataset='MCW', exclusion='no scan log record'); continue
        add(img=f, seg=seg_dir(f), ID_s=k, dataset='MCW', center='Center2', site='MCW', age=s['age'],
            sex={1: 'M', 0: 'F'}.get(s['ismale'], np.nan), group='DCM' if k.startswith('SF_CM') else 'HC',
            label='SF_CM' if k.startswith('SF_CM') else 'SF_HC')
    else:
        if k not in Pi.index: add(img=f, seg=seg_dir(f), ID_s=k, dataset='MCW', exclusion='no clinical record'); continue
        r = Pi.loc[k]
        add(img=f, seg=seg_dir(f), ID_s=k, dataset='MCW', center='Center2', site='MCW', age=r['Age'],
            sex='M' if r['Sex'] == 1 else 'F', group='DCM' if r['Pathology'] == 'DCM' else 'HC',
            label='vDCM_' + str(r['Pathology']))

C = pd.DataFrame(rows)
C['img_id'] = [pseudo_id(p) for p in C['img']]
C['seg_file'] = C['seg'].map(resolve_seg_file)
C.loc[C['exclusion'].isna() & C['seg_file'].isna(), 'exclusion'] = 'segmentation missing'
C.loc[C['exclusion'].isna() & C['age'].isna(), 'exclusion'] = 'age missing'
C.loc[C['exclusion'].isna() & (C['age'] < 18), 'exclusion'] = 'age < 18'
C.loc[C['exclusion'].isna() & (C['age'] >= 90), 'exclusion'] = 'age >= 90'
C.to_csv(ROOT/'cohort_images.csv', index=False)

print('images', len(C), '| participants', C.groupby('dataset')['ID_s'].nunique().to_dict())
print(C.groupby(['dataset','group'], dropna=False).size().to_string())
print('\nexclusions (images):')
print(C.groupby(['dataset','exclusion']).size().to_string())
ok = C[C['exclusion'].isna()]
print('\neligible participants by dataset/group:')
print(ok.groupby(['dataset','group'])['ID_s'].nunique().to_string())
print('\nJHU readers: 2 readers', int((JC.n_readers >= 2).sum()), '| age conflict', int((JC.age_range > 1).sum()),
      '| sex conflict', int(JC.sex_conflict.sum()), '| prior surgery', int((JC.prior_surg == 1).sum()))
