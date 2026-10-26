import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *

SIEMENS = {'AERA', 'SKYRA', 'VERIO', 'AVANTO', 'ESPREE', 'SYMPHONY', 'SYMPHONYP', 'SOLA', 'ALTEA', 'VIDA FIT', 'PRISMA'}


def with_acquisition(tag=None):
    tag = tag or PRIMARY
    G = load_gaps(tag); P = G[G['role'] != 'retest'].copy()
    A1 = pd.read_csv(ROOT/'jhu_clinical_acquisition.csv')[['img_id', 'Manufacturer', 'ManufacturerModelName', 'MagneticFieldStrength']]
    A2 = pd.read_csv(ROOT/'jhu_dcm_scanner.csv').drop_duplicates('ID_s').rename(columns={'model': 'ManufacturerModelName', 'field': 'MagneticFieldStrength'})[['ID_s', 'ManufacturerModelName', 'MagneticFieldStrength']]
    P = P.merge(A1, on='img_id', how='left')
    dcm = P['dataset'] == 'JHU_DCM'
    P.loc[dcm, ['ManufacturerModelName', 'MagneticFieldStrength']] = P.loc[dcm, ['ID_s']].merge(A2, on='ID_s', how='left')[['ManufacturerModelName', 'MagneticFieldStrength']].values
    P['scanner'] = P['ManufacturerModelName'].astype(str).str.upper().str.replace('MAGNETOM ', '', regex=False).str.strip()
    P['fieldT'] = pd.to_numeric(P['MagneticFieldStrength'], errors='coerce').round(1)
    return P


def batch_of(r, level):
    if r['dataset'] == 'Duke': return f"Duke_{r['protocol']}"
    if r['dataset'] in ('SpineGeneric', 'OpenNeuro'): return r['site']
    if r['dataset'] == 'MCW': return 'Center2'
    if r['scanner'] in ('NAN', 'NONE', ''): return 'C1_unknown'
    if level == 'model': return 'C1_' + r['scanner']
    return 'C1_' + ('Siemens' if r['scanner'] in SIEMENS else 'other') + f"_{r['fieldT']:.1f}T"
