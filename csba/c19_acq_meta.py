import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import pydicom, nibabel as nib
from concurrent.futures import ProcessPoolExecutor

ROOTD = DATA/'jhu_dicom'
F = ['PatientName', 'SeriesDescription', 'Manufacturer', 'ManufacturerModelName', 'MagneticFieldStrength', 'SliceThickness',
     'RepetitionTime', 'EchoTime', 'PixelSpacing', 'Rows', 'Columns', 'StudyDate', 'InstitutionName', 'StationName', 'MRAcquisitionType']


def study(d):
    out = []
    for s in sorted(p for p in d.iterdir() if p.is_dir()):
        fs = sorted(s.glob('*.dcm'))
        if not fs: continue
        try: h = pydicom.dcmread(fs[0], stop_before_pixels=True)
        except Exception: continue
        r = {k: str(getattr(h, k, '')) for k in F}
        r['n_files'] = len(fs); out.append(r)
    return out


if __name__ == '__main__':
    with ProcessPoolExecutor(32) as ex:
        R = [r for rs in ex.map(study, sorted(p for p in ROOTD.iterdir() if p.is_dir())) for r in rs]
    D = pd.DataFrame(R)
    D['stem'] = D['PatientName'].str.split('^').str[0]
    C = pd.read_csv(ROOT/'cohort_images.csv'); J = C[C['dataset'] == 'JHU_clinical'].copy()
    rows = []
    for _, r in J.iterrows():
        stem = os.path.basename(r['img']).split('_')[0]
        h = nib.load(r['img']).header; sh = h.get_data_shape(); z = h.get_zooms()
        c = D[(D['stem'] == stem) & D['SeriesDescription'].str.contains('T2', case=False) & D['SeriesDescription'].str.contains('sag', case=False)]
        m = c[c['n_files'].astype(int).isin(sh)]
        m = m if len(m) else c
        x = m.iloc[0] if len(m) else None
        rows.append(dict(img_id=r['img_id'], n_cand=len(c), **({k: x[k] for k in ['Manufacturer', 'ManufacturerModelName', 'MagneticFieldStrength', 'SliceThickness', 'RepetitionTime', 'EchoTime', 'StudyDate', 'InstitutionName', 'MRAcquisitionType']} if x is not None else {})))
    O = pd.DataFrame(rows); O.to_csv(ROOT/'jhu_clinical_acquisition.csv', index=False)
    print('series', len(D), '| images matched', int(O['Manufacturer'].notna().sum()), '/', len(O))
    print(O['Manufacturer'].str.upper().value_counts().to_dict()); print(O['MagneticFieldStrength'].value_counts().to_dict())
