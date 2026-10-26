import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from concurrent.futures import ProcessPoolExecutor

T1DIR = DATA/'jhu_sagittal_t1'
BODY = [13, 14, 15, 16, 17]


def work(a):
    iid, t2, seg = a
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    stem = os.path.basename(t2).split('_')[0]
    best = None
    for f in sorted(T1DIR.glob(f'{stem}_SpinePath_Sagittal_T1*.nii.gz')):
        try:
            t1 = sitk.ReadImage(str(f), sitk.sitkFloat32)
            lab = sitk.Resample(sitk.ReadImage(seg, sitk.sitkUInt8), t1, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            L = sitk.GetArrayFromImage(lab); I = sitk.GetArrayFromImage(t1)
        except Exception:
            continue
        vb = ndi.binary_erosion(np.isin(L, BODY), iterations=1)
        dk = ndi.binary_erosion(np.isin(L, DISCS), iterations=1)
        cs = L == CSF
        if vb.sum() < 200 or dk.sum() < 50 or cs.sum() < 100: continue
        r = dict(img_id=iid, t1_file=f.name, n_vox=int(vb.sum()), t1_vert=float(np.median(I[vb])),
                 t1_disc=float(np.median(I[dk])), t1_csf=float(np.median(I[cs])))
        r['t1_vert_disc'] = r['t1_vert']/max(r['t1_disc'], 1e-6); r['t1_vert_csf'] = r['t1_vert']/max(r['t1_csf'], 1e-6)
        if best is None or r['n_vox'] > best['n_vox']: best = r
    return best


if __name__ == '__main__':
    S = pd.read_csv(ROOT/'splits.csv')
    S = S[(S['dataset'] == 'JHU_clinical') & (S['role'] != 'retest')]
    with ProcessPoolExecutor(32) as ex:
        res = [r for r in ex.map(work, [(r.img_id, r.img, r.seg_file) for r in S.itertuples()]) if r]
    T = pd.DataFrame(res); T.to_csv(ROOT/'t1_marrow.csv', index=False)
    print(len(T), 'of', len(S), '| vert/disc median', T['t1_vert_disc'].median().round(2), '| vert>disc fraction', (T['t1_vert_disc'] > 1).mean().round(2))
