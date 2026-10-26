import os, re, glob, json, math, random, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
import SimpleITK as sitk
import scipy.ndimage as ndi
from scipy.interpolate import splprep, splev, interp1d
warnings.filterwarnings('ignore')

ROOT = Path(os.environ.get('CSBA_ROOT', 'work'))        # outputs (npz, preds, tables, figures)
RAW = Path(os.environ.get('CSBA_RAW', 'data/raw'))       # NIfTI images and TotalSpineSeg labels
DATA = Path(os.environ.get('CSBA_DATA', 'data'))         # metadata, clinical and proteomic tables
NPZ = ROOT/'npz'

CORD, CSF = 1, 2
VERTS = [12, 13, 14, 15, 16, 17, 21]
VNAMES = ['C2','C3','C4','C5','C6','C7','T1']
DISCS = [63, 64, 65, 66, 67, 71]
DNAMES = ['C2C3','C3C4','C4C5','C5C6','C6C7','C7T1']
SPINE = [CORD, CSF] + VERTS + DISCS
LNAME = dict(zip([CORD,CSF]+VERTS+DISCS, ['cord','CSF']+VNAMES+DNAMES))

N_SLICES, XY, XY_MM = 192, 96, 0.6
ANCHORS = np.round(np.linspace(0.07, 0.93, len(VERTS)) * (N_SLICES-1)).astype(int)
ANCHOR_SPAN = int(ANCHORS[-1] - ANCHORS[0])
FEATHER_MM = 4.0
UNIFORM_MM = 0.623
JITTER_MM = 1.5
MIN_VOX = dict(cord=200, csf=200, vert=50, disc=20)
REGIONS = ['spine','cord','csf','disc'] + VNAMES


def pseudo_id(*parts):
    return hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:10]


def norm_sex(s):
    s = str(s).strip().upper()
    return 'F' if s.startswith('F') else ('M' if s.startswith('M') else np.nan)


def resolve_seg_file(seg_dir):
    p = Path(str(seg_dir))
    if p.is_file(): return str(p)
    if p.is_dir():
        for pat in ['*step2_output.nii.gz', '*step2*.nii.gz']:
            h = sorted(p.glob(pat))
            if h: return str(h[0])
    return None


def reorient(img, interp=sitk.sitkLinear):
    f = sitk.DICOMOrientImageFilter(); f.SetDesiredCoordinateOrientation('LPS')
    return f.Execute(img)


def phys_to_index(img, pts):
    o = np.array(img.GetOrigin()); s = np.array(img.GetSpacing())
    D = np.array(img.GetDirection()).reshape(3,3)
    return (np.linalg.solve(D, (pts - o).T).T) / s


def dense_centerline(mask_lps, n_dense=1200, smooth=60.0):
    st = sitk.LabelShapeStatisticsImageFilter(); st.Execute(mask_lps)
    present = [v for v in VERTS if st.HasLabel(v)]
    if len(present) < 4 or not st.HasLabel(CORD): return None
    vz = {v: st.GetCentroid(v)[2] for v in present}
    sup_larger = vz[min(present)] > vz[max(present)]
    m = sitk.GetArrayFromImage(mask_lps)
    sp = np.array(mask_lps.GetSpacing()); D = np.array(mask_lps.GetDirection()).reshape(3,3)
    org = np.array(mask_lps.GetOrigin())
    z, y, x = np.where(m == CORD)
    cord = (D * sp @ np.c_[x, y, z].T).T + org
    if len(cord) < 60: return None
    hi = max(vz[min(present)], vz[max(present)]) + 12.0
    lo = min(vz[min(present)], vz[max(present)]) - 12.0
    cord = cord[(cord[:,2] <= hi) & (cord[:,2] >= lo)]
    if len(cord) < 60: return None
    cz = cord[:,2]
    order = np.argsort(-cz) if sup_larger else np.argsort(cz)
    cord, cz = cord[order], cz[order]
    nb = int(np.clip(len(cord)//10, 10, 80))
    edges = np.linspace(cz[0], cz[-1], nb+1)
    pick = (lambda i: (cz <= edges[i]) & (cz > edges[i+1])) if sup_larger else \
           (lambda i: (cz >= edges[i]) & (cz < edges[i+1]))
    ctrl = np.array([cord[pick(i)].mean(0) for i in range(nb) if pick(i).any()])
    if len(ctrl) < 6: return None
    try:
        tck, _ = splprep(ctrl.T, s=smooth, k=3)
        cl = np.array(splev(np.linspace(0, 1, n_dense), tck)).T
    except Exception:
        return None
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))]
    return cl, seg, {v: np.array(st.GetCentroid(v)) for v in present}, sup_larger


def level_anchor_map(cl, arc, centroids, uniform=False, jitter=None):
    pos = {v: float(arc[int(np.argmin(np.linalg.norm(cl - c, axis=1)))]) for v, c in centroids.items()}
    if jitter is not None:
        pos = {v: p + float(jitter.normal(0, JITTER_MM)) for v, p in pos.items()}
    have = [v for v in VERTS if v in pos]
    if len(have) < 4: return None
    t = np.array([pos[v] for v in have])
    if not np.all(np.diff(t) > 0):
        keep = [0]
        for i in range(1, len(t)):
            if t[i] > t[keep[-1]]: keep.append(i)
        have = [have[i] for i in keep]; t = t[keep]
        if len(have) < 4: return None
    a = ANCHORS[[VERTS.index(v) for v in have]].astype(float)
    if uniform:
        a0, t0 = a[0], t[0]
        f = lambda k: t0 + (k - a0) * UNIFORM_MM
    else:
        f = interp1d(a, t, kind='linear', fill_value='extrapolate')
    samp = f(np.arange(N_SLICES, dtype=float))
    samp = np.maximum.accumulate(np.clip(samp, arc[0], arc[-1]))
    return samp, {LNAME[v]: int(ANCHORS[VERTS.index(v)]) for v in have}


def straighten_at(img_lps, cl, arc, samp_arc, order=1, cval=0.0):
    idx = np.clip(np.searchsorted(arc, samp_arc), 1, len(arc)-1)
    w = (samp_arc - arc[idx-1]) / np.maximum(arc[idx] - arc[idx-1], 1e-6)
    C = cl[idx-1] * (1-w)[:,None] + cl[idx] * w[:,None]
    T = np.gradient(C, axis=0); T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    ref = np.array([0., -1., 0.]); half = (XY-1)/2
    g = (np.arange(XY) - half) * XY_MM
    gv, gh = np.meshgrid(g, g, indexing='ij')
    pts = np.empty((N_SLICES, XY, XY, 3))
    for k in range(N_SLICES):
        t = T[k]
        n1 = ref - t*np.dot(ref, t)
        if np.linalg.norm(n1) < 1e-3: n1 = np.array([1.,0.,0.]) - t*t[0]
        n1 /= np.linalg.norm(n1) + 1e-9
        n2 = np.cross(t, n1); n2 /= np.linalg.norm(n2) + 1e-9
        pts[k] = C[k] + gv[...,None]*n1 + gh[...,None]*n2
    ii = phys_to_index(img_lps, pts.reshape(-1,3))
    arr = sitk.GetArrayFromImage(img_lps).astype(np.float32)
    v = ndi.map_coordinates(arr, np.vstack([ii[:,2], ii[:,1], ii[:,0]]), order=order, mode='constant', cval=cval)
    return v.reshape(N_SLICES, XY, XY)


def curvature_metrics(cl, cents):
    """Native sagittal alignment from the 3D centerline: arc/chord ratio between C2 and C7
    centroids, and the angle between the C2-C4 and C5-C7 centroid directions."""
    out = dict(arc_chord=np.nan, c2c7_angle=np.nan)
    if not all(v in cents for v in [12, 14, 15, 17]): return out
    pa = cl[int(np.argmin(np.linalg.norm(cl - cents[12], axis=1)))]
    pb = cl[int(np.argmin(np.linalg.norm(cl - cents[17], axis=1)))]
    ia, ib = sorted([int(np.argmin(np.linalg.norm(cl - c, axis=1))) for c in (cents[12], cents[17])])
    arc = np.sum(np.linalg.norm(np.diff(cl[ia:ib+1], axis=0), axis=1))
    chord = np.linalg.norm(pa - pb)
    u = cents[14] - cents[12]; w = cents[17] - cents[15]
    ang = math.degrees(math.acos(np.clip(np.dot(u, w)/(np.linalg.norm(u)*np.linalg.norm(w)+1e-9), -1, 1)))
    out.update(arc_chord=float(arc/max(chord, 1e-6)), c2c7_angle=float(ang))
    return out


def n4_correct(img):
    img = sitk.Maximum(img, 0.0)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    sz = img.GetSize(); f = [4 if s >= 64 else 1 for s in sz]
    small, msmall = sitk.Shrink(img, f), sitk.Shrink(mask, f)
    c = sitk.N4BiasFieldCorrectionImageFilter(); c.SetMaximumNumberOfIterations([50]*4)
    c.Execute(small, msmall)
    return sitk.Cast(img / sitk.Exp(c.GetLogBiasFieldAsImage(img)), sitk.sitkFloat32)


def extract(img_path, seg_path, uniform=False, jitter_seed=None, n4=False):
    f = resolve_seg_file(seg_path)
    if f is None: return None, 'no_segmentation'
    mri = reorient(sitk.ReadImage(str(img_path), sitk.sitkFloat32))
    if n4: mri = n4_correct(mri)
    mask = sitk.Resample(reorient(sitk.ReadImage(f, sitk.sitkUInt8), sitk.sitkNearestNeighbor),
                         mri, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    cc = dense_centerline(mask)
    if cc is None: return None, 'no_centerline'
    cl, arc, cents, sup = cc
    jit = np.random.default_rng(jitter_seed) if jitter_seed is not None else None
    lm = level_anchor_map(cl, arc, cents, uniform=uniform, jitter=jit)
    if lm is None: return None, 'no_anchors'
    samp, anchors_used = lm
    img_s = straighten_at(mri, cl, arc, samp, order=1)
    lab_s = straighten_at(mask, cl, arc, samp, order=0).round().astype(np.int16)
    struct = np.isin(lab_s, SPINE)
    if struct.sum() < 500: return None, 'empty'
    dz = float(np.median(np.diff(samp)))
    dist = ndi.distance_transform_edt(~struct, sampling=(max(dz,1e-3), XY_MM, XY_MM))
    w = np.exp(-(dist**2)/(2*FEATHER_MM**2)); w[struct] = 1.0
    soft = (img_s * w).astype(np.float32)
    if not np.isfinite(soft).all() or soft.max() <= 0: return None, 'blank'
    sp = mri.GetSpacing()
    meta = dict(arc_len=float(arc[-1]), slice_mm=dz, anchors=anchors_used, n_anchors=len(anchors_used),
                n_verts_vol=int(sum((lab_s==v).sum() > 50 for v in VERTS)),
                n_discs_vol=int(sum((lab_s==d).sum() > 20 for d in DISCS)),
                cord_vox=int((lab_s==CORD).sum()), csf_vox=int((lab_s==CSF).sum()),
                vox_min=float(min(sp)), vox_max=float(max(sp)), **curvature_metrics(cl, cents))
    return (soft, lab_s, meta), 'ok'
