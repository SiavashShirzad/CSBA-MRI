import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from concurrent.futures import ProcessPoolExecutor

THICK_MM = 3.3                      # median slice thickness of the 2D sagittal acquisitions
SRC = sys.argv[1] if len(sys.argv) > 1 else 'npz'
NPZ = ROOT/SRC; OUT = ROOT/(SRC + '_thick'); OUT.mkdir(exist_ok=True)
x = np.arange(XY)*XY_MM


def thick(img):
    # simulate a 2D sagittal acquisition: average over THICK_MM slabs along left-right, then interpolate back
    c = np.arange(x[0] + THICK_MM/2, x[-1], THICK_MM)
    W = (np.abs(x[None, :] - c[:, None]) <= THICK_MM/2).astype(np.float32); W /= W.sum(1, keepdims=True)
    slab = np.einsum('kx,zyx->zyk', W, img)
    out = np.empty_like(img)
    for z in range(img.shape[0]):
        for y in range(img.shape[1]):
            out[z, y] = np.interp(x, c, slab[z, y])
    return out


def work(a):
    iid, is3d = a
    src, dst = NPZ/f'{iid}.npz', OUT/f'{iid}.npz'
    if dst.exists(): return
    if not is3d: os.symlink(src, dst); return
    d = dict(np.load(src)); d['image'] = thick(d['image']).astype(np.float32)
    np.savez_compressed(dst, **d)


if __name__ == '__main__':
    S = pd.read_csv(ROOT/'splits.csv')
    S = S[[(NPZ/f'{i}.npz').exists() for i in S['img_id']]]
    jobs = list(zip(S['img_id'], S['vox_max'] < 2.0))
    with ProcessPoolExecutor(48) as ex: list(ex.map(work, jobs))
    print('3D images thickened', int(S['vox_max'].lt(2.0).sum()), 'of', len(S))
