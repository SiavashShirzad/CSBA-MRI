import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from concurrent.futures import ProcessPoolExecutor


def occ(a):
    iid, d = a
    lab = np.load(ROOT/d/f'{iid}.npz')['compartments']
    return np.stack([(lab == v).sum(axis=(1, 2)) for v in VERTS]).astype(np.float32)


if __name__ == '__main__':
    S = pd.read_csv(ROOT/'splits.csv'); S = S[S['role'] != 'retest']
    out = {}
    for d in ['npz', 'npz_uniform']:
        with ProcessPoolExecutor(24) as ex:
            O = np.stack(list(ex.map(occ, [(i, d) for i in S['img_id']], chunksize=16)))
        peak = O.argmax(2).astype(float)
        rows = []
        for j, v in enumerate(VNAMES):
            dev = peak[:, j] - ANCHORS[j]
            rows.append(dict(geometry=d, vertebra=v, anchor=int(ANCHORS[j]), mean_dev=dev.mean(), sd=dev.std(),
                             mean_abs_dev=np.abs(dev).mean(), within_5=float(np.mean(np.abs(dev) < 5))))
        out[d] = pd.DataFrame(rows)
        np.save(ROOT/f'occupancy_{d}.npy', O.mean(0)); np.save(ROOT/f'occupancy_sd_{d}.npy', O.std(0))
    R = pd.concat(out.values()); R.to_csv(ROOT/'tables'/'level_alignment.csv', index=False)
    print(R.round(2).to_string())
