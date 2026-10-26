import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

MODE = sys.argv[1] if len(sys.argv) > 1 else 'main'
UNIFORM = MODE == 'uniform'; JITTER = 'jitter' in MODE; N4 = 'n4' in MODE
OUT = ROOT/('npz' if MODE == 'main' else f'npz_{MODE}'); OUT.mkdir(exist_ok=True)


def work(a):
    iid, img, seg = a
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    p = OUT/f'{iid}.npz'
    try:
        out, status = extract(img, seg, uniform=UNIFORM, jitter_seed=int(iid[:8], 16) if JITTER else None, n4=N4)
        if out is None: return dict(img_id=iid, ok=False, reason=status)
        soft, lab, meta = out
        np.savez_compressed(p, image=soft, compartments=lab, anchors=json.dumps(meta['anchors']))
        return dict(img_id=iid, ok=True, **{k: v for k, v in meta.items() if k != 'anchors'})
    except Exception as e:
        return dict(img_id=iid, ok=False, reason=f'error {type(e).__name__}')


if __name__ == '__main__':
    C = pd.read_csv(ROOT/'cohort_images.csv')
    C = C[C['exclusion'].isna()]
    logf = ROOT/('extraction.csv' if MODE == 'main' else f'extraction_{MODE}.csv')
    prev = pd.read_csv(logf) if logf.exists() else pd.DataFrame(columns=['img_id','ok'])
    done = set(prev.loc[prev['ok'] == True, 'img_id'])
    C = C[C['seg_file'].notna()]
    if JITTER:
        S = pd.read_csv(ROOT/'splits.csv'); C = C[C['img_id'].isin(S.loc[S['role'] != 'retest', 'img_id'])]
    jobs = [(r.img_id, r.img, r.seg_file) for r in C.itertuples() if r.img_id not in done]
    print('to extract', len(jobs), '| already done', len(done), flush=True)
    t0, res = time.time(), []
    with ProcessPoolExecutor(max_workers=48) as ex:
        fut = [ex.submit(work, j) for j in jobs]
        for n, f in enumerate(as_completed(fut), 1):
            res.append(f.result())
            if n % 200 == 0: print(f'{n}/{len(jobs)} {(time.time()-t0)/60:.1f} min', flush=True)
    R = pd.concat([prev[prev['ok'] == True], pd.DataFrame(res)], ignore_index=True).drop_duplicates('img_id', keep='last')
    R['ok'] = R['ok'].astype(bool)
    R.to_csv(logf, index=False)
    print('ok', int(R['ok'].sum()), '/', len(R), '| failures', R.loc[~R['ok'], 'reason'].value_counts().to_dict() if 'reason' in R else {})
