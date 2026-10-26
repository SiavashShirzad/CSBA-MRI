import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
import c05_calibrate as cal

cal.calibrate('harm')
from batches import with_acquisition, batch_of
out = {}
for tag in ['main', 'harm']:
    Pq = with_acquisition(tag); Pq['batch'] = Pq.apply(batch_of, axis=1, level='vendor_field')
    h = Pq[(Pq['group'] == 'HC') & Pq['g_spine'].notna()]
    m = smf.mixedlm('g_spine ~ age', h, groups=h['batch']).fit()
    vb, ve = float(m.cov_re.iloc[0, 0]), float(m.scale)
    t = Pq[Pq['role'] == 'test']
    out[tag] = dict(test_mae_raw=float((t['raw_spine'] - t['age']).abs().mean()), test_mae_cal=float(t['g_spine'].abs().mean()),
                    test_r=float(np.corrcoef(t['age'], t['raw_spine'])[0, 1]), batch_icc=vb/(vb + ve), n_healthy=len(h), n_test=len(t))
a = out['harm']['test_mae_raw'] <= out['main']['test_mae_raw'] + 0.25
b = out['harm']['batch_icc'] < out['main']['batch_icc']
out['rule'] = dict(a_mae_ok=bool(a), b_batch_lower=bool(b), primary='harm' if (a and b) else 'main')
json.dump(out, open(TAB/'harm_decision.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
