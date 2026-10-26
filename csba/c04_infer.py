import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--variant', default='main'); ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--gpu', default='0'); ap.add_argument('--maps', action='store_true')
ap.add_argument('--perturb', default=''); ap.add_argument('--model_dir', default='')
A = ap.parse_args()
VAR = VARIANTS[A.variant]; DEV = f'cuda:{A.gpu}'

S = pd.read_csv(ROOT/'splits.csv')
npz_dir = VAR['npz'] if not A.perturb else f'npz_{A.perturb}'
S['npz'] = [str(ROOT/npz_dir/f'{i}.npz') for i in S['img_id']]
S = S[[os.path.exists(p) for p in S['npz']]].reset_index(drop=True)
ck = torch.load(ROOT/'models'/(A.model_dir or A.variant)/f'fold{A.fold}.pth', map_location=DEV, weights_only=False)
model = DenseAgeNet().to(DEV); model.load_state_dict(ck['model']); model.eval()
dl = DataLoader(SpineDS(S, False, ck['std'], VAR), batch_size=6, shuffle=False, num_workers=12,
                pin_memory=True, collate_fn=collate)
rows, maps, i = [], [], 0
with torch.no_grad():
    for x, L, y in dl:
        x, L = x.to(DEV), L.to(DEV)
        with torch.amp.autocast('cuda'):
            am, lv = model(x)
        am, lv = am.float(), lv.float()
        aw, au, sg, mk = region_ages(am, lv, L)
        sigma = torch.exp(0.5*lv)
        for b in range(x.shape[0]):
            r = dict(img_id=S.loc[i, 'img_id'])
            for k in REGIONS:
                r[f'a_{k}'] = float(aw[k][b]); r[f'au_{k}'] = float(au[k][b]); r[f'se_{k}'] = float(sg[k][b])
                m = mk[k][b]; r[f'sm_{k}'] = float((sigma[b]*m).sum()/(m.sum()+1e-6))
            rows.append(r)
            if A.maps:
                sm = (mk['spine'][b, 0] > 0.05).float()
                maps.append(torch.where(sm > 0, am[b, 0], torch.full_like(am[b, 0], float('nan'))).half().cpu().numpy())
            i += 1
tag = (A.model_dir or A.variant) + (f"_{A.perturb.removeprefix('n4_')}" if A.perturb else '')
(ROOT/'preds').mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(ROOT/'preds'/f'{tag}_f{A.fold}.csv', index=False)
if A.maps:
    np.save(ROOT/'preds'/f'{tag}_f{A.fold}_maps.npy', np.stack(maps))
print('done', tag, A.fold, len(rows), flush=True)
