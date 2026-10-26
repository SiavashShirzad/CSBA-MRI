import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import *
import argparse, time

ap = argparse.ArgumentParser()
ap.add_argument('--variant', default='main'); ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--gpu', default='0'); ap.add_argument('--epochs', type=int, default=120)
ap.add_argument('--loso', default='');
ap.add_argument('--seed', type=int, default=42); ap.add_argument('--workers', type=int, default=12)
A = ap.parse_args()
VAR = VARIANTS[A.variant]
DEV = f'cuda:{A.gpu}'
torch.manual_seed(A.seed + A.fold); np.random.seed(A.seed + A.fold); random.seed(A.seed + A.fold)

S = pd.read_csv(ROOT/'splits.csv')
S['npz'] = [str(ROOT/VAR['npz']/f'{i}.npz') for i in S['img_id']]
dev = S[S['role'] == 'dev']
if A.loso:
    tr = dev[dev['dataset'] != A.loso].reset_index(drop=True)
    ho = S[(S['dataset'] == A.loso) & S['role'].isin(['dev', 'test'])].reset_index(drop=True)
else:
    tr = dev[dev['fold'] != A.fold].reset_index(drop=True)
    ho = dev[dev['fold'] == A.fold].reset_index(drop=True)
tr = tr[[os.path.exists(p) for p in tr['npz']]].reset_index(drop=True)

OUT = ROOT/'models'/((f'loso_{A.loso}' if A.variant == 'main' else f'{A.variant}_loso_{A.loso}') if A.loso else A.variant); OUT.mkdir(parents=True, exist_ok=True)
std_path = OUT/f'nyul_f{A.fold}.npy'
STD = nyul_fit(tr['npz'])
np.save(std_path, STD)

dl_tr = DataLoader(SpineDS(tr, True, STD, VAR), batch_size=6, sampler=balanced_sampler(tr),
                   num_workers=A.workers, pin_memory=True, drop_last=True, collate_fn=collate,
                   persistent_workers=True)
dl_ho = DataLoader(SpineDS(ho, False, STD, VAR), batch_size=6, shuffle=False,
                   num_workers=A.workers, pin_memory=True, collate_fn=collate, persistent_workers=True)

model = DenseAgeNet().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
WU = 5
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e+1)/WU if e < WU
        else 0.5*(1+math.cos(math.pi*(e-WU)/max(1, A.epochs-WU))))
scaler = torch.amp.GradScaler('cuda')
mode = VAR['loss']


@torch.no_grad()
def monitor():
    model.eval(); P, Y = [], []
    for x, L, y in dl_ho:
        x, L = x.to(DEV), L.to(DEV)
        with torch.amp.autocast('cuda'):
            am, lv = model(x)
        aw, au, _, _ = region_ages(am.float(), lv.float(), L)
        P += (au['spine'] if mode == 'global' else aw['spine']).cpu().tolist(); Y += y.tolist()
    P, Y = np.array(P), np.array(Y)
    return float(np.nanmean(np.abs(P-Y)))


hist, t0 = [], time.time()
print(f'{A.variant} fold {A.fold} | train {len(tr)} | held-out fold {len(ho)} | {DEV}', flush=True)
for ep in range(1, A.epochs+1):
    model.train(); tot = 0.0
    for x, L, y in dl_tr:
        x, L, y = x.to(DEV), L.to(DEV), y.to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            am, lv = model(x)
        loss = dense_loss(am.float(), lv.float(), L, y, mode)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt); scaler.update(); tot += loss.item()
    sched.step()
    rec = dict(ep=ep, loss=tot/len(dl_tr), min=(time.time()-t0)/60)
    if ep % 10 == 0 or ep == A.epochs:
        rec['heldout_mae'] = monitor()
        print(f"ep {ep:03d} loss {rec['loss']:.3f} held-out MAE {rec['heldout_mae']:.2f} | {rec['min']:.0f} min", flush=True)
    hist.append(rec)

torch.save(dict(model=model.state_dict(), variant=A.variant, fold=A.fold, epochs=A.epochs, seed=A.seed,
                train_ids=tr['img_id'].tolist(), std=STD), OUT/f'fold{A.fold}.pth')
json.dump(hist, open(OUT/f'hist_f{A.fold}.json', 'w'))
print('done', flush=True)
