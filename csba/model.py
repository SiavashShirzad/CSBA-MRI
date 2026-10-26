import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

PCTS = np.array([1,10,20,30,40,50,60,70,80,90,99])
REF_VERT, REF_CSF = 0.35, 0.85
AGE_MID, AGE_HALF = 55.0, 45.0
LOGVAR_MIN, LOGVAR_MAX = -4.0, 5.0
OUTSHAPE = (N_SLICES//2, XY//2, XY//2)

VARIANTS = {
    'main':       dict(npz='npz',         anchor=True,  mask_marrow=False, loss='dense'),
    'global':     dict(npz='npz',         anchor=True,  mask_marrow=False, loss='global'),
    'uniform':    dict(npz='npz_uniform', anchor=True,  mask_marrow=False, loss='dense'),
    'noanchor':   dict(npz='npz',         anchor=False, mask_marrow=False, loss='dense'),
    'marrowmask': dict(npz='npz',         anchor=False, mask_marrow=True,  loss='dense'),
    'harm':       dict(npz='npz_n4',      anchor=True,  mask_marrow=False, loss='dense', acq_aug=True),
}


def thick_lr(img, mm):
    x = np.arange(img.shape[2])*XY_MM
    c = np.arange(x[0] + mm/2 + random.uniform(0, mm), x[-1], mm)
    if len(c) < 2: return img
    W = (np.abs(x[None, :] - c[:, None]) <= mm/2).astype(np.float32); W /= np.maximum(W.sum(1, keepdims=True), 1)
    slab = np.einsum('kx,zyx->zyk', W, img)
    idx = np.clip(np.searchsorted(c, x) - 1, 0, len(c) - 2); w = np.clip((x - c[idx])/(c[idx + 1] - c[idx]), 0, 1)
    return (slab[..., idx]*(1 - w) + slab[..., idx + 1]*w).astype(np.float32)


def nyul_fit(paths):
    L = []
    for p in paths:
        d = np.load(p); img, lab = d['image'], d['compartments']
        m = np.isin(lab, SPINE)
        if m.sum() < 500: continue
        lm = np.percentile(img[m], PCTS)
        if lm[-1] - lm[0] < 1e-6: continue
        L.append((lm - lm[0]) / (lm[-1] - lm[0]))
    return np.maximum.accumulate(np.array(L).mean(0))


def harmonize(img, lab, std, anchor=True, mask_marrow=False):
    if mask_marrow:
        img = np.where(np.isin(lab, VERTS), 0.0, img).astype(np.float32)
        m = np.isin(lab, [CORD, CSF] + DISCS)
    else:
        m = np.isin(lab, SPINE)
    if m.sum() >= 500:
        lm = np.percentile(img[m], PCTS)
        if lm[-1] - lm[0] > 1e-6:
            img = np.interp(img, np.maximum.accumulate(lm), std).astype(np.float32)
    if mask_marrow:
        img = np.where(np.isin(lab, VERTS), 0.0, img)
    if anchor:
        v, c = img[np.isin(lab, VERTS)], img[lab == CSF]
        if v.size >= 200 and c.size >= 200:
            lo, hi = np.median(v), np.median(c)
            if hi - lo > 1e-3:
                img = np.clip((img-lo)*((REF_CSF-REF_VERT)/(hi-lo)) + REF_VERT, 0, 1.5)
    return np.clip(img, 0, 1.5).astype(np.float32)


class SpineDS(Dataset):
    def __init__(self, df, train, std, var):
        self.df = df.reset_index(drop=True); self.train = train; self.std = std; self.var = var
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        d = np.load(r['npz'])
        img = harmonize(d['image'].astype(np.float32), d['compartments'], self.std,
                        self.var['anchor'], self.var['mask_marrow'])
        lab = d['compartments'].astype(np.int16)
        if self.train:
            if random.random() < 0.5:
                a = random.uniform(-5, 5)
                img = ndi.rotate(img, a, axes=(0,1), reshape=False, order=1, mode='nearest')
                lab = ndi.rotate(lab, a, axes=(0,1), reshape=False, order=0, mode='nearest')
            if random.random() < 0.5:
                img, lab = img[:,:,::-1].copy(), lab[:,:,::-1].copy()
            if random.random() < 0.4:
                s = random.randint(-2, 2); img = np.roll(img, s, 0); lab = np.roll(lab, s, 0)
            if random.random() < 0.5:
                bf = 0.30 if self.var.get('acq_aug') else 0.20
                f = ndi.zoom(np.random.randn(4,4,4).astype(np.float32)*bf, np.array(img.shape)/4, order=1)
                img = img * (1 + np.resize(f, img.shape))
            if self.var.get('acq_aug'):
                if random.random() < 0.3: img = thick_lr(img, random.uniform(2, 5))
                if random.random() < 0.2: img = ndi.gaussian_filter(img, (0, random.uniform(.3, 1.), random.uniform(.3, 1.))).astype(np.float32)
        t = torch.from_numpy(np.ascontiguousarray(img)).float()
        if self.train:
            if random.random() < 0.7: t = t*random.uniform(.82,1.18) + random.uniform(-.08,.08)
            if random.random() < 0.5:
                g = random.uniform(.78,1.30); mx = t.max()
                if mx > 0: t = (t.clamp(min=0)/mx)**g*mx
            if random.random() < 0.4: t = t + torch.randn_like(t)*random.uniform(.01,.04)
        t = t.clamp(0, 1.5)
        L = torch.from_numpy(np.ascontiguousarray(lab).astype(np.int64))
        return t[None], L, torch.tensor(float(r['age']))


def collate(b):
    x, l, a = zip(*b); return torch.stack(x), torch.stack(l), torch.stack(a)


def balanced_sampler(df, w_clip=6.0, bw=5.0):
    a = df['age'].values
    e = np.arange(np.floor(a.min()), np.ceil(a.max())+bw, bw)
    b = np.clip(np.digitize(a, e)-1, 0, len(e)-2)
    cnt = np.bincount(b, minlength=len(e)-1).astype(float)
    w = np.where(cnt > 0, 1/np.maximum(cnt,1), 0)[b]
    w = np.clip(w/w.mean(), None, w_clip)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(df), True)


class Blk(nn.Module):
    def __init__(self, ci, co, st=1):
        super().__init__()
        self.c1 = nn.Conv3d(ci,co,3,st,1,bias=False); self.b1 = nn.InstanceNorm3d(co, affine=True)
        self.c2 = nn.Conv3d(co,co,3,1,1,bias=False);  self.b2 = nn.InstanceNorm3d(co, affine=True)
        self.sc = nn.Sequential() if (st==1 and ci==co) else \
                  nn.Sequential(nn.Conv3d(ci,co,1,st,bias=False), nn.InstanceNorm3d(co, affine=True))
    def forward(self, x):
        o = F.relu(self.b1(self.c1(x))); o = self.b2(self.c2(o))
        return F.relu(o + self.sc(x))


class DenseAgeNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1,c,7,stride=(2,2,2),padding=3,bias=False),
                                  nn.InstanceNorm3d(c, affine=True), nn.ReLU(inplace=True))
        self.e1 = Blk(c, c, 1); self.e2 = Blk(c, c*2, (2,2,2))
        self.e3 = Blk(c*2, c*4, (2,1,1)); self.e4 = Blk(c*4, c*8, (2,1,1))
        self.d3 = Blk(c*8 + c*4, c*4, 1); self.d2 = Blk(c*4 + c*2, c*2, 1); self.d1 = Blk(c*2 + c, c, 1)
        self.head = nn.Conv3d(c, 2, 1)
        nn.init.zeros_(self.head.bias)
    def forward(self, x):
        s = self.stem(x)
        e1 = self.e1(s); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        up = lambda a, b: F.interpolate(a, size=b.shape[2:], mode='trilinear', align_corners=False)
        u3 = self.d3(torch.cat([up(e4, e3), e3], 1))
        u2 = self.d2(torch.cat([up(u3, e2), e2], 1))
        u1 = self.d1(torch.cat([up(u2, e1), e1], 1))
        o = self.head(u1)
        age = AGE_MID + AGE_HALF*torch.tanh(o[:,0:1]/AGE_HALF)
        logvar = LOGVAR_MIN + (LOGVAR_MAX-LOGVAR_MIN)*torch.sigmoid(o[:,1:2])
        return age, logvar


def masks_from_labels(L, size):
    out = {}
    for nm, ids in [('spine', SPINE), ('cord',[CORD]), ('csf',[CSF]), ('disc',DISCS),
                    *[(v, [vid]) for v, vid in zip(VNAMES, VERTS)]]:
        m = torch.isin(L, torch.tensor(ids, device=L.device)).float().unsqueeze(1)
        out[nm] = F.interpolate(m, size=size, mode='area')
    return out


def dense_loss(age_map, logvar, L, y, mode='dense', w_global=1.0, w_smooth=0.02):
    m = masks_from_labels(L, age_map.shape[2:])['spine']
    m = (m > 0.05).float() * m
    denom = m.sum((1,2,3,4)) + 1e-5
    if mode == 'global':
        agg = (age_map * m).sum((1,2,3,4)) / denom
        return F.smooth_l1_loss(agg, y)
    yv = y.view(-1,1,1,1,1)
    prec = torch.exp(-logvar)
    nll = 0.5*(logvar + (age_map - yv)**2 * prec)
    l_vox = ((nll * m).sum((1,2,3,4)) / denom).mean()
    w = m * prec
    agg = (age_map * w).sum((1,2,3,4)) / (w.sum((1,2,3,4)) + 1e-5)
    l_glob = F.smooth_l1_loss(agg, y)
    dz = (age_map[:,:,1:] - age_map[:,:,:-1]).abs().mean()
    dy = (age_map[:,:,:,1:] - age_map[:,:,:,:-1]).abs().mean()
    dx = (age_map[:,:,:,:,1:] - age_map[:,:,:,:,:-1]).abs().mean()
    return l_vox + w_global*l_glob + w_smooth*(dz+dy+dx)


@torch.no_grad()
def region_ages(age_map, logvar, L):
    mk = masks_from_labels(L, age_map.shape[2:])
    prec = torch.exp(-logvar); sig = torch.exp(0.5*logvar)
    aw, au, sg = {}, {}, {}
    for nm in REGIONS:
        w = mk[nm] * prec; s = w.sum((1,2,3,4))
        aw[nm] = torch.where(s > 1e-4, (age_map*w).sum((1,2,3,4))/(s+1e-6), torch.full_like(s, float('nan')))
        u = mk[nm]; s2 = u.sum((1,2,3,4))
        au[nm] = torch.where(s2 > 1e-4, (age_map*u).sum((1,2,3,4))/(s2+1e-6), torch.full_like(s2, float('nan')))
        sg[nm] = torch.where(s > 1e-4, 1/torch.sqrt(s/(s2+1e-6)+1e-9), torch.full_like(s, float('nan')))
    return aw, au, sg, mk
