import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import *
import pickle, gseapy
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

# analyte -> gene symbol (proteoforms collapse to their gene)
ALIAS = {'Oligo-SNCA': 'SNCA', 'pTau-181': 'MAPT', 'pTau-217': 'MAPT', 'pTau-231': 'MAPT', 'APOE4': 'APOE', 'pTDP43-409': 'TARDBP',
         'IL12p70': 'IL12B', 'CXCL8': 'CXCL8'}
LIBS = ['GO_Biological_Process_2023', 'Reactome_2022', 'KEGG_2021_Human']
MIN_SIZE, MAX_FRAC, B = 5, 0.5, 5000

R = pickle.load(open(ROOT/'proteomics_resid.pkl', 'rb'))
PR = pd.read_csv(TAB/'proteomics.csv')
prot = list(PR['protein']); gene = [ALIAS.get(p, p) for p in prot]
genes = sorted(set(gene)); gi = {g: i for i, g in enumerate(genes)}
M = np.zeros((len(genes), len(prot)))
for j, g in enumerate(gene): M[gi[g], j] = 1
M /= M.sum(1, keepdims=True)                              # gene statistic = mean over its analytes


def rhos(perm):
    out = np.empty(len(prot))
    for j, p in enumerate(prot):
        g_, v_ = R['RES'][p]
        gp = R['rg'][perm] if R['complete'][p] else g_[np.random.default_rng(perm[0]*7919 + j).permutation(len(g_))]
        if perm is IDENT: gp = g_
        a, b = rankdata(gp), rankdata(v_)
        out[j] = np.corrcoef(a, b)[0, 1]
    return out


IDENT = np.arange(len(R['rg']))
obs_p = M @ rhos(IDENT)
rng = np.random.default_rng(SEED)
null_p = np.array([M @ rhos(rng.permutation(len(R['rg']))) for _ in range(B)])

gdir = ROOT/'genesets'; gdir.mkdir(exist_ok=True)
rows = []
for lib in LIBS:
    f = gdir/f'{lib}.pkl'
    if not f.exists(): pickle.dump(gseapy.get_library(lib, organism='Human'), open(f, 'wb'))
    GS = pickle.load(open(f, 'rb'))
    for term, mem in GS.items():
        idx = [gi[g] for g in set(mem) if g in gi]
        if len(idx) < MIN_SIZE or len(idx) > MAX_FRAC*len(genes): continue
        # competitive statistic: mean gene rho in set minus panel mean; null by permuting participants
        s_obs = obs_p[idx].mean() - obs_p.mean(); s_null = null_p[:, idx].mean(1) - null_p.mean(1)
        p = (1 + (np.abs(s_null) >= abs(s_obs)).sum())/(B + 1)
        rows.append(dict(library=lib, term=term, size=len(idx), genes=';'.join(sorted(genes[i] for i in idx)), mean_rho=obs_p[idx].mean(),
                         score=s_obs, z=s_obs/s_null.std(), p=p))
E = pd.DataFrame(rows).drop_duplicates(['library', 'genes'])
E['q_library'] = E.groupby('library')['p'].transform(lambda p: multipletests(p, method='fdr_bh')[1])
E['q_all'] = multipletests(E['p'], method='fdr_bh')[1]
E = E.sort_values('p'); E.to_csv(TAB/'proteomics_enrichment.csv', index=False)

# cross-check: preranked GSEA on gene-level rho restricted to the panel (gene permutation)
rnk = pd.Series(obs_p, index=genes).sort_values(ascending=False)
pre = []
for lib in LIBS:
    GS = pickle.load(open(gdir/f'{lib}.pkl', 'rb'))
    gs = {k: [g for g in v if g in gi] for k, v in GS.items()}
    gs = {k: v for k, v in gs.items() if MIN_SIZE <= len(v) <= MAX_FRAC*len(genes)}
    res = gseapy.prerank(rnk=rnk, gene_sets=gs, permutation_num=5000, min_size=MIN_SIZE, max_size=200, seed=SEED, outdir=None, verbose=False, threads=8)
    d = res.res2d.assign(library=lib); pre.append(d)
PRE = pd.concat(pre); PRE.to_csv(TAB/'proteomics_gsea_prerank.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print('genes', len(genes), 'sets tested', len(E), E.groupby('library').size().to_dict())
print(E[['library', 'term', 'size', 'mean_rho', 'z', 'p', 'q_library', 'q_all']].head(20).round(4).to_string())
print(PRE.sort_values('FDR q-val')[['library', 'Term', 'NES', 'NOM p-val', 'FDR q-val']].head(10).to_string())
