"""Why does the seed-panel stretch shrink when checkpoints before step 128 are added?

Separates selection (which cells are retained) from measurement (where each cell's crossing lands)
by refitting the pooled 50-run seed table on matched cell sets across checkpoint windows. Reuses the
seed-panel setup below.

Uses the BLiMP accuracies already scored by the earlier surprisal run (the matched-word
inputs): 12 BLiMP paradigms, 400 pairs each, 22 checkpoints, Pythia 14M to 410M with the
original run and nine PolyPythias seeds per size (a seed sets both initialization and data
order), plus the 160M weight-only and data-only seeds. Crossings use the main panel's rule
(logistic with floor 0.5, criterion 0.70, two-log-unit guard), on checkpoints from step 128
(primary) or step 16. Writes window_results.json.
"""
import glob, gzip, json, os, re, traceback
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
TOK = 1024 * 2049.0
CRIT = 0.70
SIZES = ['14m', '31m', '70m', '160m', '410m']
LOGN = {'14m': np.log(14e6), '31m': np.log(31e6), '70m': np.log(70e6), '160m': np.log(160e6), '410m': np.log(410e6)}
RES = {}


def find(name):
    for root in ['/kaggle/input', '.']:
        for cand in (name, name + '.gz'):
            hits = sorted(glob.glob(os.path.join(root, '**', cand), recursive=True))
            if hits:
                return hits[0]
    raise FileNotFoundError(f'{name} not found under /kaggle/input or cwd; attach the matched-word and gaps input datasets')


def save():
    with open(os.path.join(OUT, 'window_results.json'), 'w') as f:
        json.dump(RES, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def read_rows(name):
    p = find(name)
    op = gzip.open if p.endswith('.gz') else open
    with op(p, 'rt') as f:
        return [json.loads(l) for l in f]


def run_label(model):
    m = re.match(r'EleutherAI/pythia-(\d+m)(?:-(weight-seed|data-seed|seed)(\d+))?$', model)
    if not m:
        return None
    size, kind, k = m.groups()
    if kind is None:
        return size, 'seed', 0
    return size, kind, int(k)


def logistic(x, L, x0, k):
    return 0.5 + (L - 0.5) / (1 + np.exp(-k * (x - x0)))


def cross(x, y):
    if y.max() < CRIT:
        return 'notreached', np.nan
    try:
        p, _ = curve_fit(logistic, x, y, p0=[max(y.max(), 0.6), x.mean(), 1.0],
                         bounds=([0.5, x.min() - 5, 0.01], [1.0, x.max() + 5, 20]), maxfev=20000)
    except Exception:
        return 'error', np.nan
    L, x0, k = p
    if L <= CRIT:
        return 'ceiling', np.nan
    xc = x0 - np.log((L - 0.5) / (CRIT - 0.5) - 1) / k
    if xc < x.min() - 2:
        return 'before', np.nan
    if xc > x.max() + 2:
        return 'after', np.nan
    return 'observed', xc


def lsfit(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b, X @ b


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def additive(u, e, y):
    nu, ne = u.max() + 1, e.max() + 1
    E = np.zeros((len(y), ne))
    E[np.arange(len(y)), e] = 1
    X = np.hstack([np.eye(nu)[u], E[:, 1:]])
    b, p = lsfit(X, y)
    return b[:nu], np.r_[0.0, b[nu:]], p


def stretch(u, e, y, sweeps=500):
    nu, ne = u.max() + 1, e.max() + 1
    a, b, _ = additive(u, e, y)
    a, b, g = a.copy(), b.copy(), np.ones(nu)
    idx = [np.where(u == i)[0] for i in range(nu)]
    cnt = np.bincount(u, minlength=nu)
    for _ in range(sweeps):
        for i in range(nu):
            m = idx[i]
            a[i], g[i] = np.linalg.lstsq(np.c_[np.ones(len(m)), b[e[m]]], y[m], rcond=None)[0]
        b = np.bincount(e, weights=g[u] * (y - a[u]), minlength=ne) / np.bincount(e, weights=g[u] ** 2, minlength=ne)
        sc = b.std(ddof=1)
        b = (b - b.mean()) / sc
        g = g * sc
        a = np.bincount(u, weights=y - g[u] * b[e], minlength=nu) / cnt
    return a, g, b, a[u] + g[u] * b[e]


def panel(C):
    P = C[C.status == 'observed']
    return P[P.paradigm.map(P.paradigm.value_counts()) >= 2].reset_index(drop=True)


def fit_panel(P, unit_col):
    u, ul = pd.factorize(P[unit_col].values)
    e, _ = pd.factorize(P.paradigm.values)
    y = P.x.values
    al, _, p = additive(u, e, y)
    _, g, _, ps = stretch(u, e, y)
    return dict(cells=int(len(y)), events=int(e.max() + 1), rho=float(np.corrcoef(y, p)[0, 1]), r2=r2(y, p),
                stretch_r2=r2(y, ps), gain=r2(y, ps) - r2(y, p),
                log_shift={str(l): float(v) for l, v in zip(ul, al - al.mean())},
                stretch={str(l): float(v) for l, v in zip(ul, g / g.mean())})


def between_share(vals, groups, rng, nperm=10000):
    vals, groups = np.asarray(vals, float), np.asarray(groups)

    def share(gr):
        tot = ((vals - vals.mean()) ** 2).sum()
        within = sum(((vals[gr == k] - vals[gr == k].mean()) ** 2).sum() for k in np.unique(gr))
        return 1 - within / tot
    obs = share(groups)
    null = np.array([share(rng.permutation(groups)) for _ in range(nperm)])
    return dict(share=float(obs), p=float((np.sum(null >= obs) + 1) / (nperm + 1)))


rows = read_rows('results_twins-cpuall.jsonl') + read_rows('results_twins-cpu31m.jsonl')
recs = []
for r in rows:
    lab = run_label(r['model'])
    if lab is None or r['step'] <= 0:
        continue
    for task, acc in r['blimp'].items():
        recs.append((lab[0], lab[1], lab[2], r['step'], task, float(acc)))
D = pd.DataFrame(recs, columns=['size', 'kind', 'seed', 'step', 'paradigm', 'acc']).drop_duplicates(['size', 'kind', 'seed', 'step', 'paradigm'])
D['run'] = D['size'] + '-' + D['kind'] + D['seed'].astype(str)
print(D.groupby(['size', 'kind']).seed.nunique().to_dict(), D.step.nunique(), flush=True)
RES['runs'] = {f'{s}-{k}': int(n) for (s, k), n in D.groupby(['size', 'kind']).seed.nunique().items()}


def curves(min_step):
    out = []
    for (run, size, kind, seed, par), g in D[D.step >= min_step].groupby(['run', 'size', 'kind', 'seed', 'paradigm']):
        g = g.sort_values('step')
        st, xc = cross(np.log(g.step.values * TOK), g.acc.values)
        out.append(dict(run=run, size=size, kind=kind, seed=seed, paradigm=par, status=st, x=xc))
    return pd.DataFrame(out)



def pooled_share(P, rng):
    f = fit_panel(P, 'run')
    runs = sorted(f['stretch'])
    size_of = {r_: r_.split('-')[0] for r_ in runs}
    ls = between_share([np.log(max(f['stretch'][r_], 1e-6)) for r_ in runs], [size_of[r_] for r_ in runs], rng, 5000)
    sh = between_share([f['log_shift'][r_] for r_ in runs], [size_of[r_] for r_ in runs], rng, 5000)
    return dict(cells=f['cells'], events=f['events'], rho=f['rho'], gain=f['gain'], log_stretch_share=ls, shift_share=sh,
                stretch_by_size={z: float(np.mean([f['stretch'][r_] for r_ in runs if size_of[r_] == z])) for z in SIZES})


rng = np.random.default_rng(0)
C = {m: curves(m)[lambda d: d.kind == 'seed'].reset_index(drop=True) for m in [16, 32, 64, 128]}
key = ['run', 'paradigm']
try:
    RES['by_window'] = {}
    for m, Cm in C.items():
        RES['by_window'][m] = pooled_share(panel(Cm), rng)
        RES['by_window'][m]['status_by_size'] = {z: Cm[Cm['size'] == z].status.value_counts().to_dict() for z in SIZES}
        print(m, RES['by_window'][m], flush=True)
    save()
    o16 = C[16][C[16].status == 'observed'][key + ['size', 'x']]
    o128 = C[128][C[128].status == 'observed'][key + ['size', 'x']]
    both = o16.merge(o128, on=key + ['size'], suffixes=('16', '128'))
    m = C[128][key + ['status']].merge(o16, on=key)
    RES['transitions'] = {z: m[m['size'] == z].status.value_counts().to_dict() for z in SIZES}
    RES['common_cells'] = dict(n=int(len(both)), pearson=float(np.corrcoef(both.x16, both.x128)[0, 1]),
                              mean_shift_by_size={z: float((both.x16 - both.x128)[both['size'] == z].mean()) for z in SIZES})
    variants = {
        'x16_on_cells_observed_at_128': both.rename(columns={'x16': 'x'}).drop(columns='x128'),
        'x128_on_cells_observed_at_16': both.rename(columns={'x128': 'x'}).drop(columns='x16'),
        'x16_dropping_crossings_before_step_128': o16[o16.x >= np.log(128 * TOK)],
    }
    RES['variants'] = {}
    for name, V in variants.items():
        V = V.assign(seed=V.run.str.extract(r'(\d+)$')[0].astype(int))
        RES['variants'][name] = pooled_share(panel(V.assign(status='observed')), rng)
        print(name, RES['variants'][name], flush=True)
except Exception:
    RES['error'] = traceback.format_exc()
    print(RES['error'], flush=True)
save()
