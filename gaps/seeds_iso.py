"""Seed check under the shape-free isotonic crossing rule, next to the logistic rule.

Uses the BLiMP accuracies already scored by the earlier surprisal run (the matched-word
inputs): 12 BLiMP paradigms, 400 pairs each, 22 checkpoints, Pythia 14M to 410M with the
original run and nine PolyPythias seeds per size (a seed sets both initialization and data
order), plus the 160M weight-only and data-only seeds. Crossings use the main panel's rule
(logistic with floor 0.5, criterion 0.70, two-log-unit guard), on checkpoints from step 128
(primary) or step 16. Adds the isotonic rule used in the main panel's window
check and a common-cell comparison of crossings across windows. Writes seeds_iso_results.json.
"""
import glob, gzip, json, os, re, traceback
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.isotonic import IsotonicRegression

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
    with open(os.path.join(OUT, 'seeds_iso_results.json'), 'w') as f:
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


def cross_iso(x, y):
    if y.max() < CRIT:
        return 'notreached', np.nan
    f = IsotonicRegression(increasing=True).fit_transform(x, y)
    j = int(np.argmax(f >= CRIT))
    if f[j] < CRIT:
        return 'notreached', np.nan
    if j == 0:
        return 'before', np.nan
    x0, x1, y0, y1 = x[j - 1], x[j], f[j - 1], f[j]
    return 'observed', float(x0 + (CRIT - y0) / (y1 - y0) * (x1 - x0))


RULES = {'logistic': cross, 'isotonic': cross_iso}


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


def curves(min_step, fn):
    out = []
    for (run, size, kind, seed, par), g in D[D.step >= min_step].groupby(['run', 'size', 'kind', 'seed', 'paradigm']):
        g = g.sort_values('step')
        st, xc = fn(np.log(g.step.values * TOK), g.acc.values)
        out.append(dict(run=run, size=size, kind=kind, seed=seed, paradigm=par, status=st, x=xc))
    return pd.DataFrame(out)


def analyze(C):
    res = dict(status_counts=C.status.value_counts().to_dict())
    main = C[C.kind == 'seed']
    # one panel per seed: five sizes as units
    per_seed = {}
    for s in range(10):
        P = panel(main[main.seed == s])
        if P['size'].nunique() == 5 and P.paradigm.nunique() >= 4:
            per_seed[s] = fit_panel(P, 'size')
    res['per_seed'] = per_seed
    st = {s: [v['stretch'][z] for z in SIZES] for s, v in per_seed.items()}
    res['per_seed_summary'] = dict(
        n_seeds=len(per_seed),
        stretch_14m_gt_410m=int(sum(v[0] > v[4] for v in st.values())),
        stretch_14m_31m_mean_gt_160m_410m_mean=int(sum(np.mean(v[:2]) > np.mean(v[3:]) for v in st.values())),
        spearman_stretch_logparams=[float(stats.spearmanr([LOGN[z] for z in SIZES], v)[0]) for v in st.values()],
        gain=[v['gain'] for v in per_seed.values()],
        mean_stretch={z: float(np.mean([v[i] for v in st.values()])) for i, z in enumerate(SIZES)},
        sd_stretch={z: float(np.std([v[i] for v in st.values()], ddof=1)) for i, z in enumerate(SIZES)})
    # pooled: every run of the main seed family as its own unit
    P = panel(main)
    pooled = fit_panel(P, 'run')
    runs = sorted(pooled['stretch'])
    size_of = {r_: r_.split('-')[0] for r_ in runs}
    rng = np.random.default_rng(0)
    res['pooled'] = dict(cells=pooled['cells'], events=pooled['events'], rho=pooled['rho'], gain=pooled['gain'],
                         log_stretch_between_size=between_share([np.log(max(pooled['stretch'][r_], 1e-6)) for r_ in runs], [size_of[r_] for r_ in runs], rng),
                         log_shift_between_size=between_share([pooled['log_shift'][r_] for r_ in runs], [size_of[r_] for r_ in runs], rng),
                         stretch_by_size={z: [float(np.mean([pooled['stretch'][r_] for r_ in runs if size_of[r_] == z])),
                                              float(np.std([pooled['stretch'][r_] for r_ in runs if size_of[r_] == z], ddof=1))] for z in SIZES},
                         nonpositive_stretches=int(sum(pooled['stretch'][r_] <= 0 for r_ in runs)))
    return res


CURVES = {}
for rule, fn in RULES.items():
    for min_step in [128, 16]:
        key = f'{rule}_from_step_{min_step}'
        try:
            C = curves(min_step, fn)
            CURVES[(rule, min_step)] = C
            RES[key] = analyze(C)
            print(key, RES[key]['per_seed_summary'], RES[key]['pooled'], flush=True)
        except Exception:
            RES[key] = {'error': traceback.format_exc()}
            print(RES[key]['error'], flush=True)
        save()

# crossings observed in both windows: are they the same, and do they carry the stretch?
for rule in RULES:
    key = f'{rule}_common_cells'
    try:
        a, b = CURVES[(rule, 128)], CURVES[(rule, 16)]
        k = ['run', 'size', 'kind', 'seed', 'paradigm']
        M = a[a.status == 'observed'].merge(b[b.status == 'observed'], on=k, suffixes=('_128', '_16'))
        d = M.x_16 - M.x_128
        res = dict(cells=int(len(M)), identical_share=float(np.mean(np.abs(d) < 1e-9)),
                   mean_shift=float(d.mean()), mean_abs_shift=float(np.abs(d).mean()))
        for w in ['128', '16']:
            Cw = M[k].assign(status='observed', x=M['x_' + w])
            res['from_step_' + w] = analyze(Cw)
        # cells only the step-16 window dates, by size
        only16 = b[b.status == 'observed'].merge(a[a.status == 'observed'][k], on=k, how='left', indicator=True)
        only16 = only16[only16._merge == 'left_only']
        res['only_from_16_by_size'] = {z: int((only16[only16.kind == 'seed']['size'] == z).sum()) for z in SIZES}
        RES[key] = res
        print(key, {kk: v for kk, v in res.items() if not kk.startswith('from_step')}, flush=True)
        for w in ['128', '16']:
            print(' ', w, res['from_step_' + w]['per_seed_summary'], res['from_step_' + w]['pooled'], flush=True)
    except Exception:
        RES[key] = {'error': traceback.format_exc()}
        print(RES[key]['error'], flush=True)
    save()
