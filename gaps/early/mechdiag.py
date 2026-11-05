"""Where in the real curves does the step-128 window create the Pythia stretch? Real-data diagnostics on the
six-size extended panel (released accuracies from step 128 plus our rescored steps 16 to 64):
  swap_by_size   on cells crossed in both windows, replace one size's step-128 crossings with its step-16
                 crossings and refit, to find which sizes carry the stretch gain
  per_size_slope regress each size's crossings on the common event terms under each window (a per-size
                 stretch), with and without cells whose step-128 crossing lies within one log unit of step 128
  shift_drivers  how far each crossing moves between windows against early accuracy (steps 16 to 64), its
                 distance from the window start and the fitted logistic slopes under both windows
Writes mechdiag_results.json."""

import glob, os, json, time, itertools, traceback
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy import stats
from scipy.linalg import svd as scipy_svd

T0 = time.time()
OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
RES = {}
TOK = 1024 * 2049.0
CRIT = 0.70
SIZES = ['pythia-14m', 'pythia-31m', 'pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b']
SP = ['Hamster', 'Mouse', 'Rat', 'Rabbit', 'Spiny.Mouse', 'Guinea.Pig', 'Ferret', 'Cat', 'Macaque', 'Human']
PRIM = {'Macaque', 'Human'}


def find(name):
    for root in ['/kaggle/input', '.']:
        hits = sorted(glob.glob(os.path.join(root, '**', name), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f'{name} not found under /kaggle/input or cwd; attach the oneclock-gaps-inputs dataset')


def save():
    with open(os.path.join(OUT, 'mechdiag_results.json'), 'w') as f:
        json.dump(RES, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def section(name):
    def wrap(fn):
        t = time.time()
        try:
            RES[name] = fn()
            print(f'[{name}] done in {time.time() - t:.0f}s', flush=True)
        except Exception:
            RES[name] = {'error': traceback.format_exc()}
            print(f'[{name}] FAILED\n{RES[name]["error"]}', flush=True)
        save()
        return fn
    return wrap


def pct(v, q=(2.5, 97.5)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))] if len(v) else [None, None]


# ---------------------------------------------------------------- core fits
def lsfit(X, y):
    try:
        b = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        U, s, Vt = scipy_svd(X, full_matrices=False, lapack_driver='gesvd')
        keep = s > s.max() * 1e-10
        b = Vt[keep].T @ ((U[:, keep].T @ y) / s[keep])
    return b, X @ b


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def design(u, e, extra=None, unit=True, event=True):
    n = len(u)
    B = []
    if unit:
        B.append(np.eye(u.max() + 1)[u])
    if event:
        E = np.zeros((n, e.max() + 1))
        E[np.arange(n), e] = 1
        B.append(E[:, 1:] if unit else E)
    if extra is not None:
        B.append(np.asarray(extra, float).reshape(n, -1))
    return np.hstack(B)


def additive(u, e, y, extra=None):
    """OLS of log time on unit and event dummies; event 0 is the reference."""
    nu, ne = u.max() + 1, e.max() + 1
    b, p = lsfit(design(u, e, extra), y)
    return b[:nu], np.r_[0.0, b[nu:nu + ne - 1]], p, b[nu + ne - 1:]


def unit_only_r2(u, y, extra=None):
    return r2(y, lsfit(design(u, None, extra, event=False), y)[1])


def stretch(u, e, y, sweeps=500, g0=None):
    """log t = a_i + g_i b_e by alternating least squares, as in the release's controls.py."""
    nu, ne = u.max() + 1, e.max() + 1
    a, b, _, _ = additive(u, e, y)
    a, b = a.copy(), b.copy()
    g = np.ones(nu) if g0 is None else np.asarray(g0, float).copy()
    idx = [np.where(u == i)[0] for i in range(nu)]
    cnt = np.bincount(u, minlength=nu)
    for _ in range(sweeps):
        for i in range(nu):
            m = idx[i]
            X = np.c_[np.ones(len(m)), b[e[m]]]
            a[i], g[i] = np.linalg.lstsq(X, y[m], rcond=None)[0]
        num = np.bincount(e, weights=g[u] * (y - a[u]), minlength=ne)
        den = np.bincount(e, weights=g[u] ** 2, minlength=ne)
        b = num / den
        sc = b.std(ddof=1)
        b = (b - b.mean()) / sc
        g = g * sc
        a = np.bincount(u, weights=y - g[u] * b[e], minlength=nu) / cnt
    return a, g, b, a[u] + g[u] * b[e]


def codes(*cols):
    return [pd.factorize(np.asarray(c))[0] for c in cols]


def mult_rel(alpha, labels, ref):
    i = list(labels).index(ref)
    return {str(l): float(np.exp(alpha[k] - alpha[i])) for k, l in enumerate(labels)}


# ---------------------------------------------------------------- tables
def load_tables():
    df = pd.read_csv(find('event_data.csv'))
    rows = []
    for s in SP:
        for ei in range(len(df)):
            t = df[s].iloc[ei]
            if t and t > 0:
                rows.append((f'ev{ei}', s, float(t), int(df['Cortical'].iloc[ei]), int(df['Limbic'].iloc[ei]), s in PRIM))
    R = pd.DataFrame(rows, columns=['ev', 'sp', 't', 'cort', 'limb', 'prim'])
    R = R[R.ev.map(R.ev.value_counts()) >= 2].reset_index(drop=True)
    a = pd.read_feather(find('aoas.feather'))
    lex = {}
    for meas in ['produces', 'understands']:
        d = a[(a.measure == meas) & a.aoa.notna() & a.uni_lemma.notna()].copy()
        d = d[d.uni_lemma.map(d.uni_lemma.value_counts()) >= 2].reset_index(drop=True)
        lex[meas] = d
    A = pd.read_csv(find('machine_acq_70.csv'))
    A = A[A.paradigm.map(A.paradigm.value_counts()) >= 2].reset_index(drop=True)
    return R, lex, A


R, LEX, A70 = load_tables()
MAM_EXTRA = np.c_[R.prim * R.cort, R.prim * R.limb].astype(float)
TABLES = {
    'mammals': (R.sp.values, R.ev.values, np.log(R.t.values)),
    'lexical': (LEX['produces'].language.values, LEX['produces'].uni_lemma.values, np.log(LEX['produces'].aoa.values.astype(float))),
    'lexical_comprehension': (LEX['understands'].language.values, LEX['understands'].uni_lemma.values, np.log(LEX['understands'].aoa.values.astype(float))),
    'machines': (A70['size'].values, A70.paradigm.values, np.log(A70.tok.values.astype(float))),
}
_m = A70[A70.paradigm.map(A70.paradigm.value_counts()) == 6]
TABLES['machines_complete'] = (_m['size'].values, _m.paradigm.values, np.log(_m.tok.values.astype(float)))
print({k: len(v[2]) for k, v in TABLES.items()}, flush=True)

BLIMP = pd.read_csv(find('blimp_by_step_all.csv'))
BLIMP_DEDUP = pd.read_csv(find('blimp_by_step_dedup.csv'))


# ---------------------------------------------------------------- machine crossings
def logistic(x, L, x0, k):
    return 0.5 + (L - 0.5) / (1 + np.exp(-k * (x - x0)))


def logistic_floor(x, c, L, x0, k):
    return c + (L - c) / (1 + np.exp(-k * (x - x0)))


def cross_logistic(x, y, crit=CRIT):
    if y.max() < crit:
        return 'notreached', np.nan
    try:
        p, _ = curve_fit(logistic, x, y, p0=[max(y.max(), 0.6), x.mean(), 1.0],
                         bounds=([0.5, x.min() - 5, 0.01], [1.0, x.max() + 5, 20]), maxfev=20000)
    except Exception:
        return 'error', np.nan
    L, x0, k = p
    if L <= crit:
        return 'ceiling', np.nan
    xc = x0 - np.log((L - 0.5) / (crit - 0.5) - 1) / k
    if xc < x.min() - 2:
        return 'before', xc
    if xc > x.max() + 2:
        return 'after', xc
    return 'observed', xc


def cross_floor(x, y, crit=CRIT):
    if y.max() < crit:
        return 'notreached', np.nan
    c0 = float(min(np.mean(y[:2]), 0.69))
    try:
        p, _ = curve_fit(logistic_floor, x, y, p0=[c0, max(y.max(), 0.71), x.mean(), 1.0],
                         bounds=([0.0, 0.5, x.min() - 5, 0.01], [0.69, 1.0, x.max() + 5, 20]), maxfev=20000)
    except Exception:
        return 'error', np.nan
    c, L, x0, k = p
    if L <= crit or L <= c:
        return 'ceiling', np.nan
    xc = x0 - np.log((L - c) / (crit - c) - 1) / k
    if xc < x.min() - 2:
        return 'before', xc
    if xc > x.max() + 2:
        return 'after', xc
    return 'observed', xc


def cross_raw(x, y, crit=CRIT, sustained=False, tol=0.05):
    """First checkpoint at or above crit, interpolated linearly in accuracy against log tokens
    from the preceding checkpoint. Sustained: accuracy never again falls below crit - tol."""
    if y.max() < crit:
        return 'notreached', np.nan
    if sustained:
        cand = [j for j in range(len(y)) if y[j] >= crit and (y[j:] >= crit - tol).all()]
        if not cand:
            return 'unsustained', np.nan
        j = cand[0]
    else:
        j = int(np.argmax(y >= crit))
    if j == 0:
        return 'before', np.nan
    x0, x1, y0, y1 = x[j - 1], x[j], y[j - 1], y[j]
    return 'observed', float(x0 + (crit - y0) / (y1 - y0) * (x1 - x0))


def curve_table(d, fn=cross_logistic, crit=CRIT, max_step=None):
    rows = []
    for (s, p), g in d.groupby(['size', 'paradigm']):
        g = g.sort_values('step')
        if max_step is not None and s.startswith('pythia-410m'):
            g = g[g.step <= max_step]
        x = np.log(g.step.values * TOK)
        y = g.acc.values
        st, xc = fn(x, y, crit)
        hit = np.where(y >= crit)[0]
        rows.append(dict(size=s, paradigm=p, status=st, x=xc, xmin=x.min(), xmax=x.max(),
                         xfirsthit=x[hit[0]] if len(hit) else np.nan))
    return pd.DataFrame(rows)


def panel_from(C):
    P = C[C.status == 'observed']
    return P[P.paradigm.map(P.paradigm.value_counts()) >= 2].reset_index(drop=True)


def machine_summary(P, nperm=0, with_stretch=True, ref='pythia-1b', seed=0):
    u, e = codes(P['size'].values, P.paradigm.values)
    ul = pd.factorize(P['size'].values)[1]
    el = pd.factorize(P.paradigm.values)[1]
    y = P.x.values.astype(float)
    al, be, p, _ = additive(u, e, y)
    out = dict(cells=int(len(y)), events=int(e.max() + 1), rho=float(np.corrcoef(y, p)[0, 1]), r2=r2(y, p),
               unit_only_r2=unit_only_r2(u, y), multipliers=mult_rel(al, ul, ref) if ref in ul else None)
    if nperm:
        rng = np.random.default_rng(seed)
        U, sv, _ = np.linalg.svd(design(u, e), full_matrices=False)
        U = U[:, sv > sv.max() * 1e-10]
        null = []
        for _ in range(nperm):
            yp = y.copy()
            for i in range(u.max() + 1):
                m = u == i
                yp[m] = rng.permutation(yp[m])
            null.append(np.corrcoef(yp, U @ (U.T @ yp))[0, 1])
        null = np.array(null)
        out.update(null_mean=float(null.mean()), null_sd=float(null.std()), z=float((out['rho'] - null.mean()) / null.std()))
    if with_stretch:
        a, g, b, ps = stretch(u, e, y)
        q = u.max()
        n = len(y)
        pa = u.max() + 1 + e.max()
        r2s = r2(y, ps)
        out.update(stretch_r2=r2s, gain=r2s - out['r2'],
                   F=float(((r2s - out['r2']) / q) / ((1 - r2s) / (n - pa - q))),
                   stretches={str(l): float(v) for l, v in zip(ul, g / g.mean())})
    out['beta'] = dict(zip(map(str, el), map(float, be)))
    return out


def spearman_beta(b1, b2):
    common = sorted(set(b1) & set(b2))
    if len(common) < 3:
        return None, len(common)
    return float(stats.spearmanr([b1[c] for c in common], [b2[c] for c in common])[0]), len(common)


PRIMARY_CURVES = curve_table(BLIMP)
PRIMARY = machine_summary(panel_from(PRIMARY_CURVES))
print('primary machine panel recomputed:', PRIMARY['cells'], round(PRIMARY['rho'], 4),
      PRIMARY_CURVES.status.value_counts().to_dict(), flush=True)
RES['primary_check'] = {k: v for k, v in PRIMARY.items() if k != 'beta'}
RES['primary_check']['status_counts'] = PRIMARY_CURVES.status.value_counts().to_dict()




def cv(uu, ee, y, reps=5, folds=10):
    rng = np.random.default_rng(0)
    sse_a = sse_s = 0.0
    wins = []
    for rep in range(reps):
        fold = rng.permutation(len(y)) % folds
        for k in range(folds):
            tr = fold != k
            ok = (~tr) & np.isin(ee, np.unique(ee[tr])) & np.isin(uu, np.unique(uu[tr]))
            ul, el = pd.unique(uu[tr]), pd.unique(ee[tr])
            um, em = {l: i for i, l in enumerate(ul)}, {l: i for i, l in enumerate(el)}
            ut = np.array([um[x] for x in uu[tr]]); et = np.array([em[x] for x in ee[tr]])
            us = np.array([um[x] for x in uu[ok]]); es = np.array([em[x] for x in ee[ok]])
            al, be, _, _ = additive(ut, et, y[tr])
            a, g, b, _ = stretch(ut, et, y[tr])
            fa = ((y[ok] - al[us] - be[es]) ** 2).sum()
            fs = ((y[ok] - a[us] - g[us] * b[es]) ** 2).sum()
            sse_a += fa; sse_s += fs; wins.append(bool(fs < fa))
    return dict(mse_ratio_stretch_over_additive=sse_s / sse_a, folds_stretch_better=int(sum(wins)), folds=len(wins))


def summarize(P, ref, base_beta=None):
    s = machine_summary(P, ref=ref)
    s['cv'] = cv(P['size'].values, P.paradigm.values, P.x.values.astype(float))
    if base_beta is not None:
        s['beta_spearman_vs_released128'] = spearman_beta(s['beta'], base_beta)
    s['beta'] = None
    return s


EARLY = pd.concat([pd.read_csv(p) for n in ['early_small.csv', 'early_1b.csv']
                   for p in sorted(glob.glob(os.path.join('/kaggle/input' if os.path.isdir('/kaggle/input') else '.', '**', n), recursive=True))[:1]])
NEW_SIZES = [s for s in SIZES if s in set(EARLY['size'])]
REF = 'pythia-1b' if 'pythia-1b' in NEW_SIZES else NEW_SIZES[-1]
RES['sizes'] = NEW_SIZES
print('sizes', NEW_SIZES, 'early rows', len(EARLY), flush=True)



REL = BLIMP[BLIMP['size'].isin(NEW_SIZES)][['size', 'step', 'paradigm', 'acc']]
EXT = pd.concat([EARLY[EARLY.step < 128][['size', 'step', 'paradigm', 'acc']], REL])
WIN = {'from16': EXT, 'from128': REL}


def fit_short(P):
    s = summarize(P, REF, BASE)
    return dict(cells=s['cells'], events=s['events'], rho=s['rho'], gain=s['gain'], stretches=s['stretches'],
                cv_ratio=s['cv']['mse_ratio_stretch_over_additive'], cv_wins=s['cv']['folds_stretch_better'],
                beta_spearman=s['beta_spearman_vs_released128'])


BASE = machine_summary(panel_from(curve_table(REL)), ref=REF)['beta']




def fitq(x, y):
    try:
        q, _ = curve_fit(logistic, x, y, p0=[max(y.max(), 0.6), x.mean(), 1.0],
                         bounds=([0.5, x.min() - 5, 0.01], [1.0, x.max() + 5, 20]), maxfev=20000)
        return q
    except Exception:
        return None


C16, C128 = curve_table(EXT), curve_table(REL)
O16 = C16[C16.status == 'observed'].set_index(['size', 'paradigm']).x
O128 = C128[C128.status == 'observed'].set_index(['size', 'paradigm']).x
COMMON = O16.index.intersection(O128.index)
X0 = np.log(128 * TOK)
D = pd.DataFrame({'x128': O128[COMMON].values, 'x16': O16[COMMON].values}, index=COMMON).reset_index()
RES['common_cells'] = int(len(D))


def gain_of(T, col):
    P = T.rename(columns={col: 'x'})[['size', 'paradigm', 'x']]
    P = P[P.paradigm.map(P.paradigm.value_counts()) >= 2].reset_index(drop=True)
    s = machine_summary(P, ref=REF)
    return dict(cells=s['cells'], gain=s['gain'], stretches=s['stretches'])


@section('swap_by_size')
def _():
    out = {'all_128': gain_of(D, 'x128'), 'all_16': gain_of(D, 'x16')}
    groups = {sz: [sz] for sz in NEW_SIZES}
    groups.update(small=['pythia-14m', 'pythia-31m'], large=['pythia-410m', 'pythia-1b'], middle=['pythia-70m', 'pythia-160m'])
    for name, szs in groups.items():
        T = D.copy()
        m = T['size'].isin(szs)
        T['xs'] = np.where(m, T.x16, T.x128)
        out['swap_' + name] = gain_of(T, 'xs')
        print(name, round(out['swap_' + name]['gain'], 4), flush=True)
    return out


@section('per_size_slope')
def _():
    P = D.rename(columns={'x16': 'x'})
    P = P[P.paradigm.map(P.paradigm.value_counts()) >= 2]
    u, e = codes(P['size'].values, P.paradigm.values)
    el = pd.factorize(P.paradigm.values)[1]
    _, be, _, _ = additive(u, e, P.x.values.astype(float))
    beta = dict(zip(el, be))
    T = D[D.paradigm.isin(beta)].copy()
    T['beta'] = T.paradigm.map(beta)
    out = {}
    for sz in NEW_SIZES:
        g = T[T['size'] == sz]
        near = g[g.x128 - X0 < 1.0]
        far = g[g.x128 - X0 >= 1.0]
        row = dict(cells=int(len(g)), near_start=int(len(near)))
        for nm, gg in [('all', g), ('far_from_start', far)]:
            if len(gg) >= 4:
                row[nm] = dict(slope128=float(np.polyfit(gg.beta, gg.x128, 1)[0]), slope16=float(np.polyfit(gg.beta, gg.x16, 1)[0]))
        if len(near) >= 2:
            row['near_mean_shift'] = float((near.x128 - near.x16).mean())
        out[sz] = row
    print(out, flush=True)
    return out


@section('shift_drivers')
def _():
    E = EXT.copy()
    early = E[E.step < 128].groupby(['size', 'paradigm']).acc.agg(['mean', 'min'])
    T = D.join(early, on=['size', 'paradigm'])
    T['dx'] = T.x128 - T.x16
    T['d0'] = T.x128 - X0
    T['early_dev'] = T['mean'] - 0.5
    k16, k128 = [], []
    for _, r in T.iterrows():
        for src, store in [(EXT, k16), (REL, k128)]:
            g = src[(src['size'] == r['size']) & (src.paradigm == r.paradigm)].sort_values('step')
            q = fitq(np.log(g.step.values * TOK), g.acc.values)
            store.append(q[2] if q is not None else np.nan)
    T['k16'], T['k128'] = k16, k128
    by = {}
    for sz, g in T.groupby('size'):
        by[sz] = dict(cells=int(len(g)), mean_dx=float(g.dx.mean()), mean_early_dev=float(g.early_dev.mean()),
                      share_early_below_047=float((g['mean'] < 0.47).mean()), median_k16=float(g.k16.median()),
                      median_k128=float(g.k128.median()), mean_d0=float(g.d0.mean()))
    cor = {c: float(T[['dx', c]].dropna().corr().iloc[0, 1]) for c in ['d0', 'early_dev', 'min', 'k16', 'k128']}
    F = T.dropna(subset=['d0', 'early_dev', 'min'])
    X = np.c_[np.ones(len(F)), F.d0, F.early_dev, F['min']]
    b, p = lsfit(X, F.dx.values)
    out = dict(by_size=by, corr_with_dx=cor, ols=dict(coef=dict(zip(['const', 'd0', 'early_dev', 'early_min'], map(float, b))), r2=r2(F.dx.values, p), n=int(len(F))))
    print(out, flush=True)
    return out

save()
