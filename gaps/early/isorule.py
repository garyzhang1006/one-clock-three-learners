"""Is there a crossing rule whose Pythia acquisition times don't depend on the first checkpoint?
Simulations bounded both measurement explanations of the window effect (a late window can't manufacture the
stretch from shift-only curves, and early checkpoints can't erase a real one), which leaves curve structure that
one logistic per curve doesn't describe. Sections:
  two_phase     per size-paradigm curve from step 16, BIC of one logistic against a sum of two logistics
                (an early partial rise and a later one), by size
  isotonic      shape-free rule: the first point where the isotonic (monotone) fit of accuracy on log tokens
                reaches 0.70, linearly interpolated, 'before' at the first checkpoint; window 16 vs 128, gain,
                held-out ratio, stretches, and the window shift of crossings on common cells
  iso_bootstrap 100 binomial resamples of every accuracy under the isotonic rule in both windows
Writes isorule_results.json."""

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
    with open(os.path.join(OUT, 'isorule_results.json'), 'w') as f:
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



from sklearn.isotonic import IsotonicRegression


def cross_iso(x, y, crit=CRIT):
    if y.max() < crit:
        return 'notreached', np.nan
    f = IsotonicRegression(increasing=True).fit_transform(x, y)
    j = int(np.argmax(f >= crit))
    if f[j] < crit:
        return 'notreached', np.nan
    if j == 0:
        return 'before', np.nan
    x0, x1, y0, y1 = x[j - 1], x[j], f[j - 1], f[j]
    return 'observed', float(x0 + (crit - y0) / (y1 - y0) * (x1 - x0))


def two_logistic(x, a1, m1, k1, a2, m2, k2):
    return 0.5 + a1 / (1 + np.exp(-k1 * (x - m1))) + a2 / (1 + np.exp(-k2 * (x - m2)))


@section('two_phase')
def _():
    out = defaultdict(lambda: dict(n=0, two_better=0, early_amp=[]))
    for (sz, p), g in EXT.groupby(['size', 'paradigm']):
        g = g.sort_values('step')
        x, y = np.log(g.step.values * TOK), g.acc.values
        n = len(y)
        if y.max() < 0.6 or n < 10:
            continue
        try:
            q1, _ = curve_fit(logistic, x, y, p0=[max(y.max(), 0.6), x.mean(), 1.0],
                              bounds=([0.5, x.min() - 5, 0.01], [1.0, x.max() + 5, 20]), maxfev=20000)
            r1 = ((y - logistic(x, *q1)) ** 2).sum()
            q2, _ = curve_fit(two_logistic, x, y, p0=[0.05, x.min() + 2, 2.0, max(y.max() - 0.55, 0.05), x.mean(), 1.0],
                              bounds=([0, x.min() - 2, 0.05, 0, x.min() - 2, 0.05], [0.5, x.max() + 2, 20, 0.5, x.max() + 2, 20]), maxfev=40000)
            r2_ = ((y - two_logistic(x, *q2)) ** 2).sum()
        except Exception:
            continue
        b1 = n * np.log(r1 / n) + 3 * np.log(n)
        b2 = n * np.log(r2_ / n) + 6 * np.log(n)
        d = out[sz]
        d['n'] += 1
        d['two_better'] += int(b2 < b1 - 2)
        early = q2[0] if q2[1] < q2[4] else q2[3]
        d['early_amp'].append(float(early))
    return {sz: dict(n=d['n'], share_two_phase=d['two_better'] / max(d['n'], 1), median_early_amp=float(np.median(d['early_amp'])) if d['early_amp'] else None)
            for sz, d in out.items()}


@section('isotonic')
def _():
    out = {}
    C = {}
    for wn, T in WIN.items():
        C[wn] = curve_table(T, fn=cross_iso)
        out[wn] = fit_short(panel_from(C[wn]))
        out[wn]['status'] = C[wn].status.value_counts().to_dict()
        print(wn, {k: v for k, v in out[wn].items() if k != 'stretches'}, flush=True)
    O = {w: c[c.status == 'observed'].set_index(['size', 'paradigm']).x for w, c in C.items()}
    com = O['from16'].index.intersection(O['from128'].index)
    d = (O['from128'][com] - O['from16'][com])
    out['common'] = dict(cells=int(len(com)), mean_shift=float(d.mean()), share_identical=float((d.abs() < 1e-9).mean()),
                         by_size={sz: float(v) for sz, v in d.groupby(level=0).mean().items()})
    D = pd.DataFrame({'a': O['from128'][com].values, 'b': O['from16'][com].values}, index=com).reset_index()
    for col, nm in [('a', 'common_from128'), ('b', 'common_from16')]:
        P = D.rename(columns={col: 'x'})[['size', 'paradigm', 'x']]
        P = P[P.paradigm.map(P.paradigm.value_counts()) >= 2].reset_index(drop=True)
        s_ = machine_summary(P, ref=REF)
        out[nm] = dict(cells=s_['cells'], gain=s_['gain'], stretches=s_['stretches'])
    return out


@section('iso_bootstrap')
def _():
    rng = np.random.default_rng(3)
    res = {w: dict(gain=[], s14=[], s1b=[]) for w in WIN}
    for b in range(100):
        for wn, T in WIN.items():
            Tb = T.copy()
            Tb['acc'] = rng.binomial(200, Tb.acc.clip(0, 1).values) / 200
            s_ = machine_summary(panel_from(curve_table(Tb, fn=cross_iso)), ref=REF)
            res[wn]['gain'].append(s_['gain'])
            res[wn]['s14'].append(s_['stretches'].get('pythia-14m', np.nan))
            res[wn]['s1b'].append(s_['stretches'].get(REF, np.nan))
    out = {}
    for wn, v in res.items():
        g, a, z = map(np.array, (v['gain'], v['s14'], v['s1b']))
        out[wn] = dict(gain_median=float(np.median(g)), gain_ci=pct(g), s14_ci=pct(a), s1b_ci=pct(z), share_14m_gt_1b=float(np.mean(a > z)))
    diff = np.array(res['from128']['gain']) - np.array(res['from16']['gain'])
    out['diff_128_minus_16'] = dict(median=float(np.median(diff)), ci=pct(diff))
    print(out, flush=True)
    return out

save()
