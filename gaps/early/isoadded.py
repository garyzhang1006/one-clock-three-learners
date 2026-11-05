"""Which acquisitions break the Pythia stretch, and is the size-by-event relation curved rather than linear?
Under the isotonic crossing rule, crossings observed in both windows are window-invariant and carry the stretch,
and the cells that only the step-16 window dates lower the gain from 0.103 to 0.033. Sections:
  added_cells  the cells observed from step 16 but not from step 128: counts by size and paradigm, their crossings,
               and their residuals from the stretch model fitted to the common cells (predicted vs observed)
  curvature    on the full step-16 isotonic panel, a size-specific quadratic in the event term,
               x = a_u + g_u b_e + h_u b_e^2 (b_e fixed from the additive fit), against the stretch model:
               in-sample R2 and 10-fold x 5 cross-validated error
Writes isoadded_results.json."""

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
    with open(os.path.join(OUT, 'isoadded_results.json'), 'w') as f:
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




C = {w: curve_table(T, fn=cross_iso) for w, T in WIN.items()}
O = {w: c[c.status == 'observed'].set_index(['size', 'paradigm']).x for w, c in C.items()}
COM = O['from16'].index.intersection(O['from128'].index)
ADDED = O['from16'].index.difference(O['from128'].index)


@section('added_cells')
def _():
    st128 = C['from128'].set_index(['size', 'paradigm']).status
    A = pd.DataFrame({'x': O['from16'][ADDED].values}, index=ADDED).reset_index()
    A['status128'] = [st128.get(ix, 'missing') for ix in ADDED]
    P = pd.DataFrame({'x': O['from16'][COM].values}, index=COM).reset_index()
    P = P[P.paradigm.map(P.paradigm.value_counts()) >= 2].reset_index(drop=True)
    u, e = codes(P['size'].values, P.paradigm.values)
    ul = list(pd.factorize(P['size'].values)[1]); el = list(pd.factorize(P.paradigm.values)[1])
    a, g, b, _ = stretch(u, e, P.x.values.astype(float))
    al, be, _, _ = additive(u, e, P.x.values.astype(float))
    ok = A['size'].isin(ul) & A.paradigm.isin(el)
    A.loc[ok, 'pred_stretch'] = [a[ul.index(z)] + g[ul.index(z)] * b[el.index(q)] for z, q in zip(A['size'][ok], A.paradigm[ok])]
    A.loc[ok, 'pred_additive'] = [al[ul.index(z)] + be[el.index(q)] for z, q in zip(A['size'][ok], A.paradigm[ok])]
    A['res_stretch'] = A.x - A.pred_stretch
    A['res_additive'] = A.x - A.pred_additive
    x128 = float(np.log(128 * TOK))
    out = dict(n_added=int(len(A)), n_common=int(len(COM)), status128=A.status128.value_counts().to_dict(),
               by_size=A.groupby('size').size().to_dict(), share_before_step128=float((A.x < x128).mean()),
               new_paradigms=sorted(set(A.paradigm) - set(el)),
               mean_res_stretch_by_size=A[ok].groupby('size').res_stretch.mean().to_dict(),
               mean_res_additive_by_size=A[ok].groupby('size').res_additive.mean().to_dict(),
               mse_stretch=float((A[ok].res_stretch ** 2).mean()), mse_additive=float((A[ok].res_additive ** 2).mean()),
               cells=A.round(3).to_dict('records'))
    print({k: v for k, v in out.items() if k != 'cells'}, flush=True)
    return out


def quad_fit(u, b, y):
    nu = u.max() + 1
    a, g, h = np.zeros(nu), np.zeros(nu), np.zeros(nu)
    for i in range(nu):
        m = u == i
        X = np.c_[np.ones(m.sum()), b[m], b[m] ** 2]
        a[i], g[i], h[i] = np.linalg.lstsq(X, y[m], rcond=None)[0]
    return a, g, h


@section('curvature')
def _():
    P = panel_from(C['from16'])
    uu, ee, y = P['size'].values, P.paradigm.values, P.x.values.astype(float)
    u, e = codes(uu, ee)
    al, be, pa, _ = additive(u, e, y)
    _, _, _, ps = stretch(u, e, y)
    bb = be[e]
    a, g, h = quad_fit(u, bb, y)
    pq = a[u] + g[u] * bb + h[u] * bb ** 2
    ul = pd.factorize(uu)[1]
    rng = np.random.default_rng(0)
    sse = dict(additive=0.0, stretch=0.0, quad=0.0)
    for rep in range(5):
        fold = rng.permutation(len(y)) % 10
        for k in range(10):
            tr = fold != k
            te = (~tr) & np.isin(ee, np.unique(ee[tr])) & np.isin(uu, np.unique(uu[tr]))
            ul_, el_ = pd.unique(uu[tr]), pd.unique(ee[tr])
            um, em = {l: i for i, l in enumerate(ul_)}, {l: i for i, l in enumerate(el_)}
            ut = np.array([um[x] for x in uu[tr]]); et = np.array([em[x] for x in ee[tr]])
            us = np.array([um[x] for x in uu[te]]); es = np.array([em[x] for x in ee[te]])
            a1, b1, _, _ = additive(ut, et, y[tr])
            a2, g2, b2, _ = stretch(ut, et, y[tr])
            a3, g3, h3 = quad_fit(ut, b1[et], y[tr])
            sse['additive'] += ((y[te] - a1[us] - b1[es]) ** 2).sum()
            sse['stretch'] += ((y[te] - a2[us] - g2[us] * b2[es]) ** 2).sum()
            sse['quad'] += ((y[te] - a3[us] - g3[us] * b1[es] - h3[us] * b1[es] ** 2) ** 2).sum()
    out = dict(cells=int(len(y)), r2=dict(additive=r2(y, pa), stretch=r2(y, ps), quad=r2(y, pq)),
               cv_mse_ratio_vs_additive={k: v / sse['additive'] for k, v in sse.items()},
               slope=dict(zip(map(str, ul), map(float, g))), curvature=dict(zip(map(str, ul), map(float, h))))
    print(out, flush=True)
    return out

save()
