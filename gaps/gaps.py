"""Analyses listed as not run in the One Clock paper, on the released panels.

Each section writes into gaps_results.json and prints a summary line, and a failing
section records its traceback without stopping the others. Inputs are the release
files (event_data.csv, aoas.feather, blimp_by_step_*.csv, machine_acq_*.csv,
machine_event_terms.csv), found anywhere under /kaggle/input or the working directory.

Sections
  A  cell-level cross-validation of the shift-only and shift-plus-stretch models
  B  held-unit prediction against baselines (event-only, anchor mean, anchor stretch)
  C  crossing-measurement bootstrap: binomial resampling of every BLiMP accuracy
  D  event (paradigm, lemma) cluster bootstrap for multipliers, stretches, primate terms
  E  alternative crossing rules: raw first crossing, sustained crossing, free floor
  F  censored (Tobit) shift and stretch models over all 402 machine curves
  G  lexical-norm unit bootstrap recomputed with all 500 draws
  H  power of the two-group event-pairing test under simulated cortical delays
  I  design-aware variance decomposition of unit and event effects
  J  deduplication check with matched 410M windows and common cells
  K  machine panel without the nine extrapolated crossings
  L  spiny mouse removed from the stretch fit; lexical stretch restarts
"""
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
    with open(os.path.join(OUT, 'gaps_results.json'), 'w') as f:
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


# ---------------------------------------------------------------- A: cross-validation
@section('A_cross_validation')
def _():
    out = {}
    for name in ['mammals', 'lexical', 'machines', 'machines_complete']:
        uu, ee, y = TABLES[name]
        rng = np.random.default_rng(0)
        sse_a = sse_s = sst = 0.0
        ntest = nskip = 0
        fold_wins = []
        for rep in range(5):
            fold = rng.permutation(len(y)) % 10
            for k in range(10):
                tr = fold != k
                ok = (~tr) & np.isin(ee, np.unique(ee[tr])) & np.isin(uu, np.unique(uu[tr]))
                nskip += int(((~tr) & ~ok).sum())
                ul, el = pd.unique(uu[tr]), pd.unique(ee[tr])
                um, em = {l: i for i, l in enumerate(ul)}, {l: i for i, l in enumerate(el)}
                ut = np.array([um[x] for x in uu[tr]])
                et = np.array([em[x] for x in ee[tr]])
                us = np.array([um[x] for x in uu[ok]])
                es = np.array([em[x] for x in ee[ok]])
                al, be, _, _ = additive(ut, et, y[tr])
                a, g, b, _ = stretch(ut, et, y[tr])
                pa = al[us] + be[es]
                ps = a[us] + g[us] * b[es]
                ya = y[ok]
                fa, fs = ((ya - pa) ** 2).sum(), ((ya - ps) ** 2).sum()
                sse_a += fa
                sse_s += fs
                sst += ((ya - y[tr].mean()) ** 2).sum()
                ntest += int(ok.sum())
                fold_wins.append(fs < fa)
        out[name] = dict(oos_r2_additive=1 - sse_a / sst, oos_r2_stretch=1 - sse_s / sst,
                         mse_ratio_stretch_over_additive=sse_s / sse_a, folds_stretch_better=int(sum(fold_wins)),
                         folds=len(fold_wins), test_cells=ntest, skipped_cells=nskip)
        print('A', name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[name].items()}, flush=True)
    return out


# ---------------------------------------------------------------- B: held-unit baselines
@section('B_heldunit_baselines')
def _():
    def run(name, unit, event, y, n_anchor, rng):
        recs = defaultdict(list)
        for u0 in pd.unique(unit):
            tr, te = unit != u0, unit == u0
            if te.sum() < n_anchor + 3:
                continue
            ev_tr = pd.unique(event[tr])
            ui = pd.factorize(unit[tr])[0]
            ei = pd.Categorical(event[tr], categories=ev_tr).codes
            al, be, _, _ = additive(ui, ei, y[tr])
            bmap = dict(zip(ev_tr, be))
            a_s, g_s, b_s, _ = stretch(ui, ei, y[tr])
            smap = dict(zip(ev_tr, b_s))
            mean_alpha = float(np.mean(al))
            idx = np.where(te)[0]
            usable = [i for i in idx if event[i] in bmap]
            if len(usable) < n_anchor + 3:
                continue
            for _ in range(200):
                perm = rng.permutation(usable)
                anc, tst = perm[:n_anchor], perm[n_anchor:]
                obs = y[tst]
                bt = np.array([bmap[event[i]] for i in tst])
                ba = np.array([bmap[event[i]] for i in anc])
                st = np.array([smap[event[i]] for i in tst])
                sa = np.array([smap[event[i]] for i in anc])
                preds = {
                    'additive_anchor': np.mean(y[anc] - ba) + bt,
                    'event_only': mean_alpha + bt,
                    'anchor_mean': np.full(len(tst), np.mean(y[anc])),
                }
                c = np.linalg.lstsq(np.c_[np.ones(n_anchor), sa], y[anc], rcond=None)[0]
                preds['stretch_anchor'] = c[0] + c[1] * st
                for k, pr in preds.items():
                    recs[k].append((u0, np.mean((obs - pr) ** 2), np.var(obs),
                                    np.median(np.abs(np.exp(obs) - np.exp(pr)) / np.exp(obs))))
        res = {}
        for k, v in recs.items():
            d = pd.DataFrame(v, columns=['unit', 'mse', 'var', 'mape'])
            res[k] = dict(r2=float(1 - d.mse.sum() / d['var'].sum()), median_rel_err=float(d.mape.median()),
                          units=int(d.unit.nunique()))
        return res

    out = {}
    for n_anchor in [3, 5]:
        rng = np.random.default_rng(0)  # same draw sequence as the release's loo.py for three anchors
        for name in ['mammals', 'lexical', 'lexical_comprehension', 'machines']:
            uu, ee, y = TABLES[name]
            out[f'{name}_{n_anchor}'] = run(name, np.asarray(uu), np.asarray(ee), y, n_anchor, rng)
            print('B', name, n_anchor, {k: round(v['r2'], 3) for k, v in out[f'{name}_{n_anchor}'].items()}, flush=True)
    return out


# ---------------------------------------------------------------- C: crossing measurement bootstrap
@section('C_crossing_bootstrap')
def _():
    rng = np.random.default_rng(1)
    draws = []
    for b in range(200):
        d = BLIMP.copy()
        d['acc'] = rng.binomial(d['n'].values.astype(int), d['acc'].values) / d['n'].values
        s = machine_summary(panel_from(curve_table(d)))
        s['spearman_beta_primary'] = spearman_beta(s['beta'], PRIMARY['beta'])[0]
        del s['beta']
        draws.append(s)
        if b % 25 == 0:
            print('C draw', b, round(s['rho'], 3), round(s['gain'], 3), flush=True)
    out = {}
    for k in ['cells', 'events', 'rho', 'r2', 'unit_only_r2', 'stretch_r2', 'gain', 'F', 'spearman_beta_primary']:
        v = [d[k] for d in draws]
        out[k] = dict(median=float(np.median(v)), ci=pct(v))
    for key in ['multipliers', 'stretches']:
        out[key] = {s: dict(median=float(np.median([d[key][s] for d in draws if s in d[key]])),
                            ci=pct([d[key][s] for d in draws if s in d[key]])) for s in SIZES}
    order = [all(np.diff([d['stretches'][s] for s in SIZES]) < 0) for d in draws]
    out['share_draws_stretch_14m_gt_1b'] = float(np.mean([d['stretches']['pythia-14m'] > d['stretches']['pythia-1b'] for d in draws]))
    out['share_draws_stretch_monotone'] = float(np.mean(order))
    out['share_draws_410m_slower_than_160m'] = float(np.mean([d['multipliers']['pythia-410m'] > d['multipliers']['pythia-160m'] for d in draws]))
    out['n_draws'] = len(draws)
    return out


# ---------------------------------------------------------------- D + I: event bootstrap and decomposition
def decomposition(u, e, y):
    al, be, p, _ = additive(u, e, y)
    vy = y.var()
    au, bc = al[u], be[e]
    full = r2(y, p)
    uo = unit_only_r2(u, y)
    eo = r2(y, lsfit(design(u, e, unit=False), y)[1])
    return dict(full_r2=full, unit_only_r2=uo, event_only_r2=eo, unit_given_event=full - eo,
                event_given_unit=full - uo, var_unit=float(au.var() / vy), var_event=float(bc.var() / vy),
                two_cov=float(2 * np.cov(au, bc, bias=True)[0, 1] / vy), var_resid=float((y - p).var() / vy),
                span=float(np.exp(al.max() - al.min())))


def event_boot(u_lab, e_lab, y, B, rng, fn):
    ev = pd.unique(e_lab)
    by = {x: np.where(e_lab == x)[0] for x in ev}
    out = []
    for _ in range(B):
        pick = rng.choice(ev, size=len(ev), replace=True)
        idx = np.concatenate([by[x] for x in pick])
        lab = np.concatenate([np.full(len(by[x]), f'{x}#{j}') for j, x in enumerate(pick)])
        uu = u_lab[idx]
        if len(pd.unique(uu)) < len(pd.unique(u_lab)):
            continue
        out.append(fn(uu, lab, y[idx], idx))
    return out


@section('I_decomposition')
def _():
    out = {}
    for name, (uu, ee, y) in TABLES.items():
        u, e = codes(uu, ee)
        point = decomposition(u, e, y)
        rng = np.random.default_rng(2)
        B = 200 if name.startswith('lexical') else 500
        bs = event_boot(np.asarray(uu), np.asarray(ee), y, B, rng, lambda a, b, c, i: decomposition(*codes(a, b), c))
        point['ci'] = {k: pct([d[k] for d in bs]) for k in point}
        point['n_draws'] = len(bs)
        out[name] = point
        print('I', name, {k: round(v, 3) for k, v in point.items() if isinstance(v, float)}, flush=True)
    return out


@section('D_event_bootstrap')
def _():
    out = {}
    # machines: multipliers and stretches with paradigms resampled
    uu, ee, y = TABLES['machines']
    uu, ee = np.asarray(uu), np.asarray(ee)

    def mach(a, b, c, i, st):
        u, e = codes(a, b)
        ul = pd.factorize(a)[1]
        al, _, p, _ = additive(u, e, c)
        r = dict(mult=mult_rel(al, ul, 'pythia-1b'), uo=unit_only_r2(u, c), rho=float(np.corrcoef(c, p)[0, 1]))
        if st:
            _, g, _, ps = stretch(u, e, c)
            r['stretch'] = {str(l): float(v) for l, v in zip(ul, g / g.mean())}
            r['gain'] = r2(c, ps) - r2(c, p)
        return r

    bs = event_boot(uu, ee, y, 500, np.random.default_rng(3), lambda a, b, c, i: mach(a, b, c, i, False))
    bss = event_boot(uu, ee, y, 200, np.random.default_rng(4), lambda a, b, c, i: mach(a, b, c, i, True))
    out['machines'] = dict(
        multipliers={s: pct([d['mult'][s] for d in bs]) for s in SIZES},
        unit_only_r2=pct([d['uo'] for d in bs]), rho=pct([d['rho'] for d in bs]),
        stretches={s: pct([d['stretch'][s] for d in bss]) for s in SIZES}, gain=pct([d['gain'] for d in bss]),
        share_410m_slower_than_160m=float(np.mean([d['mult']['pythia-410m'] > d['mult']['pythia-160m'] for d in bs])),
        share_stretch_14m_gt_1b=float(np.mean([d['stretch']['pythia-14m'] > d['stretch']['pythia-1b'] for d in bss])),
        n=[len(bs), len(bss)])
    print('D machines', out['machines']['multipliers'], out['machines']['stretches'], flush=True)

    # mammals: primate interactions, multipliers, stretches with events resampled
    sp, ev, ym = R.sp.values, R.ev.values, np.log(R.t.values)

    def mam(a, b, c, i, st):
        u, e = codes(a, b)
        ul = pd.factorize(a)[1]
        _, _, _, d = additive(u, e, c, MAM_EXTRA[i])
        al, _, p, _ = additive(u, e, c)
        r = dict(dC=float(d[0]), dL=float(d[1]), mult=mult_rel(al, ul, 'Hamster'))
        if st:
            _, g, _, _ = stretch(u, e, c)
            r['stretch'] = {str(l): float(v) for l, v in zip(ul, g / g.mean())}
        return r

    bm = event_boot(sp, ev, ym, 500, np.random.default_rng(5), lambda a, b, c, i: mam(a, b, c, i, False))
    bms = event_boot(sp, ev, ym, 200, np.random.default_rng(6), lambda a, b, c, i: mam(a, b, c, i, True))
    out['mammals'] = dict(cortical=pct([d['dC'] for d in bm]), limbic=pct([d['dL'] for d in bm]),
                          share_cortical_positive=float(np.mean([d['dC'] > 0 for d in bm])),
                          multipliers={s: pct([d['mult'][s] for d in bm]) for s in SP},
                          stretches={s: pct([d['stretch'][s] for d in bms]) for s in SP}, n=[len(bm), len(bms)])
    print('D mammals', out['mammals']['cortical'], out['mammals']['limbic'], flush=True)

    # lexical norms: multipliers with lemmas resampled
    lu, le, ly = TABLES['lexical']
    lu, le = np.asarray(lu), np.asarray(le)

    def lexf(a, b, c, i):
        u, e = codes(a, b)
        ul = pd.factorize(a)[1]
        al, _, _, _ = additive(u, e, c)
        return dict(mult=mult_rel(al, ul, 'Slovak'), span=float(np.exp(al.max() - al.min())))

    bl = event_boot(lu, le, ly, 200, np.random.default_rng(7), lexf)
    langs = sorted(pd.unique(lu))
    out['lexical'] = dict(multipliers={s: pct([d['mult'][s] for d in bl]) for s in langs},
                          span=pct([d['span'] for d in bl]), n=len(bl))
    return out


# ---------------------------------------------------------------- E: alternative crossing rules
@section('E_crossing_rules')
def _():
    out = {}
    rules = {'raw_first': lambda x, y, c: cross_raw(x, y, c),
             'sustained': lambda x, y, c: cross_raw(x, y, c, sustained=True),
             'free_floor': cross_floor}
    for name, fn in rules.items():
        C = curve_table(BLIMP, fn)
        s = machine_summary(panel_from(C), nperm=1000)
        s['spearman_beta_primary'], s['shared_events'] = spearman_beta(s['beta'], PRIMARY['beta'])
        P = panel_from(C).merge(panel_from(PRIMARY_CURVES), on=['size', 'paradigm'], suffixes=('', '_p'))
        s['common_cells'] = int(len(P))
        s['common_cell_pearson_log'] = float(np.corrcoef(P.x, P.x_p)[0, 1]) if len(P) > 2 else None
        s['common_cell_median_abs_diff_log'] = float(np.median(np.abs(P.x - P.x_p))) if len(P) else None
        s['status_counts'] = C.status.value_counts().to_dict()
        del s['beta']
        out[name] = s
        print('E', name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if not isinstance(v, dict)}, flush=True)
    return out


# ---------------------------------------------------------------- F: censored models
def censored_cells(C, variant):
    """Observed, right-censored and left-censored cells over all curves of paradigms with an
    accepted crossing. 'touched': fit-guard ceiling and late exclusions are right-censored at the
    last checkpoint, early exclusions left-censored at their first checkpoint at or above 0.70.
    'drop_touched': guard exclusions are dropped, only non-attainment is censored."""
    keep = set(C.loc[C.status == 'observed', 'paradigm'])
    rows = []
    for r in C[C.paradigm.isin(keep)].itertuples():
        if r.status == 'observed':
            rows.append((r.size, r.paradigm, r.x, 0))
        elif r.status == 'notreached':
            rows.append((r.size, r.paradigm, r.xmax, 1))
        elif variant == 'touched' and r.status in ('ceiling', 'after'):
            rows.append((r.size, r.paradigm, r.xmax, 1))
        elif variant == 'touched' and r.status == 'before':
            rows.append((r.size, r.paradigm, r.xfirsthit, -1))
    return pd.DataFrame(rows, columns=['size', 'paradigm', 'y', 'cens'])


def _lam(w):
    return np.exp(stats.norm.logpdf(w) - stats.norm.logcdf(w))


def tobit(D, model='shift'):
    u, e = codes(D['size'].values, D.paradigm.values)
    ul = pd.factorize(D['size'].values)[1]
    y, c = D.y.values.astype(float), D.cens.values
    nu, ne = u.max() + 1, e.max() + 1
    o, rc, lc = c == 0, c == 1, c == -1
    al, be, p, _ = additive(u[o], e[o], y[o]) if len(np.unique(e[o])) == ne else (None, None, None, None)
    if al is None:
        raise RuntimeError('an event lacks observed cells')
    sig0 = np.log((y[o] - p).std())
    ref = list(ul).index('pythia-1b')

    def unpack(th):
        a = th[:nu]
        if model == 'shift':
            g = np.ones(nu)
            b = np.r_[0.0, th[nu:nu + ne - 1]]
            ls = th[nu + ne - 1]
        else:
            g = np.insert(th[nu:2 * nu - 1], ref, 1.0)
            b = np.r_[0.0, th[2 * nu - 1:2 * nu - 1 + ne - 1]]
            ls = th[-1]
        return a, g, b, ls

    def nll(th):
        a, g, b, ls = unpack(th)
        s = np.exp(ls)
        mu = a[u] + g[u] * b[e]
        ll = np.zeros(len(y))
        dmu = np.zeros(len(y))
        dls = np.zeros(len(y))
        z = (y[o] - mu[o]) / s
        ll[o] = -0.5 * z ** 2 - ls
        dmu[o] = z / s
        dls[o] = z ** 2 - 1
        w = (mu[rc] - y[rc]) / s
        ll[rc] = stats.norm.logcdf(w)
        dmu[rc] = _lam(w) / s
        dls[rc] = -_lam(w) * w
        v = (y[lc] - mu[lc]) / s
        ll[lc] = stats.norm.logcdf(v)
        dmu[lc] = -_lam(v) / s
        dls[lc] = -_lam(v) * v
        ga = np.bincount(u, weights=dmu, minlength=nu)
        gb = np.bincount(e, weights=dmu * g[u], minlength=ne)[1:]
        if model == 'shift':
            grad = np.r_[ga, gb, dls.sum()]
        else:
            gg = np.delete(np.bincount(u, weights=dmu * b[e], minlength=nu), ref)
            grad = np.r_[ga, gg, gb, dls.sum()]
        return -ll.sum(), -grad

    if model == 'shift':
        th0 = np.r_[al, be[1:], sig0]
    else:
        th0 = np.r_[al, np.ones(nu - 1), be[1:], sig0]
    res = minimize(nll, th0, jac=True, method='L-BFGS-B', options=dict(maxiter=20000, maxfun=50000))
    a, g, b, ls = unpack(res.x)
    mult = {str(l): float(np.exp(a[k] - a[ref])) for k, l in enumerate(ul)} if model == 'shift' else None
    return dict(loglik=float(-res.fun), converged=bool(res.success), sigma=float(np.exp(ls)), multipliers=mult,
                stretches={str(l): float(v) for l, v in zip(ul, g / g.mean())} if model != 'shift' else None,
                n_obs=int(o.sum()), n_right=int(rc.sum()), n_left=int(lc.sum()), events=int(ne))


@section('F_censored')
def _():
    out = {}
    for variant in ['touched', 'drop_touched']:
        D = censored_cells(PRIMARY_CURVES, variant)
        sh = tobit(D, 'shift')
        st = tobit(D, 'stretch')
        lr = 2 * (st['loglik'] - sh['loglik'])
        Do = D[D.cens == 0]
        u, e = codes(Do['size'].values, Do.paradigm.values)
        al, _, _, _ = additive(u, e, Do.y.values)
        ols = mult_rel(al, pd.factorize(Do['size'].values)[1], 'pythia-1b')
        rng = np.random.default_rng(8)
        par = pd.unique(D.paradigm)
        bs = []
        for _ in range(200):
            pick = rng.choice(par, size=len(par), replace=True)
            Db = pd.concat([D[D.paradigm == x].assign(paradigm=f'{x}#{j}') for j, x in enumerate(pick)])
            try:
                r = tobit(Db, 'shift')
                bs.append(r['multipliers'])
            except Exception:
                continue
        out[variant] = dict(shift=sh, stretch=st, lr_stretch=float(lr), lr_df=5, lr_p=float(stats.chi2.sf(lr, 5)),
                            ols_same_events=ols,
                            shift_mult_ci={s: pct([d[s] for d in bs]) for s in SIZES}, n_boot=len(bs))
        print('F', variant, sh['multipliers'], st['stretches'], round(lr, 1), flush=True)
    return out


# ---------------------------------------------------------------- G: lexical bootstrap, all 500 draws
@section('G_lexical_bootstrap500')
def _():
    def hat(X):
        try:
            U, s, _ = np.linalg.svd(X, full_matrices=False)
            fb = False
        except np.linalg.LinAlgError:
            U, s, _ = scipy_svd(X, full_matrices=False, lapack_driver='gesvd')
            fb = True
        keep = s > s.max() * 1e-10
        return U[:, keep], fb

    def rho_k0(uu, ee, t, ex):
        u, e = codes(uu, ee)
        U, fb = hat(design(u, e, ex))
        y = np.log(t)
        return float(np.corrcoef(y, U @ (U.T @ y))[0, 1]), fb

    rng = np.random.default_rng(0)  # same sequence as bootstrap_ci.py: mammals, then lexical norms
    SYS = [('mammals', R.sp.values, R.ev.values, R.t.values, MAM_EXTRA),
           ('lexical', LEX['produces'].language.values, LEX['produces'].uni_lemma.values,
            LEX['produces'].aoa.values.astype(float), None)]
    out = {}
    for nm, u, e, t, ex in SYS:
        units = pd.unique(u)
        by = {x: np.where(u == x)[0] for x in units}
        draws, fbs = [], 0
        for _ in range(500):
            pick = rng.choice(units, size=len(units), replace=True)
            idx = np.concatenate([by[x] for x in pick])
            lab = np.concatenate([np.full(len(by[x]), f'{x}#{j}') for j, x in enumerate(pick)])
            ee = e[idx]
            keep = pd.Series(ee).map(pd.Series(ee).value_counts()).values >= 2
            if keep.sum() < 20:
                continue
            r, fb = rho_k0(lab[keep], ee[keep], t[idx][keep], None if ex is None else ex[idx][keep])
            fbs += fb
            draws.append(r)
        out[nm] = dict(ci=pct(draws), n_draws=len(draws), gesvd_fallbacks=fbs)
        print('G', nm, out[nm], flush=True)
    return out


# ---------------------------------------------------------------- H: event-pairing power
def flip_test(T, cort, rng, nnull):
    def rel(a, b):
        return 0 if a < b else (2 if a > b else 1)
    pairs = defaultdict(lambda: {'np': [], 'p': []})
    for s in SP:
        ev = sorted(T[s])
        for i, j in itertools.combinations(ev, 2):
            pairs[(i, j)]['p' if s in PRIM else 'np'].append(rel(T[s][i], T[s][j]))
    flips, tested, ev_tested = [], 0, set()
    for (i, j), d in pairs.items():
        if len(d['np']) >= 3 and len(d['p']) >= 1:
            tested += 1
            ev_tested |= {i, j}
            mode = max(set(d['np']), key=d['np'].count)
            if d['np'].count(mode) / len(d['np']) >= 0.8 and all(x != mode for x in d['p']):
                flips.append((i, j))
    inflip = set()
    for i, j in flips:
        inflip |= {i, j}
    if not inflip:
        return tested, 0, 1.0
    obs = np.mean([cort[x] for x in inflip])
    tl = sorted(ev_tested)
    null = np.array([np.mean([cort[x] for x in rng.choice(tl, size=len(inflip), replace=False)]) for _ in range(nnull)])
    return tested, len(flips), float((null >= obs).mean())


@section('H_eventpair_power')
def _():
    u, e = codes(R.sp.values, R.ev.values)
    y = np.log(R.t.values)
    al, be, p, d = additive(u, e, y, MAM_EXTRA)
    rank = np.linalg.matrix_rank(design(u, e, MAM_EXTRA))
    sig = float(np.sqrt(((y - p) ** 2).sum() / (len(y) - rank)))
    base = p - d[0] * MAM_EXTRA[:, 0]
    cort = {r.ev: int(r.cort) for r in R.itertuples()}
    rng = np.random.default_rng(9)
    T = {s: {} for s in SP}
    for r in R.itertuples():
        T[r.sp][r.ev] = r.t
    obs = flip_test(T, cort, rng, 20000)
    out = dict(observed=dict(tested=obs[0], flips=obs[1], p=obs[2]), sigma=sig, cortical_estimate=float(d[0]))
    for dc in [0.0, float(d[0]), 0.5, 1.0]:
        res = []
        for _ in range(300):
            ys = base + dc * MAM_EXTRA[:, 0] + rng.normal(0, sig, len(y))
            Ts = {s: {} for s in SP}
            for k, r in enumerate(R.itertuples()):
                Ts[r.sp][r.ev] = float(np.exp(ys[k]))
            res.append(flip_test(Ts, cort, rng, 2000))
        out[f'delta_{dc:.3f}'] = dict(power=float(np.mean([r[2] < 0.05 for r in res])),
                                      median_flips=float(np.median([r[1] for r in res])),
                                      median_tested=float(np.median([r[0] for r in res])))
        print('H', dc, out[f'delta_{dc:.3f}'], flush=True)
    return out


# ---------------------------------------------------------------- J: deduplication, matched windows
@section('J_dedup_matched')
def _():
    std5 = BLIMP[BLIMP['size'] != 'pythia-1b']
    Cs = curve_table(std5, max_step=120000)
    Cd = curve_table(BLIMP_DEDUP)
    Cd['size'] = Cd['size'].str.replace('-deduped', '', regex=False)
    Ps, Pd = panel_from(Cs), panel_from(Cd)
    ss = machine_summary(Ps, nperm=1000, ref='pythia-410m')
    sd = machine_summary(Pd, nperm=1000, ref='pythia-410m')
    rho_b, nb = spearman_beta(ss['beta'], sd['beta'])
    M = Ps.merge(Pd, on=['size', 'paradigm'], suffixes=('_s', '_d'))
    out = dict(standard_matched={k: v for k, v in ss.items() if k != 'beta'},
               dedup={k: v for k, v in sd.items() if k != 'beta'},
               event_spearman=rho_b, shared_events=nb, common_cells=int(len(M)),
               common_cell_pearson_log=float(np.corrcoef(M.x_s, M.x_d)[0, 1]),
               common_cell_median_abs_diff_log=float(np.median(np.abs(M.x_s - M.x_d))),
               common_cell_mean_diff_log=float(np.mean(M.x_d - M.x_s)))
    out['common_cell_by_size'] = {s: dict(n=int((M['size'] == s).sum()),
                                          mean_diff_log=float((M.x_d - M.x_s)[M['size'] == s].mean()))
                                  for s in SIZES[:5]}
    print('J', {k: v for k, v in out.items() if not isinstance(v, dict)}, flush=True)
    return out


# ---------------------------------------------------------------- K: no extrapolated crossings
@section('K_no_extrapolation')
def _():
    P = panel_from(PRIMARY_CURVES)
    inside = (P.x >= P.xmin) & (P.x <= P.xmax)
    P2 = P[inside]
    P2 = P2[P2.paradigm.map(P2.paradigm.value_counts()) >= 2]
    s = machine_summary(P2, nperm=1000)
    s['removed'] = int((~inside).sum())
    s['spearman_beta_primary'] = spearman_beta(s['beta'], PRIMARY['beta'])[0]
    del s['beta']
    print('K', {k: v for k, v in s.items() if not isinstance(v, dict)}, flush=True)
    return s


# ---------------------------------------------------------------- L: spiny mouse, lexical restarts
@section('L_stretch_checks')
def _():
    out = {}
    for drop in [None, 'Spiny.Mouse']:
        m = R if drop is None else R[R.sp != drop]
        m = m[m.ev.map(m.ev.value_counts()) >= 2]
        u, e = codes(m.sp.values, m.ev.values)
        y = np.log(m.t.values)
        _, _, p, _ = additive(u, e, y)
        _, g, _, ps = stretch(u, e, y)
        ul = pd.factorize(m.sp.values)[1]
        out[f'mammals_without_{drop}'] = dict(cells=int(len(y)), additive_r2=r2(y, p), stretch_r2=r2(y, ps),
                                             gain=r2(y, ps) - r2(y, p),
                                             stretches={str(l): float(v) for l, v in zip(ul, g / g.mean())})
    uu, ee, y = TABLES['lexical']
    u, e = codes(uu, ee)
    rng = np.random.default_rng(10)
    rs = []
    for _ in range(20):
        _, g, _, ps = stretch(u, e, y, g0=rng.uniform(0.5, 2, u.max() + 1))
        rs.append((r2(y, ps), float((g / g.mean()).min()), float((g / g.mean()).max())))
    out['lexical_restarts'] = dict(r2=[min(r[0] for r in rs), max(r[0] for r in rs)],
                                   min_stretch=[min(r[1] for r in rs), max(r[1] for r in rs)],
                                   max_stretch=[min(r[2] for r in rs), max(r[2] for r in rs)])
    print('L', out, flush=True)
    return out


print(f'total {time.time() - T0:.0f}s', flush=True)
save()
