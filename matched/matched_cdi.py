"""Matched-word timing: children and Pythia on the same CDI words.

The One Clock paper times children on Wordbank lemmas and Pythia on BLiMP
paradigms, so no event is shared across the two systems. This script times
both on the same words. Child times are Wordbank WS age-of-acquisition norms
(child_aoa.tsv, the Chang & Bergen table); machine times are word acquisition
crossings on surprisal trajectories from an earlier scoring run
(results_twins-*.jsonl: 696 CDI words x 22 checkpoints x Pythia 14M-410M,
10 seeds per size). No model is run here.

Prespecified before the run (2026-11-03):
  S1 primary   Spearman rho between American English production log AoA and
               the machine word effect b_M from a log-additive fit over the five
               seed-0 Pythia sizes; two-sided p from 10,000 word-label
               permutations (seed 0).
  S2           noise ceilings: seed-panel reliability of b_M (seed k panel vs
               seed j panel, all 45 pairs) and AmE-vs-AusE production agreement;
               disattenuated rho is descriptive.
  S3           partial Spearman of S1 controlling log word frequency (wordfreq
               Zipf, en) and word length; each system's frequency correlation.
  S4           paper framework: joint log-additive fit over AmE production plus
               the five sizes, rho with the within-unit permutation null
               (1,000, seed 0), and a cross-system null that shuffles only the
               child rows (1,000, seed 1).
  S5           lexical class: class means of z-scored effects per system, and a
               class-by-system test on z_M - z_C (eta^2, 10,000 label perms).
  Secondary    S1 with Australian English production, AmE comprehension, and
               b_M from all 50 seed runs.

Deviations recorded after run v2 crashed (2026-11-03, before any S1 value
was seen): the logistic fits put mommy and daddy at negative AoA (-1.6 and
-1.4 months), which have no log, so words with AoA <= 0 are dropped from every
analysis; and the WS "understands" rows in child_aoa.tsv copy the "produces"
AoAs exactly (651/651 words), so the AmE comprehension secondary is not
estimable from this file and is reported as such.
"""
import glob
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working/out"))
OUT.mkdir(parents=True, exist_ok=True)
SIZES = ["14m", "31m", "70m", "160m", "410m"]
TOKENS_PER_STEP = 2_097_152
N_PERM_WORD = 10_000
N_PERM_FIT = 1_000
MIN_UNITS = 2

log_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_lines.append(s)


# Kaggle decompresses uploaded .gz files, so each input may appear with or without
# the suffix; the hash is recorded for whichever file was read.
INPUT_RUNS = ["results_twins-cpuall.jsonl", "results_twins-cpu31m.jsonl"]


def find(name):
    hits = [h for n in (name, name + ".gz")
            for h in glob.glob(f"/kaggle/input/**/{n}", recursive=True) + glob.glob(n)]
    if not hits:
        raise FileNotFoundError(f"{name} not found under /kaggle/input or cwd; attach the "
                                "oneclock-matched-cdi-inputs dataset")
    return hits[0]


def load_runs():
    runs = defaultdict(dict)
    for name in INPUT_RUNS:
        path = find(name)
        with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if "fixture" in d["model"] or "synthetic" in d["model"]:
                    continue
                runs[d["model"]][d["step"]] = d
    return runs


# surp_traj and acquisition_step are copied verbatim from the scoring run's
# analysis code so the machine crossings keep its acquisition definition.
def surp_traj(recs, word, steps):
    out = []
    for s in steps:
        v = recs[s]["cdi"].get(word)
        if v is None:
            out.append(None)
            continue
        pile, n, frame = v
        out.append(pile if (pile is not None and n >= 2) else frame)
    return out


def acquisition_step(traj, steps):
    xs = [(s, v) for s, v in zip(steps, traj) if v is not None]
    if len(xs) < 6:
        return None
    ss, vs = zip(*xs)
    early = np.mean([v for s, v in xs if s <= 64])
    floor = min(vs)
    if early - floor < 1.0:
        return None
    thresh = (early + floor) / 2
    for i in range(1, len(xs) - 1):
        if vs[i] <= thresh and vs[i + 1] <= thresh:
            lo_s = math.log10(max(ss[i - 1], 1))
            hi_s = math.log10(max(ss[i], 1))
            lo_v, hi_v = vs[i - 1], vs[i]
            if lo_v <= thresh or hi_v >= lo_v:
                return hi_s
            frac = (lo_v - thresh) / (lo_v - hi_v)
            return lo_s + frac * (hi_s - lo_s)
    return None


def run_name(size, seed):
    return f"EleutherAI/pythia-{size}" + ("" if seed == 0 else f"-seed{seed}")


def machine_table(runs, run_names):
    """Long table of ln(tokens) at acquisition for each (run, word)."""
    rows, n_none = [], {}
    for r in run_names:
        if r not in runs:
            raise KeyError(f"run {r} missing from the scoring-run results")
        recs = runs[r]
        steps = sorted(recs)
        words = sorted(recs[steps[-1]]["cdi"])
        none = 0
        for w in words:
            a = acquisition_step(surp_traj(recs, w, steps), steps)
            if a is None:
                none += 1
                continue
            rows.append((r, w, a * math.log(10) + math.log(TOKENS_PER_STEP)))
        n_none[r] = none
    return pd.DataFrame(rows, columns=["unit", "event", "y"]), n_none


dropped_nonpositive = []


def child_table(path):
    """First-occurrence CleanedSingle rule, as in the scoring run's stimulus builder."""
    d = pd.read_csv(path, sep="\t", dtype=str)
    d = d[d["form"] == "WS"]
    out = {}
    for (lang, meas), g in d.groupby(["language", "measure"], sort=False):
        best = {}
        for _, r in g.iterrows():
            w = str(r["CleanedSingle"]).strip().lower()
            if not w or " " in w or not re.fullmatch(r"[a-z][a-z'-]*", w):
                continue
            if w in best or r["aoa"] in ("", "NA") or pd.isna(r["aoa"]):
                continue
            if float(r["aoa"]) <= 0:
                dropped_nonpositive.append((lang, meas, w, float(r["aoa"])))
                best[w] = None
                continue
            best[w] = (float(r["aoa"]), r["lexical_class"])
        best = {k: v for k, v in best.items() if v is not None}
        out[(lang, meas)] = best
    return out


def keep_multi(df):
    c = df.groupby("event")["unit"].nunique()
    return df[df["event"].isin(c[c >= MIN_UNITS].index)].reset_index(drop=True)


class Additive:
    """log t = a_unit + b_event by OLS; fitted values via an orthonormal basis.

    With the design fixed, any permutation of y is refit as Q @ (Q.T @ y).
    """

    def __init__(self, df):
        self.units = sorted(df["unit"].unique())
        self.events = sorted(df["event"].unique())
        ui = df["unit"].map({u: i for i, u in enumerate(self.units)}).to_numpy()
        ei = df["event"].map({e: i for i, e in enumerate(self.events)}).to_numpy()
        n, U, E = len(df), len(self.units), len(self.events)
        X = np.zeros((n, 1 + (U - 1) + (E - 1)))
        X[:, 0] = 1
        for k in range(n):
            if ui[k] > 0:
                X[k, ui[k]] = 1
            if ei[k] > 0:
                X[k, U - 1 + ei[k]] = 1
        u, s, _ = np.linalg.svd(X, full_matrices=False)
        rank = int((s > s[0] * 1e-10).sum())
        if rank < X.shape[1]:
            raise ValueError(f"design rank {rank} < {X.shape[1]} columns; the unit-event "
                             "graph is disconnected or an event has no variation")
        self.Q = u[:, :rank]
        self.X, self.ui, self.ei, self.y = X, ui, ei, df["y"].to_numpy(float)
        coef = np.linalg.lstsq(X, self.y, rcond=None)[0]
        b = np.concatenate([[0.0], coef[U:]])
        self.b = pd.Series(b - b.mean(), index=self.events)

    def fitted(self, y):
        return self.Q @ (self.Q.T @ y)

    def rho(self, y=None):
        y = self.y if y is None else y
        return float(np.corrcoef(y, self.fitted(y))[0, 1])


def within_unit_null(m, n, seed, mask=None):
    """Shuffle times among events inside each unit (or only rows in mask)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero((m.ui == u) & (True if mask is None else mask))
              for u in range(len(m.units))]
    groups = [g for g in groups if len(g) > 1]
    out = np.empty(n)
    for i in range(n):
        y = m.y.copy()
        for g in groups:
            y[g] = y[rng.permutation(g)]
        out[i] = m.rho(y)
    return out


def null_summary(obs, null):
    return {"obs": obs, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "z": float((obs - null.mean()) / null.std(ddof=1)),
            "p_ge": float((1 + (null >= obs).sum()) / (1 + len(null)))}


def perm_spearman(x, y, n, seed):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    obs = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)])
    return {"rho": obs, "n": int(len(x)),
            "p_two_sided": float((1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (1 + n)),
            "pearson_r": float(np.corrcoef(x, y)[0, 1])}


def residual_ranks(v, covs):
    A = np.column_stack([np.ones(len(v))] + [stats.rankdata(c) for c in covs])
    r = stats.rankdata(v)
    return r - A @ np.linalg.lstsq(A, r, rcond=None)[0]


def machine_effects(runs, seeds):
    df, n_none = machine_table(runs, [run_name(s, k) for s in SIZES for k in seeds])
    df = keep_multi(df)
    return Additive(df), n_none


def main():
    runs = load_runs()
    log("runs loaded:", len(runs))
    child = child_table(find("child_aoa.tsv"))
    ame_p = child[("English (American)", "produces")]
    ame_c = child[("English (American)", "understands")]
    aus_p = child[("English (Australian)", "produces")]
    res = {"prespecified": __doc__.split("Prespecified")[1].strip(),
           "input_sha256": {os.path.basename(find(n)): hashlib.sha256(open(find(n), "rb").read()).hexdigest()
                            for n in INPUT_RUNS + ["child_aoa.tsv"]}}

    # machine word effects, seed-0 panel (the paper's standard runs)
    m0, none0 = machine_effects(runs, [0])
    res["machine_seed0"] = {"n_words": len(m0.events), "n_cells": int(len(m0.y)),
                            "not_acquired_per_run": none0, "rho_additive": m0.rho()}
    log("seed-0 machine panel:", res["machine_seed0"])
    bM = m0.b

    def s1(childmap, label, b=bM, seed=0):
        w = sorted(set(b.index) & set(childmap))
        c = np.log([childmap[x][0] for x in w])
        r = perm_spearman(c, b[w].to_numpy(), N_PERM_WORD, seed)
        log(label, r)
        return r, w

    res["S1_ame_production"], words = s1(ame_p, "S1 AmE production")
    res["S1_aus_production"], _ = s1(aus_p, "S1 AusE production")
    same = sum(ame_c[w][0] == ame_p[w][0] for w in ame_c if w in ame_p)
    res["S1_ame_comprehension"] = {"not_estimable": True, "n_identical_to_production": same,
                                   "n_words": len(ame_c)}
    log("S1 AmE comprehension not estimable:", res["S1_ame_comprehension"])
    res["dropped_nonpositive_aoa"] = dropped_nonpositive

    # S2 machine reliability: seed-k panel vs seed-j panel, all pairs
    panels = {}
    for k in range(10):
        try:
            panels[k] = machine_effects(runs, [k])[0].b
        except KeyError as e:
            log("seed panel skipped:", k, e)
    pair_r = []
    for i in panels:
        for j in panels:
            if i < j:
                w = sorted(set(panels[i].index) & set(panels[j].index))
                pair_r.append(stats.spearmanr(panels[i][w], panels[j][w])[0])
    rel_m = float(np.mean(pair_r))
    wd = sorted(set(ame_p) & set(aus_p))
    rel_c = float(stats.spearmanr([ame_p[x][0] for x in wd], [aus_p[x][0] for x in wd])[0])
    rho1 = res["S1_ame_production"]["rho"]
    res["S2_ceilings"] = {
        "machine_seed_panel_rho_mean": rel_m, "machine_seed_panel_rho_min": float(np.min(pair_r)),
        "machine_seed_panel_rho_max": float(np.max(pair_r)), "n_panel_pairs": len(pair_r),
        "child_ame_vs_aus_production_rho": rel_c, "n_child_shared": len(wd),
        "disattenuated_S1": float(rho1 / math.sqrt(rel_m * rel_c)) if rel_m > 0 and rel_c > 0 else None,
        "note": "AmE-vs-AusE agreement stands in for child reliability; dialect differences count as noise, so the disattenuated value is descriptive"}
    log("S2", res["S2_ceilings"])

    # all 50 seed runs as a secondary machine estimator
    mall, _ = machine_effects(runs, list(range(10)))
    res["S1_ame_production_allseeds"], _ = s1(ame_p, "S1 AmE production, all seeds", b=mall.b, seed=2)

    # S3 frequency and length controls
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wordfreq"], check=True,
                       timeout=600)
        from wordfreq import zipf_frequency
        freq = np.array([zipf_frequency(x, "en") for x in words])
        length = np.array([len(x) for x in words])
        c = np.log([ame_p[x][0] for x in words])
        b = bM[words].to_numpy()
        rc, rb = residual_ranks(c, [freq, length]), residual_ranks(b, [freq, length])
        obs = float(np.corrcoef(rc, rb)[0, 1])
        rng = np.random.default_rng(3)
        null = np.array([np.corrcoef(rc, rng.permutation(rb))[0, 1] for _ in range(N_PERM_WORD)])
        res["S3_controls"] = {
            "partial_rho_freq_length": obs,
            "p_two_sided": float((1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (1 + N_PERM_WORD)),
            "child_vs_freq_rho": float(stats.spearmanr(c, freq)[0]),
            "machine_vs_freq_rho": float(stats.spearmanr(b, freq)[0]),
            "child_vs_length_rho": float(stats.spearmanr(c, length)[0]),
            "machine_vs_length_rho": float(stats.spearmanr(b, length)[0]),
            "n_zero_freq": int((freq == 0).sum())}
        freq_map = dict(zip(words, freq))
    except Exception as e:
        res["S3_controls"] = {"not_computed": f"{type(e).__name__}: {e}"}
        freq_map = {}
    log("S3", res["S3_controls"])

    # S4 joint log-additive fit in the paper's framework
    mdf, _ = machine_table(runs, [run_name(s, 0) for s in SIZES])
    cdf = pd.DataFrame([("child_AmE_produces", w, math.log(ame_p[w][0])) for w in ame_p],
                       columns=["unit", "event", "y"])
    joint = keep_multi(pd.concat([mdf, cdf], ignore_index=True))
    jm = Additive(joint)
    is_child = (joint["unit"] == "child_AmE_produces").to_numpy()
    obs = jm.rho()
    res["S4_joint"] = {
        "n_cells": int(len(joint)), "n_words": len(jm.events),
        "within_unit_null": null_summary(obs, within_unit_null(jm, N_PERM_FIT, 0)),
        "child_rows_only_null": null_summary(obs, within_unit_null(jm, N_PERM_FIT, 1, is_child)),
        "child_rows_fit_r": float(np.corrcoef(jm.y[is_child], jm.fitted(jm.y)[is_child])[0, 1])}
    log("S4", res["S4_joint"])

    # S5 lexical class by system
    cls = pd.Series({w: ame_p[w][1] for w in words})
    zc = pd.Series(stats.zscore(np.log([ame_p[w][0] for w in words])), index=words)
    zm = pd.Series(stats.zscore(bM[words].to_numpy()), index=words)
    tab = pd.DataFrame({"cls": cls, "zc": zc, "zm": zm})
    res["S5_class_means"] = tab.groupby("cls").agg(n=("zc", "size"), child_z=("zc", "mean"),
                                                   machine_z=("zm", "mean")).round(3).to_dict("index")
    diff, lab = (tab.zm - tab.zc).to_numpy(), tab.cls.to_numpy()

    def eta2(labels):
        g = pd.Series(diff).groupby(labels)
        return float((g.size() * (g.mean() - diff.mean()) ** 2).sum() / ((diff - diff.mean()) ** 2).sum())

    obs = eta2(lab)
    rng = np.random.default_rng(4)
    null = np.array([eta2(rng.permutation(lab)) for _ in range(N_PERM_WORD)])
    res["S5_class_by_system"] = {"eta2": obs, "p_ge": float((1 + (null >= obs).sum()) / (1 + N_PERM_WORD))}
    log("S5", res["S5_class_means"], res["S5_class_by_system"])

    # matched table for the paper and release
    mt = pd.DataFrame({"word": words, "lexical_class": cls[words].to_numpy(),
                       "ame_produces_aoa": [ame_p[w][0] for w in words],
                       "aus_produces_aoa": [aus_p.get(w, (np.nan,))[0] for w in words],
                       "ame_understands_aoa": [ame_c.get(w, (np.nan,))[0] for w in words],
                       "machine_effect_seed0": bM[words].to_numpy(),
                       "machine_effect_allseeds": mall.b.reindex(words).to_numpy(),
                       "zipf_freq": [freq_map.get(w, np.nan) for w in words]})
    for s in SIZES:
        r = run_name(s, 0)
        sub = mdf[mdf.unit == r].set_index("event")["y"]
        mt[f"ln_tokens_{s}"] = sub.reindex(words).to_numpy()
    mt.to_csv(OUT / "matched_words.csv", index=False)
    (OUT / "matched_results.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / "log.txt").write_text("\n".join(log_lines))
    log("done")


if __name__ == "__main__":
    main()
