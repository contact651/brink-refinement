#!/usr/bin/env python3
"""
Brink Refinement Loop - weekly cycle harness (builder). Cycle 1: Mon 26 Oct 2026.

PRACTICE TEST (plans/brink-refinement-loop.md): every idea is tried on the practice years only.
  - The Working Model (v1.0 = Claude design, Round 1 leader, brink_claude.py, unchanged) is re-fitted the same way it
    was for the exam, shifted 15 years back: fitted on forecast windows that end by Dec 1999, beta calibration on
    1995-1999 out-of-sample forecasts from a model fitted to Dec 1994, penalty held at the locked v1.0 values
    (h=1: C=0.1, else 0.01). Scored on forecast months 2000-01 .. 2014-12 minus h, open outcome tables only.
  - The 2015-2025 years are never touched here.
  - A variant = the Working Model plus extra signals; everything else identical.

KEEP RULE (written before any idea was tested, cycle 1):
  K1 at least one of the 12 gate cells (PR-AUC and Brier x 3/6/12 months x GLOBAL/DOMESTIC) improves significantly
     (one-sided country-year block bootstrap, B=500, Holm across the 12 cells, alpha 0.05);
  K2 no gate cell and no 1/18/24-month Brier cell is significantly worse (one-sided p < 0.05, no correction: stricter);
  K3 no continent is significantly worse on Brier at 12 months on either lens (one-sided p < 0.05);
  otherwise DROP (a tie goes to the simpler model).
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import brink_claude as BC
from brink_grok import targets, H, LENSES
BC.enet = lambda C: LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=C, solver="saga", max_iter=500, tol=1e-3, random_state=0)

P_TRAIN_END = pd.Period("1999-12", "M")
P_CAL_TRAIN_END = pd.Period("1994-12", "M")
P_CAL_FROM = pd.Period("1995-01", "M")
SCORE_FROM, SCORE_TO = pd.Period("2000-01", "M"), pd.Period("2014-12", "M")
C_LOCK = {1: 0.1, 3: 0.01, 6: 0.01, 12: 0.01, 18: 0.01, 24: 0.01}
GATE_H = [3, 6, 12]
B = 500
SEED = 20261026
OUT = os.path.join(ROOT, "cycle", "out"); os.makedirs(OUT, exist_ok=True)


def design(P, lens, extra, stats=None):
    X, st = BC.design(P, lens, None if stats is None else stats["base"])
    st2 = {"base": st, "extra": {}} if stats is None else stats
    for c in extra:
        x = P[c].astype(float)
        if stats is None:
            st2["extra"][c] = (float(np.nanmean(x)), float(np.nanstd(x) or 1.0))
        m, s = st2["extra"][c]
        X["miss_" + c] = x.isna().astype(float).values
        X[c] = ((x.fillna(m) - m) / s).values
    return X, st2


def fit_predict(P, Ys, extra, horizons=H):
    """Fit per lens x horizon on practice training years; return forecasts for 2000-2014 origins (PAV across h)."""
    ym = P.ym.values
    wt = BC.wts(P)
    sc = (P.ym >= SCORE_FROM) & (P.ym <= SCORE_TO)
    out = P.loc[sc, ["iso3", "ym", "continent"]].copy()
    for lens in LENSES:
        mats = []
        for h in H:
            yv = Ys[h][f"Y_{lens}_{h}"].values
            if "x_own" in extra:
                b = P[f"br_{lens}_{h}"].clip(1e-4, 1 - 1e-4)
                P = P.assign(x_own=np.log(b / (1 - b)))
            tr0 = (ym <= P_CAL_TRAIN_END - h) & ~np.isnan(yv)
            ca = (ym >= P_CAL_FROM) & (ym <= P_TRAIN_END - h) & ~np.isnan(yv)
            X0, s0 = design(P[tr0], lens, extra)
            m0 = BC.enet(C_LOCK[h]).fit(X0.values, yv[tr0], sample_weight=wt[tr0])
            Xc, _ = design(P[ca], lens, extra, s0)
            cal = BC.beta_cal_fit(m0.predict_proba(Xc.values)[:, 1], yv[ca])
            tr = (ym <= P_TRAIN_END - h) & ~np.isnan(yv)
            X, s = design(P[tr], lens, extra)
            m = BC.enet(C_LOCK[h]).fit(X.values, yv[tr], sample_weight=wt[tr])
            Xs, _ = design(P[sc], lens, extra, s)
            p = BC.beta_cal_apply(m.predict_proba(Xs.values)[:, 1], cal)
            mats.append(p)
            coefs = dict(zip(X.columns, m.coef_[0].round(4)))
            out.attrs.setdefault("coefs", {})[f"{lens}_{h}"] = {k: float(v) for k, v in coefs.items() if k in extra}
        M = BC.pav_rows(np.vstack(mats).T)
        for k, h in enumerate(H):
            out[f"p_{lens}_{h}"] = np.clip(M[:, k], 1e-4, 1 - 1e-4)
    return out


# ------------------------------------------------------------------ scoring
def brier(y, p): return float(np.mean((p - y) ** 2))
def prauc(y, p): return float(average_precision_score(y, p)) if y.sum() > 0 else np.nan
MET = {"prauc": (prauc, 1), "brier": (brier, -1)}

def joined(f, Ys, lens, h):
    Y = Ys[h][["iso3", "ym", f"Y_{lens}_{h}"]]
    d = f[["iso3", "ym", "continent", f"p_{lens}_{h}"]].merge(Y, on=["iso3", "ym"])
    d = d[(d.ym >= SCORE_FROM) & (d.ym <= SCORE_TO - h)].dropna(subset=[f"Y_{lens}_{h}"])
    return d.rename(columns={f"p_{lens}_{h}": "p", f"Y_{lens}_{h}": "y"})

def blocks(d):
    key = d.iso3.astype(str) + "_" + d.ym.dt.year.astype(str)
    codes, u = pd.factorize(key)
    order = np.argsort(codes, kind="stable"); bounds = np.searchsorted(codes[order], np.arange(len(u) + 1))
    return [order[bounds[k]:bounds[k + 1]] for k in range(len(u))]

def boot(y, pa, pb, met, bl, rng):
    fn, sign = MET[met]
    obs = sign * (fn(y, pa) - fn(y, pb))
    nb = len(bl); better = worse = 0
    for _ in range(B):
        ix = np.concatenate([bl[k] for k in rng.integers(0, nb, nb)])
        dl = sign * (fn(y[ix], pa[ix]) - fn(y[ix], pb[ix]))
        better += not (dl > 0); worse += not (dl < 0)
    return obs, (1 + better) / (B + 1), (1 + worse) / (B + 1)

def holm(p, alpha=0.05):
    o = np.argsort(p); m = len(p); r = np.zeros(m, bool)
    for k, i in enumerate(o):
        if p[i] <= alpha / (m - k): r[i] = True
        else: break
    return r

def compare(fa, fb, Ys, name):
    """fa = variant, fb = Working Model. Returns table + verdict under the keep rule."""
    rng = np.random.default_rng(SEED)
    rows = []
    for lens in LENSES:
        for h in H:
            a = joined(fa, Ys, lens, h); b = joined(fb, Ys, lens, h)[["iso3", "ym", "p"]].rename(columns={"p": "pb"})
            d = a.merge(b, on=["iso3", "ym"]); bl = blocks(d)
            for met in (["prauc", "brier"] if h in GATE_H else ["brier"]):
                obs, pbet, pwor = boot(d.y.values, d.p.values, d.pb.values, met, bl, rng)
                fn = MET[met][0]
                rows.append({"variant": name, "lens": lens, "h": h, "metric": met, "model": fn(d.y.values, d.pb.values),
                             "variant_score": fn(d.y.values, d.p.values), "diff_better": obs, "p_better": pbet, "p_worse": pwor,
                             "n": len(d), "events": int(d.y.sum()), "gate": h in GATE_H})
    T = pd.DataFrame(rows)
    g = T[T.gate].copy(); g["sig_better"] = holm(g.p_better.values) & (g.diff_better > 0)
    T = T.merge(g[["lens", "h", "metric", "sig_better"]], on=["lens", "h", "metric"], how="left")
    T["sig_worse"] = (T.p_worse < 0.05) & (T.diff_better < 0)
    reg = []
    for lens in LENSES:
        a = joined(fa, Ys, lens, 12); b = joined(fb, Ys, lens, 12)[["iso3", "ym", "p"]].rename(columns={"p": "pb"})
        d = a.merge(b, on=["iso3", "ym"])
        for c, g2 in d.groupby("continent"):
            if g2.y.sum() == 0: continue
            obs, pbet, pwor = boot(g2.y.values, g2.p.values, g2.pb.values, "brier", blocks(g2), rng)
            reg.append({"variant": name, "lens": lens, "continent": c, "diff_better": obs, "p_worse": pwor,
                        "sig_worse": bool(pwor < 0.05 and obs < 0)})
    R = pd.DataFrame(reg)
    k1 = bool(T.sig_better.fillna(False).any()); k2 = not bool(T.sig_worse.any()); k3 = not bool(R.sig_worse.any())
    return T, R, {"K1_some_gate_cell_significantly_better": k1, "K2_no_cell_significantly_worse": k2,
                  "K3_no_continent_worse": k3, "KEEP": k1 and k2 and k3}


def summary_scores(f, Ys):
    r = {}
    for lens in LENSES:
        for h in H:
            d = joined(f, Ys, lens, h)
            r[f"{lens}_{h}"] = {"prauc": prauc(d.y.values, d.p.values), "brier": brier(d.y.values, d.p.values), "n": len(d), "events": int(d.y.sum())}
    return r
