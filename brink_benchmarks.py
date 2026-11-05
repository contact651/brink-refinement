#!/usr/bin/env python3
"""
Brink Model - Round 1 build - JOB 5 (part 1): the benchmark opponents named in Exam Pack v2.2 s.5.

All benchmarks forecast the SAME exam events (Exam Pack s.2) on the same country-months, except EWP (scored on
its own target) and VIEWS (scored on C1/C2 only). Rules for each were written here BEFORE any sealed-year run
(pre-registration; the sealed years are touched only by brink_score.py in Job 6).

  Naive        'same as last year': probability 1 if the place had a crisis on that lens in the last 12 months
               (GLOBAL: any C1-C7 crisis under way; DOMESTIC: a DOMESTIC event), else 0. Report only (not gated).
  Base rate    the place's own historical rate: (k + 12 g) / (n + 12), where n = the place's past forecast windows
               of this length already resolved at t, k = how many had an event, g = the same rate pooled over all
               places (B2, Guess: shrinkage of 12 windows so a short history is not read as 0% or 100%).
  PITF-style   Goldstone et al. 2010 four-variable model with V-Dem in place of Polity: regime class (closed
               autocracy = reference; electoral autocracy = partial autocracy; electoral democracy with / without
               high polarisation = partial democracy with / without factionalism; liberal democracy = full
               democracy), infant mortality vs world mean (log), 4+ neighbours with armed conflict, state-led
               discrimination (V-Dem social group equality in civil liberties, v2clsocgrp, sign reversed: the open
               stand-in for Minorities at Risk). Ordinary logistic regression per lens x horizon, re-fitted on
               windows ending by Dec 2014 (open years only); missing values = training median + flag.
  EWP          The Early Warning Project's published statistical risk, per country, from each yearly assessment
               file (2018..2026 files; 2014-2017 files not reachable from the cloud). Scored on EWP's own target
               (onset of state-led mass killing), with onsets derived from the same files (see ewp_target).
  VIEWS        Archived VIEWS country-month state-based forecasts from the public API (runs from Jan 2021; the 2018-2020
               archive is not on the API). Pre-registered mapping: for an origin t use the newest run issued in or
               before month t; monthly chance of 25+ state-based deaths = the run's dichotomous model output
               (the main dichotomous column: 'sc_cm_sb_dich_main' in fatalities001 runs, 'main_dich' in fatalities002/003 runs; r_2021 runs have none and are skipped); chance within h months = 1 - product(1 - monthly chance) over months
               t+1..t+h available in that run. Scored on C1/C2 events only. Report only.
  Markets      Not built: no dated public archive of prediction-market prices with a human-reviewed question
               crosswalk exists in the build. Report-only by the exam; recorded as unavailable.
"""
import json, os, sys, glob, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import brink_features as F
from brink_grok import load_outcomes, targets, history_features, H, LENSES

MODELS = os.path.join(ROOT, "data", "models")
FIT_END = pd.Period("2014-12", "M")
PITF_VARS = ["pa", "pdf", "pdn", "fd", "T1", "nbr4", "discrim"]


# ------------------------------------------------------------------ Naive
def naive(P, Hh):
    out = P[["iso3", "ym"]].merge(Hh, on=["iso3", "ym"], how="left")
    res = out[["iso3", "ym"]].copy()
    for lens in LENSES:
        s = out[f"in_crisis_{lens}"]
        for h in H:
            res[f"p_{lens}_{h}"] = s.values
    return res


# ------------------------------------------------------------------ Base rate
def base_rate(P, T, k_shrink=12):
    res = P[["iso3", "ym"]].copy()
    for h in H:
        Y = targets(T, h)
        for lens in LENSES:
            M = Y.set_index(["iso3", "ym"])[f"Y_{lens}_{h}"].unstack("ym")
            M = M.reindex(columns=pd.period_range(M.columns.min(), M.columns.max(), freq="M"))
            # a window from origin s is resolved at t if s + h <= t  -> shift by h
            A = M.T
            k = (A == 1).astype(float).cumsum().shift(h)
            n = A.notna().astype(float).cumsum().shift(h)
            gk = (A == 1).sum(axis=1).cumsum().shift(h); gn = A.notna().sum(axis=1).cumsum().shift(h)
            g = (gk / gn)
            p = (k.add(k_shrink * g, axis=0)).div(n + k_shrink, axis=0)
            s = F.to_long(p.T, "p").reset_index().rename(columns={"level_1": "ym"})
            res = res.merge(s.rename(columns={"p": f"p_{lens}_{h}"}), on=["iso3", "ym"], how="left")
    return res


# ------------------------------------------------------------------ PITF-style
def pitf_frame(isos=None):
    G = pd.read_parquet(os.path.join(ROOT, "data", "features", "grok_panel.parquet"))
    G["ym"] = pd.PeriodIndex(G.ym, freq="M")
    isos_, _ = F.universe()
    soc, _, _ = F.vdem_asof("v2clsocgrp", isos_)
    s = F.to_long(soc, "socgrp").reset_index(); s.columns = ["iso3", "ym", "socgrp"]
    G = G.merge(s, on=["iso3", "ym"], how="left")
    hp = (G.polar >= 1.0).astype(float).where(G.polar.notna())
    D = G[["iso3", "ym"]].copy()
    D["pa"] = G.reg1; D["pdf"] = G.reg2 * hp; D["pdn"] = G.reg2 * (1 - hp); D["fd"] = G.reg3
    D["T1"] = G.T1; D["nbr4"] = G.nbr4; D["discrim"] = -G.socgrp
    return D.sort_values(["iso3", "ym"]).reset_index(drop=True)

def pitf_X(D, med=None):
    X = D[PITF_VARS].copy()
    if med is None:
        med = {c: float(X[c].median()) for c in PITF_VARS}
    miss = X.isna().astype(float).add_prefix("m_")
    X = X.fillna(pd.Series(med))
    return pd.concat([X, miss[["m_pa", "m_T1", "m_nbr4", "m_discrim"]]], axis=1), med

def pitf_fit(D, T):
    params = {}
    for h in H:
        Y = D[["iso3", "ym"]].merge(targets(T, h), on=["iso3", "ym"], how="left")
        for lens in LENSES:
            y = Y[f"Y_{lens}_{h}"].values
            tr = (D.ym.values <= FIT_END - h) & ~np.isnan(y)
            X, med = pitf_X(D[tr])
            m = LogisticRegression(C=1e4, max_iter=2000).fit(X.values, y[tr])
            params[f"{lens}_{h}"] = {"features": list(X.columns), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
                                     "median": med, "n": int(tr.sum()), "events": int(y[tr].sum())}
    return params

def pitf_predict(D, params):
    res = D[["iso3", "ym"]].copy()
    for lens in LENSES:
        for h in H:
            md = params[f"{lens}_{h}"]
            X, _ = pitf_X(D, md["median"])
            z = md["intercept"] + X[md["features"]].values @ np.array(md["coef"])
            res[f"p_{lens}_{h}"] = 1 / (1 + np.exp(-z))
    return res


# ------------------------------------------------------------------ EWP
def ewp_forecasts():
    """One row per (iso3, assessment file Y): published risk; issued Dec Y-1 (vintage table, Guess); window {Y, Y+1}."""
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "clean", "ewp_sra_*.parquet"))):
        d = pd.read_parquet(p); Y = int(d.sra_year.iloc[0])
        rc = [c for c in d.columns if c.startswith("risk_in_") and c != "risk_in_2025_26"]
        col = rc[0] if rc else "risk_in_2025_26"
        for iso, r in zip(d.ISO3, pd.to_numeric(d[col], errors="coerce")):
            rows.append((iso, Y, r))
    return pd.DataFrame(rows, columns=["iso3", "window_start", "p_ewp"])

def ewp_target():
    """
    EWP's own onsets (state-led mass killing), derived from the assessment files (Guess, E1): files 2018-2023 carry
    'anymk.ongoing' for their data year (file year - 2): an onset in data year y = ongoing in y and not in y-1.
    Files 2024-2026 carry 'mk_onset_prev_year' for their data year: onset if > 0. Data years covered: 2016-2024.
    """
    by = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "clean", "ewp_sra_*.parquet"))):
        d = pd.read_parquet(p); Y = int(d.sra_year.iloc[0]); dy = Y - 2
        if "anymk.ongoing" in d.columns:
            by[dy] = ("ongoing", dict(zip(d.ISO3, pd.to_numeric(d["anymk.ongoing"], errors="coerce"))))
        elif "mk_onset_prev_year" in d.columns:
            by[dy] = ("onset", dict(zip(d.ISO3, pd.to_numeric(d["mk_onset_prev_year"], errors="coerce"))))
    rows = []
    for dy, (kind, m) in sorted(by.items()):
        for iso, v in m.items():
            if kind == "onset":
                rows.append((iso, dy, float(v > 0) if not np.isnan(v) else np.nan))
            else:
                prev = by.get(dy - 1)
                if prev and prev[0] == "ongoing":
                    pv = prev[1].get(iso, np.nan)
                    rows.append((iso, dy, float(v == 1 and pv == 0) if not (np.isnan(v) or np.isnan(pv)) else np.nan))
    return pd.DataFrame(rows, columns=["iso3", "year", "ewp_onset"])


# ------------------------------------------------------------------ VIEWS
def views_forecasts():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "views", "*.parquet")))
    out = []
    for f in files:
        d = pd.read_parquet(f)
        dc = [c for c in d.columns if "dich" in c and "main" in c]   # sc_cm_sb_dich_main (fatalities001) or main_dich (002/003)
        if not dc:
            continue
        run = d.run.iloc[0]
        parts = run.split("_")
        try:
            yr, mo = int(parts[-3]), int(parts[-2])
        except Exception:
            yr, mo = int(parts[1]), int(parts[2])
        issued = pd.Period(year=yr, month=max(mo, 1), freq="M")
        x = pd.DataFrame({"iso3": d.isoab, "fm": [pd.Period(year=int(a), month=int(b), freq="M") for a, b in zip(d.year, d.month)],
                          "pm": pd.to_numeric(d[dc[0]], errors="coerce"), "issued": issued, "run": run})
        out.append(x)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def views_predict(origins, V):
    """origins: DataFrame iso3, ym. Returns p_GLOBAL_h for C1/C2 scoring (same numbers used for DOMESTIC C1/C2)."""
    if V.empty:
        return pd.DataFrame()
    issues = sorted(V.issued.unique())
    res = []
    by_issue = {i: g.set_index(["iso3", "fm"]).pm for i, g in V.groupby("issued")}
    for t in sorted(origins.ym.unique()):
        cand = [i for i in issues if i <= t]
        if not cand:
            continue
        s = by_issue[cand[-1]]
        isos = origins[origins.ym == t].iso3.values
        row = {"iso3": isos, "ym": t}
        for h in H:
            ps = []
            for iso in isos:
                vals = [s.get((iso, t + k), np.nan) for k in range(1, h + 1)]
                vals = [v for v in vals if not np.isnan(v)]
                ps.append(1 - np.prod([1 - v for v in vals]) if vals else np.nan)
            row[f"p_GLOBAL_{h}"] = ps; row[f"p_DOMESTIC_{h}"] = ps
        res.append(pd.DataFrame(row))
    return pd.concat(res, ignore_index=True)


# ------------------------------------------------------------------ build all on chosen tables
def build(long_paths, target_paths, last_origin=None):
    L, T = load_outcomes(long_paths, target_paths)
    D = pitf_frame()
    if last_origin:
        D = D[D.ym <= pd.Period(last_origin, "M")].reset_index(drop=True)
    Hh = history_features(L, T)
    return L, T, D, Hh


if __name__ == "__main__":
    O = os.path.join(ROOT, "data", "outcomes")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "fit":
        L, T, D, Hh = build([os.path.join(O, "outcomes_long_open.csv.gz")], [os.path.join(O, "targets_open.csv.gz")])
        params = pitf_fit(D, T)
        js = json.dumps({"pitf_style": params}, indent=1)
        open(os.path.join(MODELS, "pitf_style_params.json"), "w").write(js)
        import hashlib; print("PITF-style params sha256", hashlib.sha256(js.encode()).hexdigest())
        E = ewp_target(); E.to_csv(os.path.join(MODELS, "ewp_target_derived.csv"), index=False)
        print("EWP derived target rows (not printed: sealed-period outcomes):", len(E))
