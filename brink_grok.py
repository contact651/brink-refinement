#!/usr/bin/env python3
"""
Brink Model - Round 1 build - JOB 2: Grok 4.7's design ("The last block"), as a working program.

Design: briefs/brink-design-grok.md (verbatim entry, 18 Oct 2026). Exam: Exam Pack v2.2 (SHA-256 bd3d64f1...ebaec1).
Built to the design as written; where it is unclear the smallest reasonable choice is taken and listed in
GROK_CHOICES below (all Guess). Nothing here was tuned on 2015-2025.

What it does (plain words)
  1. Builds Grok's signals for every country and month from data public at that date (brink_features.py):
     violence speeding up (S1), income fall (S3), refugee jump (S4), state violence (C2), infant mortality vs the
     world (T1), share of neighbours at war, regime class x polarisation, the published PITF score, and the
     'already in crisis' flag; the stress x containment interaction L = S x C; a missing-data flag per signal.
  2. Puts each signal on the GLOBAL lens (vs the population-weighted world that month) and the DOMESTIC lens
     (vs the place's own last 120 months), separately.
  3. Fits one elastic-net logistic model per lens and horizon (1,3,6,12,18,24 months) on forecast months up to
     Dec 1999 only, penalty chosen by walk-forward inside 1980-1999, then Platt scaling on 1990-1999, caps 1%-99%.
     Coefficients are then LOCKED (saved to grok_params.json).
  4. Predicts every forecast month with the locked coefficients. The target is the EXAM's crisis event
     (C1-C7 onset or escalation, Exam Pack s.2) on each lens, not Grok's own crisis cuts (scorecard issue 5).
  5. Atrocity Alarm: separate model on the exam's C2 onsets per country-year, threshold = lowest cut with
     <= 1 false alarm per 10 training non-cases.

Usage:
  python3 brink_grok.py fit          # build features, fit on open years, save locked params
  python3 brink_grok.py predict OUT  # forecasts for all origins whose outcome history is available
"""
import json, os, sys, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import brink_features as F

FEAT = os.path.join(ROOT, "data", "features"); os.makedirs(FEAT, exist_ok=True)
MODELS = os.path.join(ROOT, "data", "models"); os.makedirs(MODELS, exist_ok=True)
H = [1, 3, 6, 12, 18, 24]
LENSES = ["GLOBAL", "DOMESTIC"]
TRAIN_END = pd.Period("1999-12", "M")
PLATT_FROM = pd.Period("1990-01", "M")
C_GRID = [0.001, 0.01, 0.1, 1.0]
CV_YEARS = list(range(1980, 1997, 3))      # 1980, 1983, ..., 1995 (Guess G9: compute)
POLAR_CUT = 1.0                            # Guess G4: v2cacamps >= 1.0 on V-Dem's latent scale = 'high polarisation'
PITF_COEF = {"partial_autocracy": 1.85, "partial_dem_faction": 3.61, "partial_dem_nofaction": 1.83,
             "full_democracy": 0.981, "imr": 1.59, "nbr4": 3.09}   # Goldstone et al. 2010, Table 1, full problem set (Confirmed: paper)

GROK_CHOICES = [
    "G1 Target: the exam's C1-C7 onset/escalation event on each lens (Exam Pack s.2), not Grok's own crisis cuts (10% GDP, 0.10 civil liberties, 1% refugees). Grok's cuts are not built as a target.",
    "G2 Death data before June 2015 have no vintage: GED 4.0 with 'year y usable from June y+1' (hindsight-risk flag). So in open years S1 compares the last 3 vs 12 months of the newest usable year, not the last 3 calendar months.",
    "G3 Infant mortality from World Bank WDI (UN IGME series) as published, instead of UN WPP 2024 (WPP has no vintages). World mean = unweighted mean across countries in the same edition and year.",
    "G4 Regime class from V-Dem v2x_regime (0 closed autocracy, 1 electoral autocracy, 2 electoral democracy, 3 liberal democracy) x high polarisation (v2cacamps >= 1.0 on V-Dem's latent scale; the codebook's ordinal top categories). Polarised half-open = electoral autocracy or electoral democracy AND high polarisation.",
    "G5 Neighbour share = share of land neighbours (today's borders, GeoDataSource) with an active UCDP/PRIO conflict in the latest usable year (June y+1 rule).",
    "G6 PITF roster term = the published Goldstone 2010 Table 1 linear score on V-Dem analogues (partial autocracy = electoral autocracy; partial democracy with factionalism = electoral democracy + high polarisation; full democracy = liberal democracy; infant mortality vs world mean; 4+ neighbours at war). State-led discrimination left out (no open MAR data; Grok refuses identity measures). The linear score is used in place of logit(p) (the conditional-logit model has no intercept); the fitted intercept absorbs the difference.",
    "G7 EWP and prediction-market roster terms are dropped (coefficient 0): no EWP forecasts exist before 2014 and no market prices before ~2015, so they cannot be fitted on origins up to Dec 1999 (the scorecard flagged this).",
    "G8 'Already in crisis' (GLOBAL) = any C1-C7 crisis month (in_crisis) in t-11..t; (DOMESTIC) = any DOMESTIC event in t-11..t. Month t counts as known at the end of month t.",
    "G9 Penalty chosen on walk-forward folds y = 1980, 1983, ..., 1995 (every third year of Grok's 1980-1996 range) on quarterly forecast months, to keep compute to minutes; one penalty per horizon, minimising the Brier score averaged over both lenses. The final fit uses every month.",
    "G10 DOMESTIC standardisation denominator = max(1.4826 x MAD, 0.1 x training SD) so constant yearly series do not divide by zero.",
    "G11 Censorship flag = freedom-of-expression index (v2x_freexp_altinf) down 0.10 or more over two published years AND reported deaths in the last 3 usable months at zero after a positive 12-month count; violence features then set to missing.",
    "G12 Atrocity Alarm label = the exam's C2 onset per country-year (the exam scores the alarm on C2); EWP's own list is used for the EWP benchmark in Job 5. Forecast for year y issued in December y-1 at 12 months. Threshold = the LOWEST probability cut whose false-alarm rate on training non-cases is <= 0.10 ('highest threshold' read as the boundary of the cap; a literal highest threshold would never fire).",
    "G13 Elastic-net mixing 0.5 (locked by Grok); sklearn saga solver; features standardised with training-window mean and SD; missing values set to 0 after standardisation plus a missing flag.",
]


# ------------------------------------------------------------------ 1. features
def build_features():
    isos, cy = F.universe()
    M = F.MONTHS
    pop, popy, h_pop = F.wdi_asof("population", isos)
    d1, d3, d12, lagm, h_ged = F.deaths_asof(isos)
    d3 = d3.reindex([F.ucdp_iso(i) for i in isos]); d3.index = isos
    d12 = d12.reindex([F.ucdp_iso(i) for i in isos]); d12.index = isos
    r3 = d3 * 4 / pop * 1e5; r12 = d12 / pop * 1e5
    S1 = np.log1p(r3) - np.log1p(r12)
    g, gy, h_gdp = F.wdi_asof("gdp_pc_growth", isos)
    S3 = (-g).clip(lower=0)
    # refugees + asylum seekers by origin, yearly change per person (lag rule July y+1)
    u = F.rd("unhcr_by_origin")
    u["flight"] = u.refugees.fillna(0) + u.asylum_seekers.fillna(0)
    u.loc[u.refugees.isna() & u.asylum_seekers.isna(), "flight"] = np.nan
    w = F.rd("wdi_current"); wp = w[w.indicator == "population"].set_index(["iso3", "year"]).value
    u = u.set_index(["iso3", "year"]).sort_index()
    ch = u.flight.groupby(level=0).diff()
    rate = (ch / wp.reindex(ch.index)).clip(lower=0)
    S4, _ = F.yearly_lagged(rate.rename("S4"), isos, 7)
    clphy, _, h_v = F.vdem_asof("v2x_clphy", isos)
    C2 = 1 - clphy
    imr, imry, _ = F.wdi_asof("infant_mortality", isos)
    wm = F.wdi_world_mean_asof("infant_mortality", "mean")
    # world mean for the same edition/year as each value
    w_ed = F.rd("wdi_vintages"); eds = sorted(w_ed.edition.unique())
    edp = {e: pd.Period(f"{e[:4]}-{e[4:6]}", "M") for e in eds}
    T1 = pd.DataFrame(np.nan, index=isos, columns=M)
    for t in M:
        cur = [e for e in eds if edp[e] <= t]; e = cur[-1] if cur else eds[0]
        yrs = imry[t]
        means = yrs.map(lambda y: wm.get((e, int(y)), np.nan) if not np.isnan(y) else np.nan)
        T1[t] = np.log(imr[t]) - np.log(means.astype(float))
    reg, _, _ = F.vdem_asof("v2x_regime", isos)
    pol, _, _ = F.vdem_asof("v2cacamps", isos)
    fx2 = F.vdem_change_asof("v2x_freexp_altinf", isos, years_back=2)
    # neighbours at war
    act = F.acd_active_by_year(); nb = F.borders()
    nshare = pd.DataFrame(np.nan, index=isos, columns=M); nbr4 = nshare.copy()
    for t in M:
        ly = F._lag_year(t, 6)
        for iso in isos:
            ns = [x for x in nb.get(iso, []) if x]
            if not ns:
                continue
            k = sum(1 for x in ns if not np.isnan(act.get((x, ly), np.nan)))
            nshare.at[iso, t] = k / len(ns); nbr4.at[iso, t] = float(k >= 4)
    # assemble long panel
    def L(df, n):
        return F.to_long(df, n)
    P = pd.concat([L(S1, "S1"), L(S3, "S3"), L(S4, "S4"), L(C2, "C2"), L(T1, "T1"), L(nshare, "nshare"),
                   L(nbr4, "nbr4"), L(reg, "regime"), L(pol, "polar"), L(fx2, "freexp_ch2"), L(pop, "pop"),
                   L(d3, "d3"), L(d12, "d12")], axis=1)
    P.index.names = ["iso3", "ym"]
    P = P.reset_index()
    P = P[[(i, t.year) in cy for i, t in zip(P.iso3, P.ym)]]
    hind = (h_ged.reindex(P.ym).values | h_gdp.reindex(P.ym).values | h_v.reindex(P.ym).values)
    P["hindsight_flag"] = hind.astype(int)
    # regime class
    hp = (P.polar >= POLAR_CUT).astype(float); hp[P.polar.isna()] = np.nan
    for k in range(4):
        P[f"reg{k}"] = (P.regime == k).astype(float); P.loc[P.regime.isna(), f"reg{k}"] = np.nan
    P["reg_x_polar"] = P[["reg0", "reg1", "reg2", "reg3"]].mul(hp, axis=0).values.tolist()
    for k in range(4):
        P[f"reg{k}_pol"] = [r[k] for r in P.reg_x_polar]
    P = P.drop(columns="reg_x_polar")
    P["half_open_polar"] = (((P.regime == 1) | (P.regime == 2)) & (hp == 1)).astype(float)
    P.loc[P.regime.isna() | hp.isna(), "half_open_polar"] = np.nan
    P["elec_autoc"] = P.reg1
    # published PITF score (no intercept)
    lp = (PITF_COEF["partial_autocracy"] * P.reg1 + PITF_COEF["partial_dem_faction"] * (P.reg2 * hp)
          + PITF_COEF["partial_dem_nofaction"] * (P.reg2 * (1 - hp)) + PITF_COEF["full_democracy"] * P.reg3
          + PITF_COEF["imr"] * P.T1 + PITF_COEF["nbr4"] * P.nbr4)
    P["pitf_lp"] = lp
    # censorship flag (G11)
    cens = (P.freexp_ch2 <= -0.10) & (P.d3 == 0) & (P.d12 > 0)
    P["censored"] = cens.astype(float)
    P.loc[P.censored == 1, ["S1"]] = np.nan
    P.to_parquet(os.path.join(FEAT, "grok_panel.parquet"), index=False)
    return P


# ------------------------------------------------------------------ outcomes (open tables only unless Job 6 passes sealed)
def load_outcomes(paths_long, paths_targets):
    L = pd.concat([pd.read_csv(p) for p in paths_long], ignore_index=True)
    T = pd.concat([pd.read_csv(p) for p in paths_targets], ignore_index=True)
    L["ym"] = pd.PeriodIndex(L.ym, freq="M"); T["ym"] = pd.PeriodIndex(T.ym, freq="M")
    return L, T

def history_features(L, T):
    """'Already in crisis' per lens, known at end of month t (G8)."""
    inc = L.groupby(["iso3", "ym"]).in_crisis.max().unstack("ym")
    dom = T.set_index(["iso3", "ym"]).domestic_event.unstack("ym")
    out = {}
    for nm, M in [("GLOBAL", inc), ("DOMESTIC", dom)]:
        M = M.reindex(columns=pd.period_range(M.columns.min(), M.columns.max(), freq="M"))
        r = M.T.rolling(12, min_periods=1).max().T
        out[nm] = F.to_long(r, "in_crisis_" + nm)
    return pd.concat(out.values(), axis=1).reset_index().rename(columns={"level_1": "ym"})

def targets(T, h):
    """Y(t,h) per lens: 1 if a scored event in t+1..t+h; 0 if all months 0; NaN otherwise (grey zone/missing)."""
    out = {}
    for lens, col in [("GLOBAL", "global_event"), ("DOMESTIC", "domestic_event")]:
        M = T.set_index(["iso3", "ym"])[col].unstack("ym")
        M = M.reindex(columns=pd.period_range(M.columns.min(), M.columns.max(), freq="M"))
        A = M.T
        ev = (A == 1).astype(float)[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1)
        zero = (A == 0).astype(float)[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1)
        y = pd.DataFrame(np.nan, index=A.index, columns=A.columns)
        y[zero >= h] = 0.0
        y[ev > 0] = 1.0
        # windows running past the last month in the table stay unknown
        y.iloc[len(y) - h:, :] = np.where(ev.iloc[len(y) - h:, :] > 0, 1.0, np.nan)
        out[lens] = F.to_long(y.T, f"Y_{lens}_{h}")
    return pd.concat(out.values(), axis=1).reset_index().rename(columns={"level_1": "ym"})


# ------------------------------------------------------------------ lens transforms
CONT = ["S1", "S3", "S4", "C2", "T1", "nshare"]
IND = ["reg1", "reg2", "reg3", "reg1_pol", "reg2_pol", "half_open_polar", "elec_autoc"]

def lens_transform(P, sd_train):
    P = P.sort_values(["iso3", "ym"]).reset_index(drop=True)
    G = P[["iso3", "ym"]].copy(); D = P[["iso3", "ym"]].copy()
    w = P["pop"]
    for c in CONT + IND:
        x = P[c]
        # GLOBAL: vs the population-weighted world this month
        df = pd.DataFrame({"ym": P.ym, "x": x, "w": w.where(x.notna())})
        df["wx"] = df.x * df.w; df["wxx"] = df.x ** 2 * df.w
        s = df.groupby("ym")[["w", "wx", "wxx"]].transform("sum")
        mu = s.wx / s.w; var = (s.wxx / s.w - mu ** 2).clip(lower=1e-12)
        G[c] = (x - mu) / np.sqrt(var)
        # DOMESTIC: vs own previous 120 months (needs 60)
        g = P.groupby("iso3")[c]
        med = g.transform(lambda s: s.shift(1).rolling(120, min_periods=60).median())
        mad = g.transform(lambda s: (s.shift(1) - s.shift(1).rolling(120, min_periods=60).median()).abs().rolling(120, min_periods=60).median())
        den = np.maximum(1.4826 * mad, 0.1 * sd_train.get(c, 1.0))
        D[c] = (x - med) / den
    return G, D

def design_matrix(Z, P, Hh, lens, scaler=None):
    X = pd.DataFrame(index=Z.index)
    for c in CONT + IND:
        X[c] = Z[c]
    if scaler is None:
        scaler = {c: (float(np.nanmean(X[c])), float(np.nanstd(X[c]) or 1.0)) for c in X.columns}
    for c in list(X.columns):
        m, s = scaler[c]; X[c] = (X[c] - m) / (s if s > 0 else 1.0)
    miss = X.isna().astype(float).add_prefix("miss_")
    # interaction L = S x C (means of present members)
    S = X[["S1", "S3", "S4", "C2"]].mean(axis=1, skipna=True)
    C = pd.concat([X["half_open_polar"], X["elec_autoc"], X["T1"], X["nshare"]], axis=1).mean(axis=1, skipna=True)
    X["Lint"] = S * C
    X = X.fillna(0.0)
    X = pd.concat([X, miss[[f"miss_{c}" for c in ["S1", "S3", "S4", "C2", "T1", "nshare", "reg1"]]]], axis=1)
    X["in_crisis"] = Hh[f"in_crisis_{lens}"].fillna(0).values
    X["miss_hist"] = Hh[f"in_crisis_{lens}"].isna().astype(float).values
    X["pitf"] = P["pitf_lp"].fillna(0).values; X["miss_pitf"] = P["pitf_lp"].isna().astype(float).values
    return X, scaler


# ------------------------------------------------------------------ fit
def brier(y, p):
    return float(np.mean((p - y) ** 2))

def fit(open_long, open_targets):
    P = pd.read_parquet(os.path.join(FEAT, "grok_panel.parquet")) if os.path.exists(os.path.join(FEAT, "grok_panel.parquet")) else build_features()
    P["ym"] = pd.PeriodIndex(P.ym, freq="M")
    P = P.sort_values(["iso3", "ym"]).reset_index(drop=True)
    L, T = load_outcomes([open_long], [open_targets])
    Hh = history_features(L, T)
    P = P.merge(Hh, on=["iso3", "ym"], how="left")
    tr = P.ym <= TRAIN_END
    sd_train = {c: float(np.nanstd(P.loc[tr, c])) for c in CONT + IND}
    G, D = lens_transform(P, sd_train)
    Ys = {h: targets(T, h) for h in H}
    params = {"choices": GROK_CHOICES, "train_end": str(TRAIN_END), "sd_train": sd_train, "models": {}}
    Zs = {"GLOBAL": G, "DOMESTIC": D}
    # penalty per horizon by walk-forward (both lenses averaged)
    for h in H:
        Y = P[["iso3", "ym"]].merge(Ys[h], on=["iso3", "ym"], how="left")
        scores = {c: [] for c in C_GRID}
        for lens in LENSES:
            yv = Y[f"Y_{lens}_{h}"].values
            for y in CV_YEARS:
                end = pd.Period(year=y, month=12, freq="M")
                q = P.ym.dt.month.isin([1, 4, 7, 10]).values
                trm = (P.ym.values <= (end - h)) & ~np.isnan(yv) & q
                vam = (P.ym.values > end) & (P.ym.values <= end + 36) & ~np.isnan(yv) & q
                if yv[trm].sum() < 5 or vam.sum() == 0:
                    continue
                Xtr, sc = design_matrix(Zs[lens][trm], P[trm], P[trm], lens)
                Xva, _ = design_matrix(Zs[lens][vam], P[vam], P[vam], lens, sc)
                for c in C_GRID:
                    m = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=c, solver="saga", max_iter=300, tol=1e-3)
                    m.fit(Xtr.values, yv[trm])
                    scores[c].append(brier(yv[vam], m.predict_proba(Xva.values)[:, 1]))
        best = min(C_GRID, key=lambda c: np.mean(scores[c]) if scores[c] else 9)
        print(f"h={h} penalty C={best} cv Brier=" + ", ".join(f"{c}:{np.mean(v):.4f}" for c, v in scores.items() if v), flush=True)
        for lens in LENSES:
            yv = Y[f"Y_{lens}_{h}"].values
            trm = (P.ym.values <= TRAIN_END) & ~np.isnan(yv)
            Xtr, sc = design_matrix(Zs[lens][trm], P[trm], P[trm], lens)
            m = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=best, solver="saga", max_iter=1000, tol=1e-4)
            m.fit(Xtr.values, yv[trm])
            praw = np.clip(m.predict_proba(Xtr.values)[:, 1], 1e-6, 1 - 1e-6)
            pm = trm & (P.ym.values >= PLATT_FROM)
            lp = np.log(praw / (1 - praw))[pm[trm]]
            pl = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), yv[pm])
            params["models"][f"{lens}_{h}"] = {
                "C": best, "features": list(Xtr.columns), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
                "scaler": sc, "platt_a": float(pl.intercept_[0]), "platt_b": float(pl.coef_[0][0]),
                "n_train": int(trm.sum()), "events_train": int(yv[trm].sum())}
    # Atrocity Alarm (G12)
    params["atrocity"] = fit_alarm(P, L, Zs)
    js = json.dumps(params, indent=1, default=str)
    open(os.path.join(MODELS, "grok_params.json"), "w").write(js)
    print("locked params sha256", hashlib.sha256(js.encode()).hexdigest())
    return params


def c2_country_year(L):
    c2 = L[L.ctype == "C2"].copy(); c2["year"] = c2.ym.dt.year
    g = c2.groupby(["iso3", "year"]).agg(on=("onset", "max"), obs=("observed", "max"), near=("near", "max"))
    y = g.on.where(g.obs == 1)
    y[(y == 1) & (g.near == 1)] = np.nan
    return y

ALARM_FEATS = ["S1", "C2", "polar", "half_open_polar"]

def alarm_matrix(P, L, scaler=None):
    D = P[P.ym.dt.month == 12].copy()
    D["year_next"] = D.ym.dt.year + 1
    c2 = L[L.ctype == "C2"].copy(); c2["year"] = c2.ym.dt.year
    prior = c2.groupby(["iso3", "year"]).onset.max().groupby(level=0).cummax()
    D["prior_onset"] = [prior.get((i, t.year), np.nan) for i, t in zip(D.iso3, D.ym)]
    osv = P.d12  # all violence as published; one-sided series enters via C2 (state violence) and S1 (Guess)
    X = D[ALARM_FEATS + ["prior_onset"]].copy()
    X["log_d12"] = np.log1p(D.d12)
    if scaler is None:
        scaler = {c: (float(np.nanmean(X[c])), float(np.nanstd(X[c]) or 1.0)) for c in X.columns}
    for c in list(X.columns):
        m, s = scaler[c]; X[c] = (X[c] - m) / (s or 1.0)
    miss = X.isna().astype(float).add_prefix("miss_")
    X = pd.concat([X.fillna(0), miss], axis=1)
    X["censored"] = D.censored.fillna(0).values
    return D, X, scaler

def fit_alarm(P, L, Zs):
    y = c2_country_year(L)
    D, X, sc = alarm_matrix(P, L)
    yy = np.array([y.get((i, yn), np.nan) for i, yn in zip(D.iso3, D.year_next)])
    tr = (D.ym <= TRAIN_END).values & ~np.isnan(yy)
    m = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1, solver="saga", max_iter=2000).fit(X.values[tr], yy[tr])
    p = m.predict_proba(X.values[tr])[:, 1]
    neg = np.sort(p[yy[tr] == 0])
    thr = float(neg[int(np.ceil(0.9 * len(neg))) - 1]) if len(neg) else 1.0   # FPR <= 0.10 boundary
    fires_cases = float(np.mean(p[yy[tr] == 1] > thr)) if (yy[tr] == 1).any() else 0.0
    return {"features": list(X.columns), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "scaler": sc,
            "threshold": thr, "train_recall": fires_cases, "train_fpr": float(np.mean(p[yy[tr] == 0] > thr)),
            "n_train": int(tr.sum()), "cases_train": int(yy[tr].sum())}


# ------------------------------------------------------------------ predict
def predict(params, long_paths, target_paths, last_origin=None):
    P = pd.read_parquet(os.path.join(FEAT, "grok_panel.parquet"))
    P["ym"] = pd.PeriodIndex(P.ym, freq="M")
    P = P.sort_values(["iso3", "ym"]).reset_index(drop=True)
    L, T = load_outcomes(long_paths, target_paths)
    Hh = history_features(L, T)
    P = P.merge(Hh, on=["iso3", "ym"], how="left")
    if last_origin is not None:
        keep = P.ym <= pd.Period(last_origin, "M")
    else:
        keep = pd.Series(True, index=P.index)
    G, D = lens_transform(P, params["sd_train"])
    Zs = {"GLOBAL": G, "DOMESTIC": D}
    out = P.loc[keep, ["iso3", "ym", "hindsight_flag"]].copy()
    for lens in LENSES:
        for h in H:
            md = params["models"][f"{lens}_{h}"]
            X, _ = design_matrix(Zs[lens][keep], P[keep], P[keep], lens, md["scaler"])
            X = X[md["features"]]
            z = md["intercept"] + X.values @ np.array(md["coef"])
            praw = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
            p = 1 / (1 + np.exp(-(md["platt_a"] + md["platt_b"] * np.log(praw / (1 - praw)))))
            out[f"p_{lens}_{h}"] = np.clip(p, 0.01, 0.99)
    # confidence (3.7, simplified): share of the seven signal families present x horizon factor
    present = P.loc[keep, ["S1", "S3", "S4", "C2", "T1", "nshare", "regime"]].notna().mean(axis=1)
    out["q_data"] = present.values
    # atrocity alarm
    a = params["atrocity"]
    Dd, Xa, _ = alarm_matrix(P[keep], L, a["scaler"])
    z = a["intercept"] + Xa[a["features"]].values @ np.array(a["coef"])
    pa = 1 / (1 + np.exp(-z))
    alarm = pd.DataFrame({"iso3": Dd.iso3.values, "year": Dd.year_next.values, "p_atrocity": pa, "alarm": (pa > a["threshold"]).astype(int)})
    return out, alarm


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    O = os.path.join(ROOT, "data", "outcomes")
    if cmd == "features":
        build_features()
    elif cmd == "fit":
        if not os.path.exists(os.path.join(FEAT, "grok_panel.parquet")):
            build_features()
        fit(os.path.join(O, "outcomes_long_open.csv.gz"), os.path.join(O, "targets_open.csv.gz"))
    elif cmd == "predict_open":
        params = json.load(open(os.path.join(MODELS, "grok_params.json")))
        out, alarm = predict(params, [os.path.join(O, "outcomes_long_open.csv.gz")], [os.path.join(O, "targets_open.csv.gz")], "2014-12")
        out.to_parquet(os.path.join(MODELS, "grok_forecasts_open.parquet"), index=False)
        alarm.to_parquet(os.path.join(MODELS, "grok_alarm_open.parquet"), index=False)
        print(out.describe().T.to_string())
