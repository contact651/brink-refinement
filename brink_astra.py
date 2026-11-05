#!/usr/bin/env python3
"""
Brink Model - Round 1 build - JOB 4: GPT-6 Astra's design ("Pressure, damage and recovery"), as a working program.

Design: briefs/brink-design-astra.md (verbatim entry, 18 Oct 2026). Exam: Exam Pack v2.2 (SHA-256 bd3d64f1...ebaec1).
Smallest reasonable choices where the design is unclear or needs data that does not exist are in ASTRA_CHOICES
(all Guess). Nothing was tuned on 2015-2025.

What it does (plain words)
  - Six signal families, measured as public at each month: violence (rate, change, neighbours), food prices,
    weather, restriction, coping capacity, recovery (a proxy current-conditions score and whether it fails to
    come back down).
  - A small monthly survival model: the chance a new crisis (or escalation) starts in month m after the
    forecast, given no crisis yet, with a month-specific baseline, a shrunk per-place intercept, and three
    interactions (food x low reserves, violence growth x restriction growth, weather x infant mortality).
    The chance of a crisis within h months is 1 - product of (1 - monthly chance), so it never falls with h.
  - One expert per family + the full interaction model + a pooled baseline; their curves are mixed with
    non-negative weights summing to 1, fitted on out-of-sample forecasts and shrunk toward the baseline.
  - Fitted on everything known by Dec 2014, then locked.
Usage: python3 brink_astra.py features | fit | predict_open
"""
import json, os, sys, hashlib, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import brink_features as F
from brink_grok import load_outcomes, targets, H, LENSES

FEAT = os.path.join(ROOT, "data", "features"); MODELS = os.path.join(ROOT, "data", "models")
FIT_END = pd.Period("2014-12", "M")
DEV_SPLIT = pd.Period("2004-12", "M")         # experts trained to here produce out-of-sample curves for 2005-2014
PEN_GRID = [0.01, 0.1, 1, 10, 100]           # L2 penalty lambda (design's grid); C = 1/lambda
HMAX = 24
FAM = {
    "violence": ["v_rate3", "v_diff3", "v_adj"],
    "food": ["food_yoy", "food_ch3"],
    "weather": ["pre_anom3", "tmx_anom1"],
    "restriction": ["r_civ", "r_fx", "r_civ_ch", "r_fx_ch"],
    "capacity": ["log_imr", "log_res"],
    "recovery": ["S", "dS3", "n_above"],
}
ALLF = sum(FAM.values(), [])
INTER = [("food_yoy", "low_res"), ("v_diff3", "r_civ_ch"), ("pre_anom3", "log_imr")]

ASTRA_CHOICES = [
    "A1 Unit = country-month (exam unit); the design's province level waits for province outcomes.",
    "A2 Outcome = the exam's C1-C7 event ledger (Exam Pack s.2). The design's own ledger (person-days of service and rights deprivation across 8 UDHR domains) does not exist in open data - the design itself says it is 'not exam-ready'.",
    "A3 Both lenses: the design publishes one absolute probability with GLOBAL/DOMESTIC percentile readings. The exam scores a GLOBAL and a DOMESTIC event, so the same model structure is fitted once per lens as a 'task-specific head' (design section 8 allows task heads with the same features and procedure).",
    "A4 Strict vintages: the design prefers excluding records without a real vintage. Before 2015 almost no vintages exist, so excluding them would leave nothing to fit on; lag rules are used instead and every such row carries hindsight_flag = 1 (reported separately in Job 6).",
    "A5 Weather: ERA5 needs an account, so CRU TS 4.09 country averages (Open Government Licence) are used. CRU publishes yearly, so with the lag rule (year y usable April y+2) the newest usable month is 15-27 months old; the signal is therefore slow.",
    "A6 Recovery family uses a proxy current-conditions score S = (D + R)/2: D = min(1, life-years lost / (1% of population)), with life-years lost = average monthly UCDP deaths (last 3 usable months) x 40 years (one reference value for unknown age, Guess); R = 1 - V-Dem civil liberties. Features: S, its 3-month change, and how many of the last 12 months were above the place's median of the 5 years before.",
    "A7 Features at missing values: training median + missing flag (design 3.3). Observation age is not added as a separate feature.",
    "A8 Fitting rows use forecast months in January and July only (every 6 months) to keep the monthly survival expansion small; prediction is for every month. Months in the window that are grey-zone or missing end the record there (censored).",
    "A9 Penalty chosen once, for the full interaction model, on two time-ordered folds (train to 1999 -> test 2000-04; train to 2004 -> test 2005-09) by monthly survival log-loss, larger penalty on ties; the experts reuse it.",
    "A10 Mixture: experts trained on windows ending by Dec 2004 give out-of-sample curves for 2005-2014; weights minimise mean log-loss over the six horizons plus mu x distance from 'all weight on the pooled baseline'; mu chosen from the same grid by fitting on 2005-09 and testing on 2010-14; final weights refitted on 2005-14; final experts retrained on everything to Dec 2014. The design's separate untouched promotion test is the live round (Round 2).",
    "A11 Uncertainty bootstrap (3.6) is not run for the exam (display feature, not scored). Evidence confidence c = q x v / (1 + h/12) is reported with v = 1.",
    "A12 Atrocity Alarm: survival head on exam C2 onsets/escalations, 12-month chance issued each December for the next year; threshold = lowest cut whose false-alarm rate (Wilson 95% upper bound) on 2005-2014 out-of-sample country-year non-cases is <= 0.10; if it catches no case it is reported as 'not useful'.",
    "A13 Benchmarks as inputs: the design reports results with and without each benchmark as an input. Round 1 builds the version WITHOUT benchmark inputs (none has a record before 2015 on the exam target except the PITF-style rebuild, built in Job 5).",
]


# ------------------------------------------------------------------ features
def build_features():
    isos, cy = F.universe()
    M = F.MONTHS
    pop, _, h_p = F.wdi_asof("population", isos)
    d1, d3, d12, lagm, h_g = F.deaths_asof(isos)
    U = [F.ucdp_iso(i) for i in isos]
    d3 = d3.reindex(U); d3.index = isos; d12 = d12.reindex(U); d12.index = isos
    rate3 = np.log1p(d3 / pop * 1e5)
    prev3 = rate3.shift(3, axis=1)
    v_diff = rate3 - prev3
    nb = F.borders()
    v_adj = pd.DataFrame(np.nan, index=isos, columns=M)
    for iso in isos:
        ns = [x for x in nb.get(iso, []) if x in d3.index]
        if ns:
            v_adj.loc[iso] = np.log1p(d3.loc[ns].sum(min_count=1) / pop.loc[ns].sum(min_count=1) * 1e5).values
    # food: month m usable at the end of m+1
    w = F.rd("wfp_food_yoy_cm")
    w["ym"] = [pd.Period(year=int(y), month=int(m), freq="M") for y, m in zip(w.year, w.month)]
    fm = w.pivot_table(index="iso3", columns="ym", values="food_yoy_median").reindex(index=isos, columns=M)
    food = fm.shift(1, axis=1); food_ch3 = food - fm.shift(4, axis=1)
    # weather: seasonal z-anomalies from preceding years only (>= 10 years), newest usable month by CRU rule
    c = F.rd("cru_country_monthly")
    c = c[c.iso3.isin(isos)]
    c["ym"] = [pd.Period(year=int(y), month=int(m), freq="M") for y, m in zip(c.year, c["month"])]
    def anom(var):
        X = c[c["var"] == var].pivot_table(index="iso3", columns="ym", values="value").reindex(index=isos, columns=M)
        Z = pd.DataFrame(np.nan, index=isos, columns=M)
        for mo in range(1, 13):
            cols = [m for m in M if m.month == mo]
            sub = X[cols]
            mu = sub.T.expanding(min_periods=10).mean().shift(1).T
            sd = sub.T.expanding(min_periods=10).std().shift(1).T
            Z[cols] = (sub - mu) / sd
        return Z
    zp, zt = anom("pre"), anom("tmx")
    pre3 = zp.T.rolling(3, min_periods=3).mean().T
    pre_anom3 = pd.DataFrame(np.nan, index=isos, columns=M); tmx1 = pre_anom3.copy()
    for t in M:
        y = t.year - 2 if t.month >= 4 else t.year - 3
        A = pd.Period(year=y, month=12, freq="M")
        if t >= pd.Period("2025-03", "M"):
            A = max(A, pd.Period("2024-12", "M"))
        if A in pre3.columns:
            pre_anom3[t] = pre3[A]; tmx1[t] = zt[A]
    civ, _, h_v = F.vdem_asof("v2x_civlib", isos)
    fx, _, _ = F.vdem_asof("v2x_freexp_altinf", isos)
    civ_ch = F.vdem_change_asof("v2x_civlib", isos, 1); fx_ch = F.vdem_change_asof("v2x_freexp_altinf", isos, 1)
    imr, _, h_w = F.wdi_asof("infant_mortality", isos)
    res, _, _ = F.wdi_asof("reserves_months_imports", isos)
    # recovery proxy
    D = (d3 / 3 * 40 / (0.01 * pop)).clip(upper=1)
    R = 1 - civ
    S = (D + R) / 2
    dS3 = S - S.shift(3, axis=1)
    medp = S.T.rolling(60, min_periods=36).median().shift(12).T
    above = (S > medp).astype(float).where(S.notna() & medp.notna())
    n_above = above.T.rolling(12, min_periods=6).sum().T
    parts = {"v_rate3": rate3, "v_diff3": v_diff, "v_adj": v_adj, "food_yoy": food, "food_ch3": food_ch3,
             "pre_anom3": pre_anom3, "tmx_anom1": tmx1, "r_civ": 1 - civ, "r_fx": 1 - fx, "r_civ_ch": -civ_ch,
             "r_fx_ch": -fx_ch, "log_imr": np.log(imr), "log_res": np.log1p(res), "S": S, "dS3": dS3,
             "n_above": n_above, "pop": pop}
    P = pd.concat([F.to_long(v, k) for k, v in parts.items()], axis=1)
    P.index.names = ["iso3", "ym"]; P = P.reset_index()
    P = P[[(i, t.year) in cy for i, t in zip(P.iso3, P.ym)]]
    P["hindsight_flag"] = (h_g.reindex(P.ym).values | h_w.reindex(P.ym).values | h_v.reindex(P.ym).values).astype(int)
    P.to_parquet(os.path.join(FEAT, "astra_panel.parquet"), index=False)
    return P


# ------------------------------------------------------------------ survival data
def event_matrix(T, lens):
    col = {"GLOBAL": "global_event", "DOMESTIC": "domestic_event"}[lens]
    E = T.set_index(["iso3", "ym"])[col].unstack("ym")
    return E.reindex(columns=pd.period_range(pd.Period("1946-01", "M"), E.columns.max(), freq="M"))

def standardise(P, stats=None):
    X = P[ALLF].copy()
    if stats is None:
        stats = {"median": {c: float(X[c].median()) for c in ALLF}}
    miss = X.isna().astype(float).add_prefix("m_")
    X = X.fillna(pd.Series(stats["median"]))
    if "sd" not in stats:
        stats["sd"] = {c: (float(X[c].mean()), float(X[c].std() or 1.0)) for c in ALLF}
    for c in ALLF:
        m, s = stats["sd"][c]; X[c] = (X[c] - m) / s
    X["low_res"] = -X["log_res"]
    for a, b in INTER:
        X[f"{a}*{b}"] = X[a] * X[b]
    X = X.drop(columns=["low_res"])
    return pd.concat([X, miss], axis=1), stats

def family_cols(fam, cols):
    if fam == "baseline":
        return []
    if fam == "full":
        return list(cols)
    keep = FAM[fam]
    return [c for c in cols if c in keep or c in [f"m_{k}" for k in keep]]

def person_period(origins_idx, E, iso_pos, month_pos, P):
    """Rows (origin row, month-after m, y) while at risk; censor at the first unknown month."""
    Ev = E.values
    rows, ms, ys = [], [], []
    for r in origins_idx:
        i = iso_pos[P.iso3.iat[r]]; k = month_pos[P.ym.iat[r]]
        for m in range(1, HMAX + 1):
            if k + m >= Ev.shape[1]:
                break
            v = Ev[i, k + m]
            if np.isnan(v):
                break
            rows.append(r); ms.append(m); ys.append(v)
            if v == 1:
                break
    return np.array(rows), np.array(ms), np.array(ys)

def sparse_design(Xd, rows, ms, places, iso_codes, use_place=True):
    n = len(rows)
    blocks = [sp.csr_matrix(Xd[rows])] if Xd.shape[1] else []
    md = sp.csr_matrix((np.ones(n), (np.arange(n), ms - 1)), shape=(n, HMAX))
    blocks.append(md[:, 1:])           # month 1 is the reference (intercept)
    if use_place:
        pi = np.array([places.get(c, -1) for c in iso_codes[rows]])
        ok = pi >= 0
        pdm = sp.csr_matrix((np.ones(ok.sum()), (np.arange(n)[ok], pi[ok])), shape=(n, len(places)))
        blocks.append(pdm)
    return sp.hstack(blocks).tocsr()

def fit_hazard(Xd, rows, ms, ys, places, iso_codes, lam, use_place=True):
    A = sparse_design(Xd, rows, ms, places, iso_codes, use_place)
    m = LogisticRegression(penalty="l2", C=1.0 / lam, solver="lbfgs", max_iter=400, tol=1e-5)
    m.fit(A, ys)
    return m

def curves(model, Xd, places, iso_codes, use_place=True):
    """Cumulative probability for h=1..24 for every row of Xd."""
    n = Xd.shape[0]
    coef = model.coef_[0]; b0 = model.intercept_[0]
    k = Xd.shape[1]
    beta = coef[:k]; alpha = np.r_[0.0, coef[k:k + HMAX - 1]]
    u = coef[k + HMAX - 1:] if use_place else None
    base = b0 + (Xd @ beta if k else np.zeros(n))
    if use_place:
        pi = np.array([places.get(c, -1) for c in iso_codes])
        base = base + np.where(pi >= 0, u[np.clip(pi, 0, None)], 0.0)
    lam = 1 / (1 + np.exp(-(base[:, None] + alpha[None, :])))
    return 1 - np.cumprod(1 - lam, axis=1)

def nll(model, Xd, rows, ms, ys, places, iso_codes, use_place=True):
    A = sparse_design(Xd, rows, ms, places, iso_codes, use_place)
    p = np.clip(model.predict_proba(A)[:, 1], 1e-9, 1 - 1e-9)
    return float(-np.mean(ys * np.log(p) + (1 - ys) * np.log(1 - p)))


# ------------------------------------------------------------------ fit
def prep(long_paths, target_paths):
    P = pd.read_parquet(os.path.join(FEAT, "astra_panel.parquet"))
    P["ym"] = pd.PeriodIndex(P.ym, freq="M")
    P = P.sort_values(["iso3", "ym"]).reset_index(drop=True)
    L, T = load_outcomes(long_paths, target_paths)
    return P, L, T

def mixture_weights(curves_by_expert, Ys, mu, base_idx):
    K = len(curves_by_expert)
    def loss(w):
        w = np.abs(w); w = w / w.sum()
        tot = 0.0
        for hi, h in enumerate(H):
            y = Ys[h]; ok = ~np.isnan(y)
            p = np.clip(sum(w[k] * curves_by_expert[k][ok, h - 1] for k in range(K)), 1e-6, 1 - 1e-6)
            tot += -np.mean(y[ok] * np.log(p) + (1 - y[ok]) * np.log(1 - p))
        e = np.zeros(K); e[base_idx] = 1
        return tot / len(H) + mu * np.sum((w - e) ** 2)
    w0 = np.ones(K) / K
    r = minimize(loss, w0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-7})
    w = np.abs(r.x); return w / w.sum(), float(r.fun)

def fit_lens(P, T, lens, params):
    E = event_matrix(T, lens)
    iso_pos = {c: k for k, c in enumerate(E.index)}; month_pos = {m: k for k, m in enumerate(E.columns)}
    places = {c: k for k, c in enumerate(sorted(P.iso3.unique()))}
    iso_codes = P.iso3.values
    sel = P.ym.dt.month.isin([1, 7]).values & (P.ym.dt.year >= 1960).values
    def origins(end_window):
        return np.where(sel & (P.ym.values <= end_window - 1))[0]
    def rows_until(idx, end):
        r, m, y = person_period(idx, E, iso_pos, month_pos, P)
        keep = np.array([P.ym.iat[a] + int(b) <= end for a, b in zip(r, m)]) if len(r) else np.array([], bool)
        return r[keep], m[keep], y[keep]
    # standardise on fitting rows (everything known by Dec 2014)
    Xall, stats = standardise(P[P.ym <= FIT_END])
    X, _ = standardise(P, stats); cols = list(X.columns); Xv = X.values
    # penalty for the full model (A9)
    folds = [(pd.Period("1999-12", "M"), pd.Period("2004-12", "M")), (pd.Period("2004-12", "M"), pd.Period("2009-12", "M"))]
    pen_scores = {}
    for lam in PEN_GRID:
        s = []
        for tr_end, va_end in folds:
            r, m, y = rows_until(origins(tr_end), tr_end)
            vo = np.where(sel & (P.ym.values > tr_end) & (P.ym.values <= va_end - 1))[0]
            rv, mv, yv = rows_until(vo, va_end)
            mod = fit_hazard(Xv[:, :], r, m, y, places, iso_codes, lam)
            s.append(nll(mod, Xv, rv, mv, yv, places, iso_codes))
        pen_scores[lam] = float(np.mean(s))
    best = min(pen_scores.values())
    lam = max(l for l, v in pen_scores.items() if v <= best * 1.001)
    print(lens, "penalty", lam, pen_scores, flush=True)
    experts = ["baseline"] + list(FAM) + ["full"]
    # out-of-sample curves 2005-2014 from experts trained to Dec 2004
    r, m, y = rows_until(origins(DEV_SPLIT), DEV_SPLIT)
    oo = np.where((P.ym.values > DEV_SPLIT) & (P.ym.values <= FIT_END - 1) & P.ym.dt.month.isin([1, 4, 7, 10]).values)[0]
    Ys = {}
    for h in H:
        Yh = P.iloc[oo][["iso3", "ym"]].merge(targets(T, h), on=["iso3", "ym"], how="left")[f"Y_{lens}_{h}"].values.copy()
        late = (P.ym.values[oo] + h) > FIT_END
        Yh[late] = np.nan
        Ys[h] = Yh
    cv = []
    for ex in experts:
        ci = [cols.index(c) for c in family_cols(ex, cols)]
        mod = fit_hazard(Xv[:, ci], r, m, y, places, iso_codes, lam, use_place=(ex != "baseline"))
        cv.append(curves(mod, Xv[oo][:, ci], places, iso_codes[oo], use_place=(ex != "baseline")))
    first = P.ym.values[oo] <= pd.Period("2009-12", "M")
    mus = {}
    for mu in PEN_GRID:
        w, _ = mixture_weights([c[first] for c in cv], {h: Ys[h][first] for h in H}, mu, 0)
        sc = 0.0
        for h in H:
            y_ = Ys[h][~first]; ok = ~np.isnan(y_)
            p = np.clip(sum(w[k] * cv[k][~first][ok, h - 1] for k in range(len(cv))), 1e-6, 1 - 1e-6)
            sc += -np.mean(y_[ok] * np.log(p) + (1 - y_[ok]) * np.log(1 - p))
        mus[mu] = sc / len(H)
    bm = min(mus.values()); mu = max(k for k, v in mus.items() if v <= bm * 1.001)
    w, _ = mixture_weights(cv, Ys, mu, 0)
    print(lens, "mixture mu", mu, dict(zip(experts, np.round(w, 3))), flush=True)
    # final experts on everything known by Dec 2014
    r, m, y = rows_until(origins(FIT_END), FIT_END)
    out = {"penalty": lam, "pen_scores": pen_scores, "mu": mu, "mu_scores": mus, "weights": dict(zip(experts, w.tolist())),
           "experts": {}, "places": places, "stats": stats, "cols": cols, "n_rows": int(len(r)), "events": int(y.sum())}
    for ex in experts:
        ci = [cols.index(c) for c in family_cols(ex, cols)]
        mod = fit_hazard(Xv[:, ci], r, m, y, places, iso_codes, lam, use_place=(ex != "baseline"))
        out["experts"][ex] = {"cols": [cols[i] for i in ci], "coef": mod.coef_[0].tolist(), "intercept": float(mod.intercept_[0]),
                              "use_place": ex != "baseline"}
    params["models"][lens] = out


class _M:
    def __init__(self, d):
        self.coef_ = np.array([d["coef"]]); self.intercept_ = np.array([d["intercept"]])

def predict_curves(params, P, lens):
    md = params["models"][lens]
    X, _ = standardise(P, md["stats"]); X = X[md["cols"]]
    tot = 0
    for ex, w in md["weights"].items():
        e = md["experts"][ex]
        c = curves(_M(e), X[e["cols"]].values, md["places"], P.iso3.values, e["use_place"])
        tot = tot + w * c
    return tot

def fit():
    O = os.path.join(ROOT, "data", "outcomes")
    P, L, T = prep([os.path.join(O, "outcomes_long_open.csv.gz")], [os.path.join(O, "targets_open.csv.gz")])
    params = {"choices": ASTRA_CHOICES, "fit_end": str(FIT_END), "models": {}}
    for lens in LENSES:
        fit_lens(P, T, lens, params)
    params["atrocity"] = fit_alarm(P, L, T, params)
    js = json.dumps(params, indent=1, default=str)
    open(os.path.join(MODELS, "astra_params.json"), "w").write(js)
    print("locked params sha256", hashlib.sha256(js.encode()).hexdigest())

def c2_events(L):
    c2 = L[L.ctype == "C2"]
    ev = ((c2.onset == 1) | (c2.escalation == 1)).astype(float).where(c2.observed == 1)
    ev[(c2.near == 1).values & (c2.escalation != 1).values] = np.nan
    E = pd.Series(ev.values, index=pd.MultiIndex.from_arrays([c2.iso3, c2.ym])).unstack()
    return E.reindex(columns=pd.period_range(pd.Period("1946-01", "M"), E.columns.max(), freq="M"))

def wilson_upper(k, n, z=1.96):
    if n == 0:
        return 1.0
    p = k / n
    return (p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)

def fit_alarm(P, L, T, params):
    E = c2_events(L)
    iso_pos = {c: k for k, c in enumerate(E.index)}; month_pos = {m: k for k, m in enumerate(E.columns)}
    places = {c: k for k, c in enumerate(sorted(P.iso3.unique()))}
    stats = params["models"]["GLOBAL"]["stats"]; cols = params["models"]["GLOBAL"]["cols"]; lam = params["models"]["GLOBAL"]["penalty"]
    X, _ = standardise(P, stats); Xv = X[cols].values
    sel = P.ym.dt.month.isin([1, 7]).values & (P.ym.dt.year >= 1989).values
    def fit_to(end):
        idx = np.where(sel & (P.ym.values <= end - 1))[0]
        r, m, y = person_period(idx, E, iso_pos, month_pos, P)
        keep = np.array([P.ym.iat[a] + int(b) <= end for a, b in zip(r, m)])
        return fit_hazard(Xv, r[keep], m[keep], y[keep], places, P.iso3.values, lam)
    dec = np.where((P.ym.dt.month == 12).values)[0]
    # out-of-sample 2005-2014 country-years
    mod = fit_to(DEV_SPLIT)
    oo = dec[(P.ym.values[dec] >= DEV_SPLIT) & (P.ym.values[dec] <= FIT_END - 12)]
    p12 = curves(mod, Xv[oo], places, P.iso3.values[oo])[:, 11]
    yy = []
    for r in oo:
        i = iso_pos.get(P.iso3.iat[r]); k = month_pos[P.ym.iat[r]]
        seg = E.values[i, k + 1:k + 13] if i is not None else np.array([np.nan])
        yy.append(1.0 if np.nansum(seg) > 0 else (0.0 if np.all(~np.isnan(seg)) else np.nan))
    yy = np.array(yy); ok = ~np.isnan(yy)
    thr = 1.0
    for c in np.sort(np.unique(p12[ok & (yy == 0)])):
        fa = int(np.sum((p12 > c) & ok & (yy == 0))); n = int(np.sum(ok & (yy == 0)))
        if wilson_upper(fa, n) <= 0.10:
            thr = float(c); break
    rec = float(np.mean(p12[ok & (yy == 1)] > thr)) if (ok & (yy == 1)).any() else 0.0
    final = fit_to(FIT_END)
    return {"coef": final.coef_[0].tolist(), "intercept": float(final.intercept_[0]), "cols": cols, "places": places,
            "stats": stats, "threshold": thr, "dev_recall": rec,
            "dev_fpr": float(np.mean(p12[ok & (yy == 0)] > thr)), "useful": rec > 0}

def predict(params, long_paths, target_paths, last_origin=None):
    P, L, T = prep(long_paths, target_paths)
    if last_origin:
        P = P[P.ym <= pd.Period(last_origin, "M")].reset_index(drop=True)
    out = P[["iso3", "ym", "hindsight_flag"]].copy()
    for lens in LENSES:
        C = predict_curves(params, P, lens)
        for h in H:
            out[f"p_{lens}_{h}"] = np.clip(C[:, h - 1], 1e-4, 1 - 1e-4)
    q = P[ALLF].notna().mean(axis=1).values
    out["evidence_q"] = q
    a = params["atrocity"]
    D = P[P.ym.dt.month == 12].reset_index(drop=True)
    X, _ = standardise(D, a["stats"])
    p12 = curves(_M(a), X[a["cols"]].values, a["places"], D.iso3.values)[:, 11]
    alarm = pd.DataFrame({"iso3": D.iso3, "year": D.ym.dt.year + 1, "p_atrocity": p12, "alarm": (p12 > a["threshold"]).astype(int)})
    return out, alarm


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    O = os.path.join(ROOT, "data", "outcomes")
    if cmd == "features":
        build_features()
    elif cmd == "fit":
        if not os.path.exists(os.path.join(FEAT, "astra_panel.parquet")):
            build_features()
        fit()
    elif cmd == "predict_open":
        params = json.load(open(os.path.join(MODELS, "astra_params.json")))
        out, alarm = predict(params, [os.path.join(O, "outcomes_long_open.csv.gz")], [os.path.join(O, "targets_open.csv.gz")], "2014-12")
        out.to_parquet(os.path.join(MODELS, "astra_forecasts_open.parquet"), index=False)
        alarm.to_parquet(os.path.join(MODELS, "astra_alarm_open.parquet"), index=False)
        print(out.describe().T.to_string())
