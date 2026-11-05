#!/usr/bin/env python3
"""
Brink Model - Round 1 build - JOB 3: Claude Opus 5.5's design ("Stress, Buffer, Trigger"), as a working program.

Design: briefs/brink-design-claude.md (verbatim entry, 18 Oct 2026). Exam: Exam Pack v2.2 (SHA-256 bd3d64f1...ebaec1).
Built as written where possible; every gap is filled by the smallest reasonable choice, listed in CLAUDE_CHOICES
(all Guess). Nothing was tuned on 2015-2025.

Version 0 (six core signals, section 2):
  f1  log1p(deaths per 100k, last 3 usable months)      f1b log1p(deaths per 100k, last 12 usable months)
  f2  log1p(nearby violence: deaths in OTHER countries within 500 km, weighted exp(-d/150 km), per 100k)
  f3  log1p(months since the last crisis on that lens), capped at 240
  f4  log(infant mortality / world median)              f5, f5b  V-Dem electoral democracy index and its square
  f6  log(population)
  + a missing flag per signal; missing values filled with the training mean within the same continent.
Model (3.3): elastic-net logistic per lens x horizon, penalty by time-ordered CV on development folds; rows before
1989 weight 0.5; fitted on forecast months whose whole window ends by Dec 2014; beta calibration (3.4) fitted on
the out-of-sample forecasts for 2010-2014 from a model trained to 2009; pool-adjacent-violators so the chance
never falls as the horizon grows (incidence target). Stacking (3.4) is built in with trust A_s = 0 (see C6).
Atrocity Alarm (3.7): separate elastic-net; label = a group passing 1,000 civilian deaths (C2 band 2) within
24 months; threshold = lowest cut with false-alarm rate <= 10% on development folds.

Usage: python3 brink_claude.py fit | predict_open
"""
import json, os, sys, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import brink_features as F
from brink_grok import load_outcomes, targets, H, LENSES

FEAT = os.path.join(ROOT, "data", "features"); MODELS = os.path.join(ROOT, "data", "models")
os.makedirs(FEAT, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
FIT_END = pd.Period("2014-12", "M")          # last month an outcome may be used (Exam Pack s.4)
CAL_FROM = pd.Period("2010-01", "M")         # trailing 60 months of out-of-sample forecasts for calibration
C_GRID = [0.001, 0.01, 0.1, 1.0]
CV_YEARS = [1995, 1998, 2001, 2004]          # time-ordered development folds: train <= y, validate y+1..y+3
FEATS = ["f1", "f1b", "f2", "f3", "f4", "f5", "f5b", "f6"]
EXTRA_CENTROIDS = {"DDR": (52.5, 12.5), "HKG": (22.3, 114.2), "PSB": (31.5, 35.0), "PSG": (31.4, 34.4),
                   "SML": (9.5, 46.0), "TWN": (23.7, 121.0), "VDR": (21.0, 105.8), "XKX": (42.6, 20.9),
                   "YMD": (13.5, 47.5), "ZZB": (-6.1, 39.3)}

CLAUDE_CHOICES = [
    "C1 Unit = country-month (the exam's unit). The design's province (ADM1) level is not built in Round 1: there are no province outcomes yet (Exam Pack s.1: province is a later Tier 3 add-on).",
    "C2 Target = the exam's C1-C7 onset/escalation event on each lens (Exam Pack s.2), not the design's own crisis rule (section 3.1). The DOMESTIC model is fitted to the exam's DOMESTIC event.",
    "C3 Nearby violence (f2): distance measured from each country's centroid (Gavin Rehkemper world-countries-centroids, MIT licence; 10 places added by hand, Guess); events in other countries within 500 km, weight exp(-d/150 km); divided by the country's own population (the design's 'population within 500 km' needs a population grid that is not in the build). Event points come from GED 26.1 restricted to months already published at t under the same rules as f1 (final releases only, no Candidate points): hindsight-risk flag.",
    "C4 Censorship correction (3.2) not applied (detection rate pi = 1): the design fits theta on Candidate-versus-final GED gaps, which exist only from 2018 (sealed years) and only for Africa until mid-2020. Fitting it would use the sealed period.",
    "C5 Refit cadence: the design refits monthly on a rolling origin. The exam forbids fitting on 2015-2025, so the model is fitted once on everything known by Dec 2014 and then frozen.",
    "C6 Stacking with outside sources (3.4) is coded, with every trust weight A_s = 0: trust must be earned on 60 months of resolved forecasts, and no outside source has a resolved record on the exam's target before 2015 except the PITF-style rebuild (Job 5). Job 5 records whether A_PITF earns weight on 2010-2014; until then the core model is the forecast.",
    "C7 Missing-data rule: the design's 'conditional mean from a fallback model (region, f4, f5, f6)' is simplified to the training mean within the same continent, plus the missing flag.",
    "C8 Candidate signals (C1-C9 in section 2) are not added: Version 0 is the six core signals, as the design says; candidates must first earn a place on development data, which is a later step.",
    "C9 f3 'months since last crisis': GLOBAL lens = months since the last month with any C1-C7 crisis under way (in_crisis); DOMESTIC lens = months since the last DOMESTIC event; capped at 240; months before any recorded crisis count from 1946-01 (Guess).",
    "C10 Bootstrap interval (3.5) is not computed for the exam: it is a display feature, not a scored output, and 200 refits x 12 models is out of scope for Round 1.",
    "C11 Atrocity Alarm label = a perpetrator crossing 1,000 civilian deaths in 12 months (exam C2 band 2) within the next 24 months, as the design's second definition; EWP's score is not an input (no EWP record before 2014). Forecast per country-year in December.",
]


# ------------------------------------------------------------------ features
def centroids():
    import country_converter as coco
    c = pd.read_csv(os.path.join(ROOT, "data", "raw", "country_centroids_gavinr.csv"))
    c["iso3"] = coco.convert(c.ISO.tolist(), src="ISO2", to="ISO3", not_found=None)
    d = {r.iso3: (r.latitude, r.longitude) for r in c.itertuples() if isinstance(r.iso3, str)}
    d.update(EXTRA_CENTROIDS)
    return d

def nearby_matrix(isos):
    """W[i, m]: deaths in other countries' events in month m within 500 km of i's centroid, weighted exp(-d/150)."""
    ev = F.rd("ucdp_ged_events_26_1").dropna(subset=["lat", "lon", "year", "month"])
    ev["ym"] = [pd.Period(year=int(y), month=int(m), freq="M") for y, m in zip(ev.year, ev.month)]
    cen = centroids()
    W = pd.DataFrame(0.0, index=isos, columns=F.MONTHS)
    lat = np.radians(ev.lat.values); lon = np.radians(ev.lon.values); best = ev.best.fillna(0).values
    evc = ev.iso3.values; evm = ev.ym.values
    col_idx = {m: k for k, m in enumerate(F.MONTHS)}
    mk = np.array([col_idx.get(m, -1) for m in evm])
    for iso in isos:
        if iso not in cen:
            W.loc[iso] = np.nan; continue
        la, lo = map(np.radians, cen[iso])
        d = 6371 * 2 * np.arcsin(np.sqrt(np.sin((lat - la) / 2) ** 2 + np.cos(la) * np.cos(lat) * np.sin((lon - lo) / 2) ** 2))
        sel = (d <= 500) & (evc != F.ucdp_iso(iso)) & (mk >= 0)
        if sel.any():
            agg = np.bincount(mk[sel], weights=best[sel] * np.exp(-d[sel] / 150), minlength=len(F.MONTHS))
            W.loc[iso] = agg
    return W

def build_features():
    isos, cy = F.universe()
    pop, _, h_pop = F.wdi_asof("population", isos)
    d1, d3, d12, lagm, h_ged = F.deaths_asof(isos)
    d3 = d3.reindex([F.ucdp_iso(i) for i in isos]); d3.index = isos
    d12 = d12.reindex([F.ucdp_iso(i) for i in isos]); d12.index = isos
    f1 = np.log1p(d3 / pop * 1e5); f1b = np.log1p(d12 / pop * 1e5)
    # f2: nearby violence, availability = newest final-release month usable at t (same rule as f1 before 2015-06,
    # newest final GED coverage after)
    W = nearby_matrix(isos)
    f2 = pd.DataFrame(np.nan, index=isos, columns=F.MONTHS)
    for t in F.MONTHS:
        if t < pd.Period("2015-06", "M"):
            A = pd.Period(year=(t.year - 1 if t.month >= 6 else t.year - 2), month=12, freq="M")
        else:
            rel = [pd.Period(r, "M") for r in F.GED_REL.values() if pd.Period(r, "M") <= t]
            A = pd.Period(year=max(rel).year - 1, month=12, freq="M")
        if A < pd.Period("1989-12", "M"):
            continue
        f2[t] = W[[A, A - 1, A - 2]].sum(axis=1)
    f2 = np.log1p(f2 / pop * 1e5)
    imr, imry, h_w = F.wdi_asof("infant_mortality", isos)
    wmed = F.wdi_world_mean_asof("infant_mortality", "median")
    eds = sorted(F.rd("wdi_vintages").edition.unique()); edp = {e: pd.Period(f"{e[:4]}-{e[4:6]}", "M") for e in eds}
    f4 = pd.DataFrame(np.nan, index=isos, columns=F.MONTHS)
    for t in F.MONTHS:
        cur = [e for e in eds if edp[e] <= t]; e = cur[-1] if cur else eds[0]
        med = imry[t].map(lambda y: wmed.get((e, int(y)), np.nan) if not np.isnan(y) else np.nan).astype(float)
        f4[t] = np.log(imr[t] / med)
    edi, _, h_v = F.vdem_asof("v2x_polyarchy", isos)
    f6 = np.log(pop)
    P = pd.concat([F.to_long(f1, "f1"), F.to_long(f1b, "f1b"), F.to_long(f2, "f2"), F.to_long(f4, "f4"),
                   F.to_long(edi, "f5"), F.to_long(f6, "f6")], axis=1)
    P.index.names = ["iso3", "ym"]; P = P.reset_index()
    P = P[[(i, t.year) in cy for i, t in zip(P.iso3, P.ym)]]
    P["f5b"] = P.f5 ** 2
    P["hindsight_flag"] = (h_ged.reindex(P.ym).values | h_w.reindex(P.ym).values | h_v.reindex(P.ym).values).astype(int)
    import country_converter as coco
    cont = dict(zip(isos, coco.convert(isos, to="continent", not_found=None)))
    cont.update({"SML": "Africa", "ZZB": "Africa", "XKX": "Europe", "DDR": "Europe", "PSG": "Asia", "PSB": "Asia", "VDR": "Asia", "YMD": "Asia"})
    P["continent"] = P.iso3.map(cont).fillna("Other")
    P.to_parquet(os.path.join(FEAT, "claude_panel.parquet"), index=False)
    return P


def history_f3(L, T):
    """Months since the last crisis, per lens, known at end of month t (C9)."""
    inc = L.groupby(["iso3", "ym"]).in_crisis.max().unstack("ym")
    dom = T.set_index(["iso3", "ym"]).domestic_event.unstack("ym")
    out = []
    for nm, M in [("GLOBAL", inc), ("DOMESTIC", dom)]:
        M = M.reindex(columns=pd.period_range(pd.Period("1946-01", "M"), M.columns.max(), freq="M"))
        A = (M.fillna(0).values == 1)
        res = np.zeros(A.shape)
        last = np.full(A.shape[0], -1)
        for k in range(A.shape[1]):
            last = np.where(A[:, k], k, last)
            res[:, k] = np.where(last >= 0, k - last, k + 1)
        df = pd.DataFrame(np.minimum(res, 240), index=M.index, columns=M.columns)
        out.append(F.to_long(df, "f3_" + nm))
    return pd.concat(out, axis=1).reset_index().rename(columns={"level_1": "ym"})


# ------------------------------------------------------------------ design matrix
def design(P, lens, stats=None):
    X = P[FEATS[:3] + ["f4", "f5", "f5b", "f6"]].copy()
    X.insert(3, "f3", np.log1p(P["f3_" + lens]))
    if stats is None:
        stats = {"cont_mean": {}, "mean": {}, "sd": {}}
        for c in X.columns:
            stats["cont_mean"][c] = X.groupby(P.continent)[c].mean().to_dict()
            stats["mean"][c] = float(X[c].mean())
    miss = X.isna().astype(float).add_prefix("miss_")
    for c in X.columns:
        fill = P.continent.map(stats["cont_mean"][c]).fillna(stats["mean"][c])
        X[c] = X[c].fillna(fill).fillna(stats["mean"][c])
    if not stats["sd"]:
        for c in X.columns:
            stats["sd"][c] = (float(X[c].mean()), float(X[c].std() or 1.0))
    for c in X.columns:
        m, s = stats["sd"][c]; X[c] = (X[c] - m) / s
    X = pd.concat([X, miss[["miss_f1", "miss_f2", "miss_f4", "miss_f5", "miss_f6"]]], axis=1)
    return X, stats

def wts(P):
    return np.where(P.ym.dt.year < 1989, 0.5, 1.0)

def enet(C):
    return LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=C, solver="saga", max_iter=500, tol=1e-3)

def beta_cal_fit(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    Z = np.c_[np.log(p), -np.log(1 - p)]
    m = LogisticRegression(C=1e6, max_iter=1000).fit(Z, y)
    return {"a": float(m.coef_[0][0]), "b": float(m.coef_[0][1]), "c": float(m.intercept_[0])}

def beta_cal_apply(p, cal):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = cal["a"] * np.log(p) - cal["b"] * np.log(1 - p) + cal["c"]
    return 1 / (1 + np.exp(-z))

def pav_rows(M):
    """Pool-adjacent-violators per row: non-decreasing across horizons (equal weights)."""
    out = M.copy()
    for r in range(M.shape[0]):
        y = list(M[r]); w = [1.0] * len(y); blocks = [[y[0], 1.0, 1]]
        for v in y[1:]:
            blocks.append([v, 1.0, 1])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                v2, w2, n2 = blocks.pop(); v1, w1, n1 = blocks.pop()
                blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
        vals = []
        for v, _, n in blocks:
            vals += [v] * n
        out[r] = vals
    return out


# ------------------------------------------------------------------ fit / predict
def prep(long_paths, target_paths):
    P = pd.read_parquet(os.path.join(FEAT, "claude_panel.parquet"))
    P["ym"] = pd.PeriodIndex(P.ym, freq="M")
    L, T = load_outcomes(long_paths, target_paths)
    P = P.merge(history_f3(L, T), on=["iso3", "ym"], how="left")
    return P.sort_values(["iso3", "ym"]).reset_index(drop=True), L, T

def fit():
    O = os.path.join(ROOT, "data", "outcomes")
    P, L, T = prep([os.path.join(O, "outcomes_long_open.csv.gz")], [os.path.join(O, "targets_open.csv.gz")])
    params = {"choices": CLAUDE_CHOICES, "fit_end": str(FIT_END), "models": {}, "trust": {"PITF": 0.0, "EWP": 0.0, "VIEWS": 0.0, "markets": 0.0}}
    ym = P.ym.values
    for h in H:
        Y = P[["iso3", "ym"]].merge(targets(T, h), on=["iso3", "ym"], how="left")
        # penalty: time-ordered CV, Brier averaged over lenses (quarterly months for compute)
        q = P.ym.dt.month.isin([1, 4, 7, 10]).values
        sc = {c: [] for c in C_GRID}
        for lens in LENSES:
            yv = Y[f"Y_{lens}_{h}"].values
            for y in CV_YEARS:
                end = pd.Period(year=y, month=12, freq="M")
                trm = (ym <= end - h) & ~np.isnan(yv) & q
                vam = (ym > end) & (ym <= end + 36) & (ym <= FIT_END - h) & ~np.isnan(yv) & q
                Xtr, st = design(P[trm], lens); Xva, _ = design(P[vam], lens, st)
                for c in C_GRID:
                    m = enet(c).fit(Xtr.values, yv[trm], sample_weight=wts(P[trm]))
                    sc[c].append(float(np.mean((m.predict_proba(Xva.values)[:, 1] - yv[vam]) ** 2)))
        best = min(C_GRID, key=lambda c: np.mean(sc[c]))
        print(f"h={h} C={best} cv Brier " + ", ".join(f"{c}:{np.mean(v):.4f}" for c, v in sc.items()), flush=True)
        for lens in LENSES:
            yv = Y[f"Y_{lens}_{h}"].values
            # (a) out-of-sample forecasts 2010-2014 from a model trained to 2009 -> beta calibration
            tr0 = (ym <= pd.Period("2009-12", "M") - h) & ~np.isnan(yv)
            ca = (ym >= CAL_FROM) & (ym <= FIT_END - h) & ~np.isnan(yv)
            X0, st0 = design(P[tr0], lens); m0 = enet(best).fit(X0.values, yv[tr0], sample_weight=wts(P[tr0]))
            Xc, _ = design(P[ca], lens, st0)
            cal = beta_cal_fit(m0.predict_proba(Xc.values)[:, 1], yv[ca])
            # (b) final core model on everything known by Dec 2014
            tr = (ym <= FIT_END - h) & ~np.isnan(yv)
            X, st = design(P[tr], lens); m = enet(best).fit(X.values, yv[tr], sample_weight=wts(P[tr]))
            params["models"][f"{lens}_{h}"] = {"C": best, "features": list(X.columns), "coef": m.coef_[0].tolist(),
                                               "intercept": float(m.intercept_[0]), "stats": st, "beta_cal": cal,
                                               "n_train": int(tr.sum()), "events_train": int(yv[tr].sum())}
    params["atrocity"] = fit_alarm(P, L)
    js = json.dumps(params, indent=1, default=str)
    open(os.path.join(MODELS, "claude_params.json"), "w").write(js)
    print("locked params sha256", hashlib.sha256(js.encode()).hexdigest())

def alarm_label(L):
    c2 = L[L.ctype == "C2"].copy()
    ev = ((c2.band >= 2) & ((c2.onset == 1) | (c2.escalation == 1))).astype(float).where(c2.observed == 1)
    c2["ev"] = ev
    M = c2.set_index(["iso3", "ym"]).ev.unstack("ym")
    M = M.reindex(columns=pd.period_range(M.columns.min(), M.columns.max(), freq="M"))
    A = M.T
    y = (A == 1).astype(float)[::-1].rolling(24, min_periods=1).sum()[::-1].shift(-1)
    known = A.notna()[::-1].rolling(24, min_periods=1).sum()[::-1].shift(-1)
    out = pd.DataFrame(np.nan, index=A.index, columns=A.columns)
    out[known >= 24] = 0.0; out[y > 0] = 1.0
    return F.to_long(out.T, "y_alarm").reset_index().rename(columns={"level_1": "ym"})

def fit_alarm(P, L):
    D = P[P.ym.dt.month == 12].merge(alarm_label(L), on=["iso3", "ym"], how="left")
    X, st = design(D, "GLOBAL")
    y = D.y_alarm.values; ym = D.ym.values
    tr = (ym <= FIT_END - 24) & ~np.isnan(y)
    # threshold on development folds: out-of-fold predictions by year blocks
    oof = np.full(len(y), np.nan)
    for y0 in [1995, 2000, 2005]:
        a = tr & (ym <= pd.Period(f"{y0}-12", "M") - 24); b = tr & (ym > pd.Period(f"{y0}-12", "M")) & (ym <= pd.Period(f"{y0 + 5}-12", "M"))
        if (y[a] == 1).sum() >= 3 and b.any():
            oof[b] = enet(0.1).fit(X.values[a], y[a]).predict_proba(X.values[b])[:, 1]
    ok = ~np.isnan(oof)
    neg = np.sort(oof[ok & (y == 0)])
    thr = float(neg[int(np.ceil(0.9 * len(neg))) - 1]) if len(neg) else 1.0
    m = enet(0.1).fit(X.values[tr], y[tr])
    return {"features": list(X.columns), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "stats": st,
            "threshold": thr, "dev_recall": float(np.mean(oof[ok & (y == 1)] > thr)) if (ok & (y == 1)).any() else None,
            "dev_fpr": float(np.mean(oof[ok & (y == 0)] > thr)) if len(neg) else None, "n_train": int(tr.sum()), "cases_train": int(np.nansum(y[tr]))}

def predict(params, long_paths, target_paths, last_origin=None):
    P, L, T = prep(long_paths, target_paths)
    if last_origin:
        P = P[P.ym <= pd.Period(last_origin, "M")].reset_index(drop=True)
    out = P[["iso3", "ym", "hindsight_flag"]].copy()
    for lens in LENSES:
        mats = []
        for h in H:
            md = params["models"][f"{lens}_{h}"]
            X, _ = design(P, lens, md["stats"]); X = X[md["features"]]
            z = md["intercept"] + X.values @ np.array(md["coef"])
            p = beta_cal_apply(1 / (1 + np.exp(-z)), md["beta_cal"])
            mats.append(p)
        M = pav_rows(np.vstack(mats).T)
        for k, h in enumerate(H):
            out[f"p_{lens}_{h}"] = np.clip(M[:, k], 1e-4, 1 - 1e-4)
    out["coverage"] = P[["f1", "f2", "f4", "f5", "f6"]].notna().mean(axis=1).values
    a = params["atrocity"]
    D = P[P.ym.dt.month == 12].reset_index(drop=True)
    Xa, _ = design(D, "GLOBAL", a["stats"])
    pa = 1 / (1 + np.exp(-(a["intercept"] + Xa[a["features"]].values @ np.array(a["coef"]))))
    alarm = pd.DataFrame({"iso3": D.iso3, "year": D.ym.dt.year + 1, "p_atrocity": pa, "alarm": (pa > a["threshold"]).astype(int)})
    return out, alarm


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    O = os.path.join(ROOT, "data", "outcomes")
    if cmd == "features":
        build_features()
    elif cmd == "fit":
        if not os.path.exists(os.path.join(FEAT, "claude_panel.parquet")):
            build_features()
        fit()
    elif cmd == "predict_open":
        params = json.load(open(os.path.join(MODELS, "claude_params.json")))
        out, alarm = predict(params, [os.path.join(O, "outcomes_long_open.csv.gz")], [os.path.join(O, "targets_open.csv.gz")], "2014-12")
        out.to_parquet(os.path.join(MODELS, "claude_forecasts_open.parquet"), index=False)
        alarm.to_parquet(os.path.join(MODELS, "claude_alarm_open.parquet"), index=False)
        print(out.describe().T.to_string())
