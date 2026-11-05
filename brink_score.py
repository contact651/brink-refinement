#!/usr/bin/env python3
"""
Brink Model - Round 1 build - JOB 5 (part 2): the scoring code (Exam Pack v2.2 s.6-7), pre-registered before
any sealed-year run.

Scores (per model, lens, horizon h in 1,3,6,12,18,24; forecast months 2015-01 .. 2025-12 minus h, so 24-month
forecasts are scored to Dec 2023 issue dates):
  - PR-AUC (average precision), Brier score, reliability curve (10 equal-width bins),
  - false-alarm rate at the threshold that catches 70% of events (FP / all non-events at that threshold).
  Rows: the forecast month's outcome window is known (Y = 0/1); grey-zone or missing windows are dropped.
  Comparisons use the rows both sides cover (cells one side does not cover are dropped from both).

PASS gates (s.7), per design:
  G-main (24 gates): beats Base rate AND PITF-style on PR-AUC (higher) and Brier (lower) at 3, 6, 12 months on
    BOTH lenses. One-sided test per gate: country-year block bootstrap (B = 1000, seed 20261022) of the score
    difference; p = (1 + #replicates with no improvement) / (B + 1).
  G-worse (48 checks): not significantly WORSE than Base rate or PITF-style on PR-AUC or Brier at 1, 18, 24 months,
    or on the false-alarm rate at 70% recall at any horizon (one-sided test for 'worse', same bootstrap).
  G-EWP (2 gates): beats EWP on EWP's own target (onset of state-led mass killing within EWP's 2-year window) on
    PR-AUC and Brier, using the design's own Atrocity Alarm probability; country-year rows.
  Holm correction at alpha = 0.05 across all G-main and G-EWP gates together; every gate must pass.
  G-worse fails the design if any check is significant after Holm correction across the 48 checks.
  Atrocity Alarm (s.6): per country-year 2015-2025, scored on exam C2 onsets: false alarms <= 1 per 10 non-cases
    AND catches at least half of onsets. Reported beside the gates (criterion 15), and required for PASS.
  Naive, VIEWS (C1/C2 only) and markets: reported side by side, not gated.
  Also reported: every score split by hindsight_flag (0 = real vintages only, 1 = some input from a lag rule).
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from brink_grok import targets, H, LENSES

B = 1000
SEED = 20261022
ALPHA = 0.05
SCORE_FROM, SCORE_TO = pd.Period("2015-01", "M"), pd.Period("2025-12", "M")
GATE_H = [3, 6, 12]
OTHER_H = [1, 18, 24]


def brier(y, p):
    return float(np.mean((p - y) ** 2))

def prauc(y, p):
    return float(average_precision_score(y, p)) if y.sum() > 0 else np.nan

def far70(y, p):
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-p, kind="mergesort"); ys = y[o]; ps = p[o]
    tp = np.cumsum(ys); k = int(np.searchsorted(tp, 0.7 * y.sum()))
    thr = ps[min(k, len(ps) - 1)]
    fire = p >= thr
    neg = (y == 0)
    return float(np.sum(fire & neg) / max(1, neg.sum()))

def reliability(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return [{"bin": f"{k / bins:.1f}-{(k + 1) / bins:.1f}", "mean_p": float(p[b == k].mean()) if (b == k).any() else None,
             "rate": float(y[b == k].mean()) if (b == k).any() else None, "n": int((b == k).sum())} for k in range(bins)]

METRICS = {"prauc": (prauc, +1), "brier": (brier, -1), "far70": (far70, -1)}   # sign: +1 higher is better


def panel(forecasts, T, lens, h, frm=None, to=None):
    """Join a forecast table to Y(t,h); keep forecast months frm..to-h with known Y."""
    frm = SCORE_FROM if frm is None else frm; to = SCORE_TO if to is None else to
    Y = targets(T, h)[["iso3", "ym", f"Y_{lens}_{h}"]]
    d = forecasts[["iso3", "ym", f"p_{lens}_{h}"] + (["hindsight_flag"] if "hindsight_flag" in forecasts else [])].merge(Y, on=["iso3", "ym"])
    d = d[(d.ym >= frm) & (d.ym <= to - h)]
    d = d.dropna(subset=[f"p_{lens}_{h}", f"Y_{lens}_{h}"])
    return d.rename(columns={f"p_{lens}_{h}": "p", f"Y_{lens}_{h}": "y"})

def score_table(models, T, frm=None, to=None):
    rows = []
    for name, f in models.items():
        for lens in LENSES:
            for h in H:
                if f"p_{lens}_{h}" not in f:
                    continue
                d = panel(f, T, lens, h, frm, to)
                if d.empty:
                    continue
                y, p = d.y.values, d.p.values
                r = {"model": name, "lens": lens, "h": h, "n": len(d), "events": int(y.sum()), "prauc": prauc(y, p),
                     "brier": brier(y, p), "far70": far70(y, p), "event_rate": float(y.mean())}
                if "hindsight_flag" in d:
                    for hf in (0, 1):
                        s = d[d.hindsight_flag == hf]
                        if len(s) and s.y.sum() > 0:
                            r[f"prauc_hf{hf}"] = prauc(s.y.values, s.p.values); r[f"brier_hf{hf}"] = brier(s.y.values, s.p.values); r[f"n_hf{hf}"] = len(s)
                rows.append(r)
    return pd.DataFrame(rows)

def blocks_index(d):
    key = d.iso3.astype(str) + "_" + d.ym.dt.year.astype(str)
    codes, uniq = pd.factorize(key)
    idx = [np.where(codes == k)[0] for k in range(len(uniq))]
    return idx

def boot_test(y, pa, pb, metric, blocks, direction, rng):
    """One-sided p-value. direction='better': H1 = A better than B. 'worse': H1 = A worse than B."""
    fn, sign = METRICS[metric]
    obs = sign * (fn(y, pa) - fn(y, pb))           # > 0 means A better
    cnt = 0; nb = len(blocks)
    for _ in range(B):
        pick = rng.integers(0, nb, nb)
        ix = np.concatenate([blocks[k] for k in pick])
        dlt = sign * (fn(y[ix], pa[ix]) - fn(y[ix], pb[ix]))
        if direction == "better" and not (dlt > 0):
            cnt += 1
        if direction == "worse" and not (dlt < 0):
            cnt += 1
    return float(obs), (1 + cnt) / (B + 1)

def pair(fa, fb, T, lens, h):
    a = panel(fa, T, lens, h).rename(columns={"p": "pa"})
    b = panel(fb, T, lens, h)[["iso3", "ym", "p"]].rename(columns={"p": "pb"})
    d = a.merge(b, on=["iso3", "ym"]).dropna(subset=["pa", "pb"])
    return d

def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); reject = np.zeros(m, bool)
    for rank, k in enumerate(order):
        if pvals[k] <= ALPHA / (m - rank):
            reject[k] = True
        else:
            break
    return reject

def gates(design_name, design_f, benches, T, ewp_rows=None):
    rng = np.random.default_rng(SEED)
    main, worse = [], []
    for bname in ["Base rate", "PITF-style"]:
        fb = benches[bname]
        for lens in LENSES:
            for h in H:
                d = pair(design_f, fb, T, lens, h)
                if d.empty or d.y.sum() == 0:
                    continue
                bl = blocks_index(d); y = d.y.values; pa = d.pa.values; pb = d.pb.values
                if h in GATE_H:
                    for met in ["prauc", "brier"]:
                        obs, p = boot_test(y, pa, pb, met, bl, "better", rng)
                        main.append({"vs": bname, "lens": lens, "h": h, "metric": met, "diff_better": obs, "p": p, "n": len(d)})
                for met in (["prauc", "brier"] if h in OTHER_H else []) + ["far70"]:
                    obs, p = boot_test(y, pa, pb, met, bl, "worse", rng)
                    worse.append({"vs": bname, "lens": lens, "h": h, "metric": met, "diff_better": obs, "p_worse": p, "n": len(d)})
    if ewp_rows is not None and len(ewp_rows):
        e = ewp_rows.dropna(subset=["p_design", "p_ewp", "y"])
        bl = [np.where((e.iso3 == i).values & (e.window_start == w).values)[0] for i, w in e[["iso3", "window_start"]].drop_duplicates().values]
        for met in ["prauc", "brier"]:
            if e.y.sum() > 0:
                obs, p = boot_test(e.y.values, e.p_design.values, e.p_ewp.values, met, bl, "better", rng)
            else:
                obs, p = np.nan, 1.0
            main.append({"vs": "EWP (own target)", "lens": "-", "h": 24, "metric": met, "diff_better": obs, "p": p, "n": len(e)})
    M = pd.DataFrame(main); W = pd.DataFrame(worse)
    if len(M):
        M["pass"] = holm(M.p.values) & (M.diff_better > 0)
    if len(W):
        W["significantly_worse"] = holm(W.p_worse.values)
    ok = bool(len(M) and M["pass"].all() and len(M) == 26) and not bool(len(W) and W.significantly_worse.any())
    return M, W, ok

def alarm_score(alarm, L, years=range(2015, 2026)):
    c2 = L[L.ctype == "C2"].copy(); c2["year"] = c2.ym.dt.year
    g = c2.groupby(["iso3", "year"]).agg(on=("onset", "max"), obs=("observed", "max"), near=("near", "max")).reset_index()
    g = g[g.year.isin(list(years)) & (g.obs == 1)]
    g = g[~((g.on == 1) & (g.near == 1))]
    d = g.merge(alarm, on=["iso3", "year"], how="inner")
    cases = d[d.on == 1]; non = d[d.on == 0]
    far = float(non.alarm.mean()) if len(non) else np.nan
    rec = float(cases.alarm.mean()) if len(cases) else np.nan
    return {"country_years": len(d), "cases": len(cases), "non_cases": len(non), "false_alarm_rate": far, "recall": rec,
            "alarms": int(d.alarm.sum()), "pass": bool(far <= 0.10 and rec >= 0.5) if len(cases) else False}

def ewp_join(alarm, ewp_f, ewp_t):
    """Rows: (iso3, window_start Y): EWP published p vs the design's alarm probability for year Y; y = EWP onset in {Y, Y+1}."""
    t = ewp_t.set_index(["iso3", "year"]).ewp_onset
    rows = []
    for r in ewp_f.itertuples():
        a, b = t.get((r.iso3, r.window_start), np.nan), t.get((r.iso3, r.window_start + 1), np.nan)
        y = 1.0 if (a == 1 or b == 1) else (0.0 if (a == 0 and b == 0) else np.nan)
        if r.window_start + 1 > 2025:
            y = np.nan
        rows.append((r.iso3, r.window_start, r.p_ewp, y))
    E = pd.DataFrame(rows, columns=["iso3", "window_start", "p_ewp", "y"])
    A = alarm.rename(columns={"year": "window_start", "p_atrocity": "p_design"})[["iso3", "window_start", "p_design"]]
    return E.merge(A, on=["iso3", "window_start"], how="inner")
