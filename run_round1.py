#!/usr/bin/env python3
"""
Brink Model - Round 1 build - JOB 6: run the locked Round 1 test exactly as Exam Pack v2.2 says.

Order (nothing is refitted; every design and benchmark uses the settings locked before this run):
  1. check the exam fingerprint parts on disk match exam_fingerprint.json (recorded in the Decision Log first);
  2. open the sealed 2015-2025 outcomes (first and only time) and join them to the open years;
  3. forecasts for every country-month with data available at each date (history features use outcomes up to
     the forecast month only); benchmarks: Naive, Base rate, PITF-style, EWP (own target), VIEWS (C1/C2, report);
  4. scores for forecast months 2015-01 .. 2025-12 minus h, both lenses, six horizons; PASS gates with Holm
     correction and country-year block bootstrap; Atrocity Alarm per country-year 2015-2025;
  5. write data/round1/*.csv and the results document.
"""
import hashlib, json, os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import brink_score as S, brink_benchmarks as BM
import brink_grok, brink_claude, brink_astra
from brink_grok import load_outcomes, history_features, H, LENSES

O = os.path.join(ROOT, "data", "outcomes"); M = os.path.join(ROOT, "data", "models"); R = os.path.join(ROOT, "data", "round1")
os.makedirs(R, exist_ok=True)
LONG = [os.path.join(O, "outcomes_long_open.csv.gz"), os.path.join(O, "outcomes_long_sealed.csv.gz")]
TARG = [os.path.join(O, "targets_open.csv.gz"), os.path.join(O, "targets_sealed.csv.gz")]

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def check_fingerprint():
    fp = json.load(open(os.path.join(ROOT, "pub", "exam_fingerprint.json")))
    files = {"brink_outcomes.py": "brink_outcomes.py", "outcomes_long_open.csv.gz": O + "/outcomes_long_open.csv.gz",
             "outcomes_long_sealed.csv.gz": O + "/outcomes_long_sealed.csv.gz", "targets_open.csv.gz": O + "/targets_open.csv.gz",
             "targets_sealed.csv.gz": O + "/targets_sealed.csv.gz", "brink_score.py": "brink_score.py",
             "brink_benchmarks.py": "brink_benchmarks.py", "pitf_style_params.json": M + "/pitf_style_params.json",
             "grok_params.json": M + "/grok_params.json", "claude_params.json": M + "/claude_params.json",
             "astra_params.json": M + "/astra_params.json"}
    bad = [k for k, p in files.items() if sha(os.path.join(ROOT, p) if not p.startswith("/") else p) != fp["parts"][k]]
    print("fingerprint", fp["exam_fingerprint"], "mismatches:", bad, flush=True)
    return fp["exam_fingerprint"], bad

def c12_targets(L):
    """C1/C2-only event table for the VIEWS report (same rules as the main targets)."""
    x = L[L.ctype.isin(["C1", "C2"])]
    g = x.groupby(["iso3", "ym"])
    T = pd.DataFrame({"global_event": g.g_event.max(), "domestic_event": g.d_event.max(),
                      "near": g.near.max(), "obs": g["observed"].sum()}).reset_index()
    for c in ("global_event", "domestic_event"):
        T.loc[(T.near == 1) & (T[c].fillna(0) == 0), c] = np.nan
        T.loc[T.obs == 0, c] = np.nan
    return T

def run_gates(args):
    name, f, benches, T, ewp_rows = args
    t0 = time.time()
    Mg, Wg, ok = S.gates(name, f, benches, T, ewp_rows)
    print(name, "gates done", round(time.time() - t0), "s", flush=True)
    return name, Mg, Wg, ok

def main():
    fp, bad = check_fingerprint()
    L, T = load_outcomes(LONG, TARG)
    designs = {}
    for nm, mod, pf in [("Grok", brink_grok, "grok_params.json"), ("Claude", brink_claude, "claude_params.json"), ("GPT Astra", brink_astra, "astra_params.json")]:
        params = json.load(open(os.path.join(M, pf)))
        out, alarm = mod.predict(params, LONG, TARG, None)
        out["ym"] = pd.PeriodIndex(out.ym, freq="M") if not isinstance(out.ym.dtype, pd.PeriodDtype) else out.ym
        out.to_parquet(os.path.join(R, f"forecasts_{nm.replace(' ', '_')}.parquet"), index=False)
        alarm.to_csv(os.path.join(R, f"alarm_{nm.replace(' ', '_')}.csv"), index=False)
        designs[nm] = (out, alarm)
        print("forecasts", nm, len(out), flush=True)
    D = BM.pitf_frame()
    pp = json.load(open(os.path.join(M, "pitf_style_params.json")))["pitf_style"]
    Hh = history_features(L, T)
    benches = {"Base rate": BM.base_rate(D[["iso3", "ym"]], T), "PITF-style": BM.pitf_predict(D, pp), "Naive": BM.naive(D[["iso3", "ym"]], Hh)}
    for k, v in benches.items():
        v.to_parquet(os.path.join(R, f"forecasts_{k.replace(' ', '_')}.parquet"), index=False)
    # score table: designs + benchmarks
    allm = {k: v[0] for k, v in designs.items()}; allm.update(benches)
    tab = S.score_table(allm, T)
    tab.to_csv(os.path.join(R, "scores_all.csv"), index=False)
    # reliability curves
    rel = []
    for nm, f in allm.items():
        for lens in LENSES:
            for h in H:
                d = S.panel(f, T, lens, h)
                if len(d):
                    for b in S.reliability(d.y.values, d.p.values):
                        rel.append({"model": nm, "lens": lens, "h": h, **b})
    pd.DataFrame(rel).to_csv(os.path.join(R, "reliability.csv"), index=False)
    # VIEWS (report only, C1/C2)
    V = BM.views_forecasts()
    vt = None
    if len(V):
        T12 = c12_targets(L)
        orig = D[(D.ym >= pd.Period("2021-01", "M"))][["iso3", "ym"]]
        vf = BM.views_predict(orig, V)
        c12 = {"VIEWS (mapped)": vf}
        for k, v in list(designs.items()) + [("Base rate", (benches["Base rate"], None)), ("PITF-style", (benches["PITF-style"], None))]:
            c12[k] = v[0]
        S.SCORE_FROM = pd.Period("2021-01", "M")
        vt = S.score_table({k: f[f.iso3.isin(vf.iso3.unique())] if k != "VIEWS (mapped)" else f for k, f in c12.items()}, T12)
        S.SCORE_FROM = pd.Period("2015-01", "M")
        vt.to_csv(os.path.join(R, "scores_views_c1c2.csv"), index=False)
    # EWP rows and gates
    ewp_f = BM.ewp_forecasts(); ewp_t = BM.ewp_target()
    jobs = []
    for nm, (f, alarm) in designs.items():
        er = S.ewp_join(alarm, ewp_f, ewp_t)
        er = er[(er.window_start >= 2015) & (er.window_start <= 2024)]
        er.to_csv(os.path.join(R, f"ewp_rows_{nm.replace(' ', '_')}.csv"), index=False)
        jobs.append((nm, f, {"Base rate": benches["Base rate"], "PITF-style": benches["PITF-style"]}, T, er))
    with Pool(2) as pool:
        res = pool.map(run_gates, jobs)
    summary = {"exam_fingerprint": fp, "fingerprint_mismatches": bad, "designs": {}}
    for nm, Mg, Wg, ok in res:
        Mg.to_csv(os.path.join(R, f"gates_main_{nm.replace(' ', '_')}.csv"), index=False)
        Wg.to_csv(os.path.join(R, f"gates_worse_{nm.replace(' ', '_')}.csv"), index=False)
        al = S.alarm_score(designs[nm][1], L)
        summary["designs"][nm] = {"gates_pass": ok, "main_gates_passed": int(Mg["pass"].sum()) if len(Mg) else 0,
                                  "main_gates_total": int(len(Mg)), "significantly_worse_checks": int(Wg.significantly_worse.sum()) if len(Wg) else 0,
                                  "alarm": al, "PASS": bool(ok and al["pass"])}
    # EWP published vs its own target, for reference
    e0 = S.ewp_join(designs["Grok"][1], ewp_f, ewp_t); e0 = e0[(e0.window_start >= 2015) & (e0.window_start <= 2024)].dropna(subset=["p_ewp", "y"])
    summary["ewp_reference"] = {"rows": len(e0), "onset_windows": int(e0.y.sum()), "ewp_prauc": S.prauc(e0.y.values, e0.p_ewp.values) if e0.y.sum() else None,
                                "ewp_brier": S.brier(e0.y.values, e0.p_ewp.values) if len(e0) else None}
    json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))

if __name__ == "__main__":
    main()
