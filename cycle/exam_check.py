#!/usr/bin/env python3
"""
Exam check (Refinement Loop step e): Working Model v1.1 candidate = v1.0 + own track record on the GLOBAL lens (I4b).
Fitted exactly the Round 1 way on open years only (everything known by Dec 2014; beta calibration on 2010-2014
forecasts from a model fitted to Dec 2009; penalty = v1.0's locked values). DOMESTIC lens and Atrocity Alarm = locked
v1.0 (claude_params.json, unchanged). Sealed 2015-2025 outcomes are joined only here, after the fingerprint check.
Scored with the frozen Round 1 scoring code (brink_score.py) against Base rate and PITF-style; v1.0 re-scored on the
same rows for the before/after.
"""
import os, sys, json, hashlib
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cycle"))
import brink_claude as BC, brink_score as S, brink_benchmarks as BM
from brink_grok import targets, H, LENSES
BC.enet = lambda C: LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=C, solver="saga", max_iter=500, tol=1e-3, random_state=0)
import practice as PR

O = os.path.join(ROOT, "data", "outcomes"); OUT = os.path.join(ROOT, "cycle", "exam"); os.makedirs(OUT, exist_ok=True)
LONG = [os.path.join(O, "outcomes_long_open.csv.gz"), os.path.join(O, "outcomes_long_sealed.csv.gz")]
TARG = [os.path.join(O, "targets_open.csv.gz"), os.path.join(O, "targets_sealed.csv.gz")]
FP = {"outcomes_long_open.csv.gz": "d87721cf5d927308407053beb3fed926fce74d9b5f0c786c03cd99d137581324",
      "outcomes_long_sealed.csv.gz": "782b7bed82203390051da8a9da5f738cf1774fc29363a6d75292fd0295313e6e",
      "targets_open.csv.gz": "73b9a8c9499d267e780b8d96df98aeb41bd7ca51388f438a7babd700923ce585",
      "targets_sealed.csv.gz": "3859943ae8fb6939fc24dc2910c7695cedf25b665ebc2ea008390fc8d62f1c12",
      "brink_score.py": "95ad90706cf36f922aee581d57fb86b5b209da25a4d7d94ab6e8227f07f5af47",
      "claude_params.json": "044791c5bde35f0fe52bfb87b83630adaea390c7049859b4369f17f22d8f6438",
      "pitf_style_params.json": "61f680afe01221c271a5e238b50a62725441e33fb36edd442855d7d2a6a07620"}
paths = {k: os.path.join(O, k) for k in FP if k.endswith(".gz")}
paths.update({"brink_score.py": os.path.join(ROOT, "brink_score.py"), "claude_params.json": os.path.join(ROOT, "claude_params.json"),
              "pitf_style_params.json": os.path.join(ROOT, "pitf_style_params.json")})
bad = [k for k, p in paths.items() if hashlib.sha256(open(p, "rb").read()).hexdigest() != FP[k]]
print("fingerprint parts mismatches:", bad, flush=True)
assert not bad

FIT_END, CAL_FROM, CAL_TRAIN_END = pd.Period("2014-12", "M"), pd.Period("2010-01", "M"), pd.Period("2009-12", "M")
C_LOCK = PR.C_LOCK

P, L, T = BC.prep(LONG, TARG)
locked = json.load(open(paths["claude_params.json"]))
v10, alarm = BC.predict(locked, LONG, TARG, None)
v10["ym"] = pd.PeriodIndex(v10.ym, freq="M") if not isinstance(v10.ym.dtype, pd.PeriodDtype) else v10.ym

br = BM.base_rate(P[["iso3", "ym"]], T)
ym = P.ym.values; wt = BC.wts(P)
v11 = v10.copy()
mats = []; coefs = {}
for h in H:
    b = br[f"p_GLOBAL_{h}"].clip(1e-4, 1 - 1e-4).values
    Px = P.assign(x_own=np.log(b / (1 - b)))
    yv = Px[["iso3", "ym"]].merge(targets(T, h), on=["iso3", "ym"], how="left")[f"Y_GLOBAL_{h}"].values
    tr0 = (ym <= CAL_TRAIN_END - h) & ~np.isnan(yv)
    ca = (ym >= CAL_FROM) & (ym <= FIT_END - h) & ~np.isnan(yv)
    X0, s0 = PR.design(Px[tr0], "GLOBAL", ["x_own"])
    m0 = BC.enet(C_LOCK[h]).fit(X0.values, yv[tr0], sample_weight=wt[tr0])
    Xc, _ = PR.design(Px[ca], "GLOBAL", ["x_own"], s0)
    cal = BC.beta_cal_fit(m0.predict_proba(Xc.values)[:, 1], yv[ca])
    tr = (ym <= FIT_END - h) & ~np.isnan(yv)
    X, s = PR.design(Px[tr], "GLOBAL", ["x_own"])
    m = BC.enet(C_LOCK[h]).fit(X.values, yv[tr], sample_weight=wt[tr])
    coefs[h] = dict(zip(X.columns, m.coef_[0].round(4).tolist())); coefs[h]["n_train"] = int(tr.sum())
    Xa, _ = PR.design(Px, "GLOBAL", ["x_own"], s)
    mats.append(BC.beta_cal_apply(m.predict_proba(Xa.values)[:, 1], cal))
M = BC.pav_rows(np.vstack(mats).T)
assert (v11.iso3.values == P.iso3.values).all()
for k, h in enumerate(H):
    v11[f"p_GLOBAL_{h}"] = np.clip(M[:, k], 1e-4, 1 - 1e-4)
json.dump({str(k): v for k, v in coefs.items()}, open(os.path.join(OUT, "v1_1_global_coefs.json"), "w"), indent=1)
v11.to_parquet(os.path.join(OUT, "forecasts_v1_1.parquet"), index=False); v10.to_parquet(os.path.join(OUT, "forecasts_v1_0.parquet"), index=False)

D = BM.pitf_frame()
pp = json.load(open(paths["pitf_style_params.json"]))["pitf_style"]
benches = {"Base rate": BM.base_rate(D[["iso3", "ym"]], T), "PITF-style": BM.pitf_predict(D, pp)}
tab = S.score_table({"v1.0": v10, "v1.1": v11, **benches}, T)
tab.to_csv(os.path.join(OUT, "scores.csv"), index=False)
print(tab[["model", "lens", "h", "n", "events", "prauc", "brier", "far70"]].round(4).to_string(), flush=True)
res = {}
for nm, f in [("v1.0", v10), ("v1.1", v11)]:
    Mg, Wg, ok = S.gates(nm, f, benches, T, None)
    Mg.to_csv(os.path.join(OUT, f"gates_main_{nm}.csv"), index=False); Wg.to_csv(os.path.join(OUT, f"gates_worse_{nm}.csv"), index=False)
    res[nm] = {"main_gates_passed_of_24": int(Mg["pass"].sum()), "significantly_worse_of_48": int(Wg.significantly_worse.sum()),
               "worse_list": Wg[Wg.significantly_worse][["vs", "lens", "h", "metric", "diff_better"]].to_dict("records"),
               "passed_list": Mg[Mg["pass"]][["vs", "lens", "h", "metric", "diff_better"]].to_dict("records")}
    print(nm, json.dumps(res[nm], indent=1, default=str), flush=True)
# direct v1.1 vs v1.0 on the exam years (GLOBAL lens; DOMESTIC identical by construction)
rng = np.random.default_rng(S.SEED); direct = []
for h in H:
    d = S.pair(v11, v10, T, "GLOBAL", h); bl = S.blocks_index(d)
    for met in ["prauc", "brier", "far70"]:
        obs, p = S.boot_test(d.y.values, d.pa.values, d.pb.values, met, bl, "better", rng)
        obs2, pw = S.boot_test(d.y.values, d.pa.values, d.pb.values, met, bl, "worse", rng)
        direct.append({"h": h, "metric": met, "v1_0": S.METRICS[met][0](d.y.values, d.pb.values), "v1_1": S.METRICS[met][0](d.y.values, d.pa.values),
                       "diff_better": obs, "p_better": p, "p_worse": pw, "n": len(d)})
Dd = pd.DataFrame(direct); Dd.to_csv(os.path.join(OUT, "direct_v1_1_vs_v1_0.csv"), index=False)
print(Dd.round(4).to_string())
res["alarm_unchanged"] = S.alarm_score(alarm, L)
json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
