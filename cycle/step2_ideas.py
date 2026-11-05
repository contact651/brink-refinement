#!/usr/bin/env python3
"""Cycle 1 step 2: test each pre-written idea on the practice years (2000-2014) against Working Model v1.0."""
import os, sys, json, pickle
import numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cycle"))
import brink_features as F, brink_benchmarks as BM
from brink_grok import H, LENSES
import practice as PR, signals as SG

P, L, T, Ys = pickle.load(open(os.path.join(PR.OUT, "panel.pkl"), "rb"))
isos = sorted(P.iso3.unique())
add = [SG.economic(isos), SG.austerity(isos), SG.type_memory(L), SG.speed(P)]
for a in add:
    a["ym"] = pd.PeriodIndex(a.ym, freq="M")
    P = P.merge(a, on=["iso3", "ym"], how="left")
br = BM.base_rate(P[["iso3", "ym"]], T)
for lens in LENSES:
    for h in H:
        P[f"br_{lens}_{h}"] = br[f"p_{lens}_{h}"].values
VARIANTS = {"v1.0": [], "I2_austerity": ["x_cut"], "I1_economic": ["x_infl", "x_gdpfall"],
            "I3_memory": ["x_since_C3", "x_since_C4", "x_since_C6", "x_since_C7"], "I4_own_record": ["x_own"],
            "I5_speed": ["x_speed"]}
pickle.dump(P, open(os.path.join(PR.OUT, "panel_plus.pkl"), "wb"))
cov = {c: float(P.loc[(P.ym >= PR.SCORE_FROM) & (P.ym <= PR.SCORE_TO), c].notna().mean()) for v in VARIANTS.values() for c in v if c != "x_own"}
print("coverage 2000-2014", cov, flush=True)

def run(k):
    f = PR.fit_predict(P, Ys, VARIANTS[k])
    f.to_parquet(os.path.join(PR.OUT, f"forecasts_{k}.parquet"))
    return k, f.attrs.get("coefs", {})

if __name__ == "__main__":
    with Pool(6) as pool:
        res = pool.map(run, list(VARIANTS))
    json.dump({k: c for k, c in res}, open(os.path.join(PR.OUT, "variant_coefs.json"), "w"), indent=1)
    base = pd.read_parquet(os.path.join(PR.OUT, "forecasts_v1.0.parquet"))
    allT, allR, verdicts = [], [], {}
    for k in VARIANTS:
        if k == "v1.0": continue
        f = pd.read_parquet(os.path.join(PR.OUT, f"forecasts_{k}.parquet"))
        Tb, R, v = PR.compare(f, base, Ys, k)
        allT.append(Tb); allR.append(R); verdicts[k] = v
        print(k, v, flush=True)
        print(Tb[["lens", "h", "metric", "model", "variant_score", "diff_better", "p_better", "p_worse", "sig_better", "sig_worse"]].round(4).to_string(), flush=True)
    pd.concat(allT).to_csv(os.path.join(PR.OUT, "practice_compare.csv"), index=False)
    pd.concat(allR).to_csv(os.path.join(PR.OUT, "practice_regions.csv"), index=False)
    json.dump(verdicts, open(os.path.join(PR.OUT, "verdicts.json"), "w"), indent=1)
