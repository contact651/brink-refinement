#!/usr/bin/env python3
"""Cycle 1 step 1: Working Model v1.0 on the practice years -> scores, miss list, false alarms, causes."""
import os, sys, json, pickle
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cycle"))
import brink_claude as BC
from brink_grok import targets, H
import practice as PR, misses as MS

O = os.path.join(ROOT, "data", "outcomes")
P, L, T = BC.prep([os.path.join(O, "outcomes_long_open.csv.gz")], [os.path.join(O, "targets_open.csv.gz")])
Ys = {h: P[["iso3", "ym"]].merge(targets(T, h), on=["iso3", "ym"], how="left") for h in H}
pickle.dump((P, L, T, Ys), open(os.path.join(PR.OUT, "panel.pkl"), "wb"))
base = PR.fit_predict(P, Ys, [])
base.to_parquet(os.path.join(PR.OUT, "forecasts_v1_0_practice.parquet"))
sc = PR.summary_scores(base, Ys)
json.dump(sc, open(os.path.join(PR.OUT, "scores_v1_0_practice.json"), "w"), indent=1)
thr = MS.thresholds(base, Ys, PR.joined)
ev = MS.miss_list(base, L, P, thr)
ev["cause"] = ev.apply(MS.cause, axis=1)
ev.to_csv(os.path.join(PR.OUT, "events_practice.csv"), index=False)
# false alarms: country-years where the 12-month chance stayed above the 70%-catch cut and nothing happened in the window
d = PR.joined(base, Ys, "GLOBAL", 12)
d["fa"] = (d.p >= thr[12]) & (d.y == 0)
fa = d.groupby("iso3").agg(false_alarm_months=("fa", "sum"), months=("fa", "size"), mean_p=("p", "mean")).reset_index()
fa = fa.sort_values("false_alarm_months", ascending=False)
fa.to_csv(os.path.join(PR.OUT, "false_alarms_practice.csv"), index=False)
print(json.dumps({"thresholds": thr, "scores": sc}, indent=1))
print(ev.status.value_counts())
print(ev.groupby(["ctype", "status"]).size().unstack(fill_value=0))
print(ev[ev.status != "seen in time"].groupby("cause").agg(n=("iso3", "size"), cost=("human_cost", "sum")).sort_values("cost", ascending=False))
print(ev[ev.status != "seen in time"].sort_values("human_cost", ascending=False).head(25)[["iso3", "ym", "ctype", "status", "p12", "p6", "p3", "human_cost", "cause"]].to_string())
print(fa.head(15).to_string())
