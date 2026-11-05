import os, sys, json, pickle
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cycle"))
import practice as PR, misses as MS
P, L, T, Ys = pickle.load(open(os.path.join(PR.OUT, "panel.pkl"), "rb"))
base = pd.read_parquet(os.path.join(PR.OUT, "forecasts_v1.0.parquet"))
i4 = pd.read_parquet(os.path.join(PR.OUT, "forecasts_I4_own_record.parquet"))
f = base.copy()
for c in [c for c in f.columns if c.startswith("p_GLOBAL_")]:
    f[c] = i4[c].values
assert (f.iso3.values == i4.iso3.values).all() and (f.ym.values == i4.ym.values).all()
f.to_parquet(os.path.join(PR.OUT, "forecasts_I4b_own_record_global.parquet"))
Tb, R, v = PR.compare(f, base, Ys, "I4b_own_record_global")
print(v); print(Tb[["lens","h","metric","model","variant_score","diff_better","p_better","p_worse","sig_better","sig_worse"]].round(4).to_string()); print(R.round(4).to_string())
Tb.to_csv(os.path.join(PR.OUT, "practice_compare_I4b.csv"), index=False); R.to_csv(os.path.join(PR.OUT, "practice_regions_I4b.csv"), index=False)
json.dump(v, open(os.path.join(PR.OUT, "verdict_I4b.json"), "w"))
# what changes in the miss list (same 70% cut rule, recomputed for the variant)
for nm, ff in [("v1.0", base), ("I4b", f)]:
    thr = MS.thresholds(ff, Ys, PR.joined); ev = MS.miss_list(ff, L, P, thr)
    ev.to_csv(os.path.join(PR.OUT, f"events_{nm}.csv"), index=False)
    d = PR.joined(ff, Ys, "GLOBAL", 12); fa = ((d.p >= thr[12]) & (d.y == 0)).sum(); neg = (d.y == 0).sum()
    print(nm, thr, ev.status.value_counts().to_dict(), "false alarms", int(fa), "of", int(neg))
    print(ev.groupby(["ctype", "status"]).size().unstack(fill_value=0))
