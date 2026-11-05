"""Fetch archived VIEWS country-month state-based forecasts (public API, api.viewsforecasting.org). Job 5 benchmark input."""
import json, os, sys, time, requests, pandas as pd
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "views")
os.makedirs(OUT, exist_ok=True)
runs = requests.get("https://api.viewsforecasting.org/", timeout=60).json()["runs"]
runs = [r for r in runs if r.startswith("fatalities") or r.startswith("r_2021")]
for run in runs:
    p = os.path.join(OUT, run + ".parquet")
    if os.path.exists(p):
        continue
    rows = []; url = f"https://api.viewsforecasting.org/{run}/cm/sb?page=1"; meta = {}
    while url:
        for k in range(3):
            try:
                d = requests.get(url, timeout=120).json(); break
            except Exception as e:
                time.sleep(5); d = None
        if d is None or "data" not in d:
            print("FAIL", run, url, flush=True); break
        meta = {"start_date": d.get("start_date"), "end_date": d.get("end_date")}
        rows += d["data"]; url = d.get("next_page") or None
    if rows:
        df = pd.DataFrame(rows); df["run"] = run; df["start_date"] = meta.get("start_date"); df["end_date"] = meta.get("end_date")
        df.to_parquet(p, index=False); print("OK", run, len(df), list(df.columns)[:14], flush=True)
