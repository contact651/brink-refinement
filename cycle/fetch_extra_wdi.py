"""Refinement cycle 2026-10-26: extra World Bank series, as published in each archived edition (same rules as
brink_data.wdi_vintages). Used by idea tests only. Series: real government consumption growth (austerity idea)."""
import os, sys, requests, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from brink_data import UA, CLEAN
SER = {"NE.CON.GOVT.KD.ZG": "govcons_real_growth"}
vlist = []
for pg in (1, 2, 3):
    js = requests.get(f"https://api.worldbank.org/v2/sources/57/version?format=json&page={pg}", headers=UA, timeout=120).json()
    vlist += js["source"][0]["concept"][0]["variable"]
by_year = {}
for v in vlist:
    y = int(v["id"][:4])
    if 2009 <= y and (y not in by_year or v["id"] < by_year[y]):
        by_year[y] = v["id"]
extra = [v["id"] for v in vlist if int(v["id"][:4]) >= 2009 and v["id"][4:6] in ("07", "09", "10", "12")]
eds = sorted(set(by_year.values()) | set(extra))
rows = []
for ed in eds:
    for code, nm in SER.items():
        url = f"https://api.worldbank.org/v2/sources/57/country/all/series/{code}/version/{ed}/time/all?format=json&per_page=50000"
        try:
            js = requests.get(url, headers=UA, timeout=240).json()
            data = js["source"]["data"] if isinstance(js, dict) else []
        except Exception as e:
            print("fail", ed, e); continue
        for d in data:
            vv = {x["concept"]: x["id"] for x in d["variable"]}
            if d.get("value") is None: continue
            rows.append((vv.get("Country"), int(str(vv.get("Time", "YR0"))[2:] or 0), nm, d["value"], ed))
    print(ed, len(rows), flush=True)
df = pd.DataFrame(rows, columns=["iso3", "year", "indicator", "value", "edition"])
df.to_parquet(os.path.join(CLEAN, "wdi_vintages_extra.parquet"), index=False)
print("saved", len(df), df.edition.nunique(), "editions")
