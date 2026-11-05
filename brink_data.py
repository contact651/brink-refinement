#!/usr/bin/env python3
"""
Brink Model — Round 1 build — JOB 1: data download and cleaning.

What it does (plain words)
  Downloads every public source the frozen exam (Exam Pack v2.2) and the three
  shortlisted designs (Grok, Claude, GPT Astra) need, keeps a copy of each raw
  file only in a local cache (never stored in the project), cleans them into
  tidy country tables keyed by ISO3, and writes:
    data/clean/*.parquet       tidy tables (re-built every run, not stored)
    data/clean/vintage_table.csv   when each piece of data became public (as-of rules)
    data/clean/source_status.csv   what downloaded, version, licence, rows, SHA-256
  Small tables (vintage table, source status) are copied to project docs.

Rules followed (Exam Pack v2.2, SHA-256 bd3d64f1...ebaec1)
  - No ACLED. EM-DAT not used (licence bars derived tables) -> public disaster lists.
  - Missing is missing (NaN), never zero.
  - Every value keeps its source, version and a release date or a Guess-labelled lag rule.
  - This script builds INPUT tables only. Outcome tables and the sealed 2015-2025
    split are built by brink_outcomes.py, which writes sealed outcomes to a separate
    file and prints only its hash.

Usage:  python3 brink_data.py [--vintages] [--only NAME[,NAME]]
"""
import argparse, hashlib, io, json, os, re, sys, time, zipfile, datetime as dt
import requests
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
CLEAN = os.path.join(ROOT, "data", "clean")
os.makedirs(RAW, exist_ok=True); os.makedirs(CLEAN, exist_ok=True)
UA = {"User-Agent": "BrinkModel-builder/1.0 (research; LONG MONEY project)"}
STATUS = []      # one row per source
VINTAGES = []    # one row per (source, version)


# ----------------------------------------------------------------- helpers
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def fetch(url, name, tries=3, timeout=300, **kw):
    """Download url into data/raw/name (cached). Returns local path or raises."""
    p = os.path.join(RAW, name)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout, stream=True, **kw)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            tmp = p + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            os.replace(tmp, p)
            return p
        except Exception as e:
            last = e; time.sleep(3 * (i + 1))
    raise RuntimeError(f"{url}: {last}")

def status(source, ok, version="", licence="", rows=0, url="", note="", sha=""):
    STATUS.append(dict(source=source, ok=ok, version=version, licence=licence, rows=rows,
                       url=url, sha256=sha, note=note,
                       retrieved_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")))
    print(f"[{'OK ' if ok else 'FAIL'}] {source} {version} rows={rows} {note}", flush=True)

def vintage(source, version, release_date, covers_to, rule, label):
    VINTAGES.append(dict(source=source, version=version, release_date=release_date,
                         covers_to=covers_to, rule=rule, label=label))

def save(df, name):
    p = os.path.join(CLEAN, name + ".parquet")
    df.to_parquet(p, index=False)
    return p

def run(name, fn, *a, **k):
    try:
        fn(*a, **k)
    except Exception as e:
        status(name, False, note=f"ERROR: {type(e).__name__}: {str(e)[:200]}")


# ----------------------------------------------------------------- country keys
# Master key = ISO3. UCDP uses Gleditsch-Ward (GW) numbers; V-Dem uses its own text ids.
GW_OVERRIDE = {  # GW codes country_converter does not map or maps differently (Likely, from GW list)
    260: "DEU", 255: "DEU", 265: "DDR", 345: "SRB", 347: "XKX", 340: "SRB", 678: "YEM", 680: "YMD",
    816: "VNM", 817: "RVN", 365: "RUS", 364: "RUS", 625: "SDN", 626: "SSD", 530: "ETH", 531: "ERI",
    315: "CZE", 316: "CZE", 317: "SVK", 711: "TIB", 713: "TWN", 731: "PRK", 732: "KOR",
    403: "STP", 591: "SYC", 835: "BRN", 860: "TLS", 950: "FJI", 990: "WSM", 101: "VEN",
}
_coco = None
def gw_to_iso3(gw):
    global _coco
    try:
        gw = int(gw)
    except Exception:
        return None
    if gw in GW_OVERRIDE:
        return GW_OVERRIDE[gw]
    if _coco is None:
        import country_converter as coco
        d = coco.CountryConverter().data
        _coco = {}
        for _, r in d.iterrows():
            for c in str(r.get("GWcode", "")).split(";"):
                c = c.strip()
                if c and c.replace(".", "").isdigit():
                    _coco.setdefault(int(float(c)), r["ISO3"])
    return _coco.get(gw)

def name_to_iso3(names):
    import country_converter as coco
    return coco.convert(list(names), to="ISO3", not_found=None)


# ----------------------------------------------------------------- 1. UCDP
GED_VERSIONS = {  # version: (file token, release date, covers-to, label)
    "26.1": ("ged261", "2026-06-01", "2025-12", "Guess: UCDP releases yy.1 in June"),
    "25.1": ("ged251", "2025-06-01", "2024-12", "Guess"), "24.1": ("ged241", "2024-06-01", "2023-12", "Guess"),
    "23.1": ("ged231", "2023-06-01", "2022-12", "Guess"), "22.1": ("ged221", "2022-06-01", "2021-12", "Guess"),
    "21.1": ("ged211", "2021-06-01", "2020-12", "Guess"), "20.1": ("ged201", "2020-06-01", "2019-12", "Guess"),
    "19.1": ("ged191", "2019-06-01", "2018-12", "Guess"), "18.1": ("ged181", "2018-06-01", "2017-12", "Guess"),
    "17.1": ("ged171", "2017-06-01", "2016-12", "Guess"), "5.0": ("ged50", "2016-06-01", "2015-12", "Guess"),
    "4.0": ("ged40", "2015-06-01", "2014-12", "Guess"),
}

def _read_zip_csv(p):
    with zipfile.ZipFile(p) as z:
        n = [x for x in z.namelist() if x.lower().endswith(".csv")][0]
        with z.open(n) as f:
            return pd.read_csv(f, low_memory=False, encoding_errors="replace")

def ged_country_month(df, version):
    """Event file -> country-month deaths by violence type (1 state-based, 2 non-state, 3 one-sided)."""
    cols = {c.lower(): c for c in df.columns}
    g = lambda k: df[cols[k]] if k in cols else np.nan
    out = pd.DataFrame({
        "gw": g("country_id"),
        "date_start": pd.to_datetime(g("date_start"), errors="coerce"),
        "date_end": pd.to_datetime(g("date_end"), errors="coerce"),
        "tv": g("type_of_violence"),
        "best": pd.to_numeric(g("best"), errors="coerce") if "best" in cols else pd.to_numeric(g("best_est"), errors="coerce"),
        "civ": pd.to_numeric(g("deaths_civilians"), errors="coerce"),
        "prec": pd.to_numeric(g("date_prec"), errors="coerce"),
        "lat": pd.to_numeric(g("latitude"), errors="coerce"), "lon": pd.to_numeric(g("longitude"), errors="coerce"),
    })
    # Event month = month of date_start (Guess: events spanning months are booked to their start month)
    out["year"] = out.date_start.dt.year; out["month"] = out.date_start.dt.month
    if out.gw.isna().all() and "gwno" in cols:          # GED 5.0 uses gwno
        out["gw"] = g("gwno")
    out["iso3"] = out.gw.map(gw_to_iso3)
    if out.iso3.isna().any() and "country" in cols:      # GED 4.0 has only country names
        miss = out.iso3.isna()
        names = df.loc[miss, cols["country"]].astype(str).str.replace(r"\s*\(.*\)", "", regex=True)
        uniq = names.unique().tolist()
        conv = dict(zip(uniq, name_to_iso3(uniq) if uniq else []))
        out.loc[miss, "iso3"] = names.map(conv)
    cm = (out.groupby(["iso3", "gw", "year", "month", "tv"], dropna=False)
             .agg(deaths_best=("best", "sum"), deaths_civilians=("civ", "sum"), events=("best", "size"))
             .reset_index())
    cm["ged_version"] = version
    ev = out[["iso3", "year", "month", "tv", "best", "lat", "lon"]].copy()
    ev["ged_version"] = version
    return cm, ev

def ucdp_ged(vintages=False):
    vers = list(GED_VERSIONS) if vintages else ["26.1"]
    for v in vers:
        tok, rel, cov, lab = GED_VERSIONS[v]
        url = f"https://ucdp.uu.se/downloads/ged/{tok}-csv.zip"
        p = fetch(url, f"{tok}-csv.zip")
        df = _read_zip_csv(p)
        cm, ev = ged_country_month(df, v)
        save(cm, f"ucdp_ged_cm_{v.replace('.', '_')}")
        if v == "26.1":
            save(ev, "ucdp_ged_events_26_1")   # event points for distance-weighted neighbour violence (Claude S2)
        unmapped = cm[cm.iso3.isna()].gw.unique().tolist()
        status("UCDP GED", True, v, "CC BY 4.0 (UCDP download page; Zenodo copy of 25.1 says ODbL 1.0 - licence locked per file, see note)",
               len(df), url, f"events={len(df)}; unmapped GW={unmapped[:10]}", sha256_file(p))
        vintage("UCDP GED", v, rel, cov, "final yearly release; usable from release date", lab)

def ucdp_candidate():
    """Monthly Candidate releases (2018 on): fill the months after the latest final GED release."""
    html = requests.get("https://ucdp.uu.se/downloads/olddw.html", headers=UA, timeout=120).text
    html += requests.get("https://ucdp.uu.se/downloads/", headers=UA, timeout=120).text
    files = sorted(set(re.findall(r"candidateged/(GEDEvent_v(\d+)_0_(\d+)\.csv)", html)))
    frames = []
    for fn, yy, k in files:
        yy, k = int(yy), int(k)
        try:
            p = fetch(f"https://ucdp.uu.se/downloads/candidateged/{fn}", fn)
            df = pd.read_csv(p, low_memory=False, encoding_errors="replace")
            cm, _ = ged_country_month(df, f"cand_{yy}.0.{k}")
        except Exception as e:
            status("UCDP Candidate", False, f"{yy}.0.{k}", note=f"{fn}: {str(e)[:120]}"); continue
        # Guess lag rule: release k of year 20yy covers events through month k-1 (earlier: through k),
        # and is public at the end of month k. We keep the file's own max event month as 'covers_to'.
        cov = df.filter(regex="(?i)^date_start$").iloc[:, 0]
        cov = pd.to_datetime(cov, errors="coerce").max()
        rel = (pd.Timestamp(year=2000 + yy, month=k, day=1) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        cm["release_date"] = rel
        frames.append(cm)
        vintage("UCDP Candidate", f"{yy}.0.{k}", rel, str(cov)[:7], "monthly release; Guess: public at end of release month k", "Guess")
    if frames:
        allc = pd.concat(frames, ignore_index=True)
        save(allc, "ucdp_candidate_cm_all_releases")
        status("UCDP Candidate", True, f"{len(frames)} monthly releases", "CC BY 4.0 (UCDP download page)", len(allc),
               "https://ucdp.uu.se/downloads/olddw.html")

def ucdp_small():
    for nm, url, fname in [
        ("UCDP/PRIO Armed Conflict", "https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-261-csv.zip", "acd261.zip"),
        ("UCDP One-sided", "https://ucdp.uu.se/downloads/nsos/ucdp-onesided-261-csv.zip", "os261.zip"),
        ("UCDP Non-state", "https://ucdp.uu.se/downloads/nsos/ucdp-nonstate-261-csv.zip", "ns261.zip"),
        ("UCDP BRD conflict", "https://ucdp.uu.se/downloads/brd/ucdp-brd-conf-261-csv.zip", "brd261.zip"),
    ]:
        p = fetch(url, fname); df = _read_zip_csv(p)
        save(df.astype({c: str for c in df.columns if df[c].dtype == object}), "ucdp_" + re.sub(r"\W+", "_", nm.lower()))
        status(nm, True, "26.1", "CC BY 4.0 (UCDP download page)", len(df), url, sha=sha256_file(p))
        vintage(nm, "26.1", "2026-06-01", "2025-12", "yearly; Guess lag rule for earlier years: year y usable 1 June y+1", "Guess")


# ----------------------------------------------------------------- 2. V-Dem
VDEM_VARS = ["country_name", "country_text_id", "COWcode", "year",
             "v2x_polyarchy", "v2x_libdem", "v2x_civlib", "v2x_clphy", "v2x_clpol", "v2x_clpriv",
             "v2x_freexp_altinf", "v2x_regime", "v2x_regime_amb", "v2cacamps", "v2mecenefm", "v2meharjrn",
             "v2regendtype", "v2reginfo", "v2x_rule", "v2xcs_ccsi", "v2x_corr", "v2clkill", "v2cltort", "v2clsocgrp",
             "e_pop", "e_gdppc", "e_miinflat", "e_civil_war", "e_miinteco", "e_miinterc"]
VDEM_RELEASE = {"16": ("2026-03-01", "Confirmed: v16 published March 2026 (Grok entry; V-Dem release statement)"),
                "15": ("2025-03-01", "Likely: V-Dem releases each March"), "14": ("2024-03-01", "Likely"),
                "13": ("2023-03-01", "Likely"), "12": ("2022-03-01", "Likely"), "11.1": ("2021-03-01", "Likely"),
                "10": ("2020-03-01", "Likely"), "9": ("2019-04-01", "Likely"), "8": ("2018-04-01", "Likely")}

def _vdem_form_link(path):
    """V-Dem serves files via a download form; no personal data is required (only the file format)."""
    s = requests.Session(); u = "https://www.v-dem.net" + path
    h = s.get(u, headers=UA, timeout=60).text
    tok = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', h).group(1)
    opts = re.findall(r'name="dataset_file" value="(\d+)"[^>]*>\s*([^<]+)<', h)
    csv = [v for v, n in opts if n.strip().upper() == "CSV"]
    r = s.post(u, data={"csrfmiddlewaretoken": tok, "dataset_file": csv[0]}, headers={**UA, "Referer": u}, timeout=60)
    m = re.findall(r'https?://[^"]+/media/[^"]+\.zip', r.text)
    return m[0] if m else None

def vdem(vintages=False):
    vers = ["16"] + (["15", "14", "13", "12", "11.1", "10", "9", "8"] if vintages else [])
    for v in vers:
        tagv = v.replace(".", "")
        paths = [f"/data/the-v-dem-dataset/country-year-v-dem-fullothers-v{tagv}/",
                 f"/data/dataset-archive/country-year-v-dem-fullothers-v{tagv}/",
                 f"/data/the-v-dem-dataset/country-year-v-dem-core-v{tagv}/",
                 f"/data/dataset-archive/country-year-v-dem-core-v{tagv}/"]
        frames, used = [], []
        for path in paths:
            try:
                link = _vdem_form_link(path)
            except Exception:
                link = None
            if not link:
                continue
            p = fetch(link, f"vdem_{tagv}_{os.path.basename(link)}")
            with zipfile.ZipFile(p) as z:
                n = [x for x in z.namelist() if x.lower().endswith(".csv")][0]
                with z.open(n) as f:
                    hdr = pd.read_csv(f, nrows=0).columns
                with z.open(n) as f:
                    df = pd.read_csv(f, usecols=[c for c in VDEM_VARS if c in hdr], low_memory=False)
            frames.append(df); used.append(link)
            if "fullothers" in path:
                pass
        if not frames:
            status("V-Dem", False, "v" + v, note="no download link resolved"); continue
        df = frames[0]
        for extra in frames[1:]:
            add = [c for c in extra.columns if c not in df.columns]
            if add:
                df = df.merge(extra[["country_text_id", "year"] + add], on=["country_text_id", "year"], how="left")
        df = df[df.year >= 1900].copy()
        df["iso3"] = df.country_text_id
        df["vdem_version"] = v
        save(df, f"vdem_cy_v{tagv}")
        rel, lab = VDEM_RELEASE.get(v, ("", "Guess"))
        status("V-Dem", True, "v" + v, "CC BY-SA 4.0 (derivatives keep share-alike)", len(df), " ; ".join(used))
        vintage("V-Dem", "v" + v, rel, f"{int(df.year.max())}-12", "annual; year y usable only from the version released after it", lab)


# ----------------------------------------------------------------- 3. World Bank WDI (current + archived vintages)
WDI = {"NY.GDP.PCAP.KD.ZG": "gdp_pc_growth", "SP.DYN.IMRT.IN": "infant_mortality", "SP.POP.TOTL": "population",
       "SP.DYN.LE00.IN": "life_expectancy", "FI.RES.TOTL.MO": "reserves_months_imports",
       "FP.CPI.TOTL.ZG": "inflation_cpi", "PA.NUS.FCRF": "fx_lcu_per_usd"}

def wdi_current():
    rows = []
    for code, nm in WDI.items():
        url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=20000&date=1960:2025"
        js = requests.get(url, headers=UA, timeout=180).json()
        if "pages" not in js[0]:
            status("World Bank WDI (current)", False, code, note=str(js[0])[:150]); continue
        pages = js[0]["pages"]; data = js[1] or []
        for pg in range(2, pages + 1):
            data += requests.get(url + f"&page={pg}", headers=UA, timeout=180).json()[1] or []
        for d in data:
            rows.append((d["countryiso3code"], int(d["date"]), nm, d["value"]))
    df = pd.DataFrame(rows, columns=["iso3", "year", "indicator", "value"])
    df = df[df.iso3.str.len() == 3]
    save(df, "wdi_current")
    status("World Bank WDI (current)", True, "API v2, retrieved today", "CC BY 4.0 (WDI catalogue; World Bank terms of use)", len(df),
           "https://api.worldbank.org/v2/")
    vintage("World Bank WDI", "current", dt.date.today().isoformat(), "2025", "current edition; use only for years with no archived vintage, with lag rule: year y usable 1 July y+1 (Guess)", "Guess")

def wdi_vintages():
    """WDI Database Archives (source 57): the values as published in each edition."""
    vlist = []
    for pg in (1, 2, 3):
        js = requests.get(f"https://api.worldbank.org/v2/sources/57/version?format=json&page={pg}", headers=UA, timeout=120).json()
        vlist += js["source"][0]["concept"][0]["variable"]
    # one edition per year (the first of each year from 2009), enough for yearly as-of lookups
    by_year = {}
    for v in vlist:
        y = int(v["id"][:4])
        if 2009 <= y and (y not in by_year or v["id"] < by_year[y]):
            by_year[y] = v["id"]
    # plus the mid-year editions (Jul/Sep/Oct) to tighten the lag
    extra = [v["id"] for v in vlist if int(v["id"][:4]) >= 2009 and v["id"][4:6] in ("07", "09", "10", "12")]
    eds = sorted(set(by_year.values()) | set(extra))
    rows = []
    for ed in eds:
        for code, nm in WDI.items():
            url = f"https://api.worldbank.org/v2/sources/57/country/all/series/{code}/version/{ed}/time/all?format=json&per_page=50000"
            try:
                js = requests.get(url, headers=UA, timeout=240).json()
                data = js["source"]["data"] if isinstance(js, dict) else []
                pages = js.get("pages", 1) if isinstance(js, dict) else 1
                for pg in range(2, (pages or 1) + 1):
                    data += requests.get(url + f"&page={pg}", headers=UA, timeout=240).json()["source"]["data"]
            except Exception as e:
                status("WDI archive", False, ed, note=f"{code}: {str(e)[:80]}"); continue
            for d in data:
                vv = {x["concept"]: x["id"] for x in d["variable"]}
                if d.get("value") is None:
                    continue
                rows.append((vv.get("Country"), int(str(vv.get("Time", "YR0"))[2:] or 0), nm, d["value"], ed))
        vintage("World Bank WDI archive", ed, f"{ed[:4]}-{ed[4:6]}-01", "", "edition as published; usable from first day of edition month", "Likely")
    df = pd.DataFrame(rows, columns=["iso3", "year", "indicator", "value", "edition"])
    save(df, "wdi_vintages")
    status("World Bank WDI archives", True, f"{len(eds)} editions {eds[0]}..{eds[-1]}", "World Bank terms of use (free reuse, attribution)", len(df),
           "https://api.worldbank.org/v2/sources/57")


# ----------------------------------------------------------------- 4. UNHCR refugees by origin
def unhcr():
    rows = []; page = 1
    while True:
        url = f"https://api.unhcr.org/population/v1/population/?limit=1000&page={page}&yearFrom=1951&yearTo=2025&coo_all=true"
        js = requests.get(url, headers=UA, timeout=120).json()
        rows += js["items"]
        if page >= js.get("maxPages", 1):
            break
        page += 1
    df = pd.DataFrame(rows)
    num = lambda s: pd.to_numeric(s.replace({"-": np.nan}), errors="coerce")
    out = pd.DataFrame({"iso3": df.coo_iso, "year": df.year.astype(int), "refugees": num(df.refugees),
                        "asylum_seekers": num(df.asylum_seekers), "idps": num(df.idps)})
    save(out, "unhcr_by_origin")
    status("UNHCR Refugee Data", True, "API v1", "CC BY 4.0 (UNHCR terms effective 1 July 2026, per Grok entry: check)", len(out),
           "https://api.unhcr.org/population/v1/population/")
    vintage("UNHCR", "API current", dt.date.today().isoformat(), "2025", "year y usable 1 July y+1 (Guess: Global Trends is published mid-year); no vintages", "Guess")


# ----------------------------------------------------------------- 5. IMF monthly prices and exchange rates
def _imf_csv(flow, key, start="1990"):
    url = f"https://api.imf.org/external/sdmx/2.1/data/IMF.STA,{flow}/{key}?startPeriod={start}&detail=dataonly"
    r = requests.get(url, headers={**UA, "Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}, timeout=600)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), low_memory=False)

def imf():
    cpi = _imf_csv("CPI", ".CPI._T.YOY_PCH_PA_PT.M")
    cpi = pd.DataFrame({"iso3": cpi.COUNTRY, "period": cpi.TIME_PERIOD, "cpi_yoy": pd.to_numeric(cpi.OBS_VALUE, errors="coerce")})
    save(cpi, "imf_cpi_yoy_m")
    status("IMF CPI (monthly YoY)", True, "IMF.STA:CPI current", "IMF copyright; free use with attribution under IMF terms (check)", len(cpi),
           "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,CPI")
    vintage("IMF CPI", "current", dt.date.today().isoformat(), "", "month m usable at end of month m+2 (Guess lag rule; IMF vintages only from 2026)", "Guess")
    eer = _imf_csv("EER", "..M")
    eer = pd.DataFrame({"iso3": eer.COUNTRY, "indicator": eer.INDICATOR, "period": eer.TIME_PERIOD,
                        "value": pd.to_numeric(eer.OBS_VALUE, errors="coerce")})
    eer = eer[eer.indicator.str.startswith("NEER")]
    save(eer, "imf_neer_m")
    status("IMF effective exchange rates (NEER, monthly)", True, "IMF.STA:EER current", "IMF terms (check)", len(eer),
           "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,EER", note=f"countries={eer.iso3.nunique()} (about 90; others fall back to the USD rate, Guess)")
    vintage("IMF EER", "current", dt.date.today().isoformat(), "", "month m usable at end of month m+2 (Guess)", "Guess")


# ----------------------------------------------------------------- 6. Sovereign defaults (BoC-BoE)
def boc_boe():
    got = []
    for ed in (2026, 2025, 2024, 2023, 2022, 2021):
        for mo in ("10", "07", "06", "09", "08", "11"):
            url = f"https://www.bankofcanada.ca/wp-content/uploads/{ed}/{mo}/BoC-BoE-Database-{ed}.xlsx"
            try:
                p = fetch(url, f"BoC-BoE-Database-{ed}.xlsx", tries=1, timeout=60)
            except Exception:
                continue
            x = pd.read_excel(p, sheet_name=None)
            for sh, df in x.items():
                df = df.copy(); df.insert(0, "sheet", sh); df.insert(0, "edition", ed)
                save(df.astype(str), f"bocboe_{ed}_{re.sub(r'W+', '_', sh)[:20]}".replace(" ", "_"))
            got.append(ed)
            status("BoC-BoE sovereign default database", True, str(ed), "Bank of Canada terms (free reuse with attribution; check)",
                   sum(len(d) for d in x.values()), url, f"sheets={list(x)}", sha256_file(p))
            vintage("BoC-BoE defaults", str(ed), f"{ed}-{mo}-01", f"{ed-1}-12", "edition usable from publication month", "Likely")
            break
    if not got:
        status("BoC-BoE sovereign default database", False, note="no edition found")


# ----------------------------------------------------------------- 7. State breakdown, coups, mass killing (PITF, CSP)
def pitf_csp():
    base = "https://www.systemicpeace.org/inscr/"
    for nm, f in [("PITF Adverse Regime Change", "PITF%20Adverse%20Regime%20Change%202018.xls"),
                  ("PITF Revolutionary War", "PITF%20Revolutionary%20War%202018.xls"),
                  ("PITF Ethnic War", "PITF%20Ethnic%20War%202018.xls"),
                  ("PITF Genocide/Politicide", "PITF%20GenoPoliticide%202018.xls"),
                  ("CSP Coups list", "CSPCoupsListv2021.xls"),
                  ("CSP Coups annual", "CSPCoupsAnnualv2021.xls")]:
        p = fetch(base + f, f.replace("%20", "_"))
        df = pd.read_excel(p)
        save(df.astype(str), re.sub(r"\W+", "_", nm.lower()))
        lic = "Center for Systemic Peace: free for research use with citation; no open redistribution licence stated (check)"
        status(nm, True, "2018" if "PITF" in nm else "v2021", lic, len(df), base + f, sha=sha256_file(p))
        vintage(nm, "2018" if "PITF" in nm else "v2021", "", "2018-12" if "PITF" in nm else "2021-12",
                "outcome source; events usable 1 month after they happen when used as a feature (Guess)", "Guess")


# ----------------------------------------------------------------- 8. Mass unrest (C7): Mass Mobilization Project (CC0)
def mmp():
    url = "https://dataverse.harvard.edu/api/access/datafile/4291456"
    p = fetch(url, "mmALL_073120.tab")
    df = pd.read_csv(p, sep="\t", low_memory=False, encoding_errors="replace")
    df["iso3"] = name_to_iso3(df.country.fillna(""))
    save(df.astype({c: str for c in df.columns if df[c].dtype == object}), "mmp_protests")
    status("Mass Mobilization Project", True, "mmALL_073120 (to Mar 2020)", "CC0 1.0", len(df), url,
           "Covers 1990 to Mar 2020, 162 countries; NO injury counts (state response only) -> C7 is a Guess proxy; Apr 2020-2025 has no ACLED-free source yet", sha256_file(p))
    vintage("Mass Mobilization", "2020-07-31", "2020-07-31", "2020-03", "single release; year y usable 1 July y+1 (Guess)", "Guess")


# ----------------------------------------------------------------- 9. Disasters (C5) without EM-DAT
def disasters():
    rows = []
    for haz in ("earthquakes", "tsunamis/events", "volcanoes"):
        url = f"https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/{haz}?minYear=1946&maxYear=2026"
        js = requests.get(url, headers=UA, timeout=180).json()
        items = js.get("items", []); pages = js.get("totalPages", 1)
        for pg in range(2, pages + 1):
            items += requests.get(url + f"&page={pg}", headers=UA, timeout=180).json().get("items", [])
        for it in items:
            rows.append(dict(hazard=haz.split("/")[0], year=it.get("year"), month=it.get("month"), country=it.get("country"),
                             deaths=it.get("deathsTotal", it.get("deaths")), injuries=it.get("injuriesTotal", it.get("injuries")),
                             source="NOAA NCEI"))
    status("NOAA NCEI hazards (earthquake, tsunami, volcano)", True, "API v1", "Public domain (US Government work)", len(rows),
           "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/")
    vintage("NOAA NCEI hazards", "current", dt.date.today().isoformat(), "", "event usable from the month after it happens (Guess)", "Guess")
    # Floods: Dartmouth Flood Observatory archive via its public ArcGIS feature service
    try:
        fl = []; off = 0
        base = ("https://services1.arcgis.com/AXaYBvnJsB5Q7sDF/ArcGIS/rest/services/Global_Archive_of_large_flood_events/"
                "FeatureServer/0/query?where=1%3D1&outFields=*&returnGeometry=false&f=json&resultRecordCount=2000&resultOffset=")
        while True:
            js = requests.get(base + str(off), headers=UA, timeout=180).json()
            fs = js.get("features", [])
            fl += [f["attributes"] for f in fs]
            if not js.get("exceededTransferLimit") or not fs:
                break
            off += len(fs)
        fdf = pd.DataFrame(fl)
        dcol = [c for c in fdf.columns if c.lower() in ("dead", "deaths")]
        bcol = [c for c in fdf.columns if c.lower() in ("began", "began_date", "start")]
        ccol = [c for c in fdf.columns if c.lower() in ("country", "countries")]
        if dcol and bcol and ccol:
            began = pd.to_datetime(fdf[bcol[0]], unit="ms", errors="coerce") if np.issubdtype(fdf[bcol[0]].dtype, np.number) \
                else pd.to_datetime(fdf[bcol[0]], errors="coerce")
            for c, d, b, dis in zip(fdf[ccol[0]], fdf[dcol[0]], began, fdf.get("Displaced", pd.Series([np.nan] * len(fdf)))):
                rows.append(dict(hazard="flood", year=b.year if pd.notna(b) else None, month=b.month if pd.notna(b) else None,
                                 country=c, deaths=d, injuries=np.nan, displaced=dis, source="DFO"))
        status("Dartmouth Flood Observatory (ArcGIS copy)", True, "feature service", "Free for research with citation (DFO terms; check)",
               len(fdf), base[:120], f"columns={list(fdf.columns)[:15]}; max began={began.max() if dcol else ''}")
        vintage("DFO floods", "current", dt.date.today().isoformat(), "", "event usable from the month after it began (Guess)", "Guess")
    except Exception as e:
        status("Dartmouth Flood Observatory (ArcGIS copy)", False, note=str(e)[:150])
    df = pd.DataFrame(rows)
    df["iso3"] = name_to_iso3(df.country.fillna("").astype(str).str.split(",").str[0].str.title())
    # A worldwide pandemic counts as ONE global event (Exam Pack s.2). Date is a Guess: WHO pandemic characterisation, Mar 2020.
    df = pd.concat([df, pd.DataFrame([dict(hazard="pandemic", year=2020, month=3, country="GLOBAL", iso3="WLD", deaths=np.nan,
                                           source="Exam Pack rule: one global event (date Guess)")])], ignore_index=True)
    for c in ("deaths", "injuries", "year", "month"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    save(df.astype({c: str for c in df.columns if df[c].dtype == object}), "disasters_public")


# ----------------------------------------------------------------- 10. Early Warning Project published scores
def ewp():
    got = []
    for y in range(2014, 2027):
        url = f"https://earlywarningproject.ushmm.org/api/sra/download/csv/{y}-sra"
        try:
            p = fetch(url, f"ewp_{y}_sra.csv", tries=1, timeout=60)
        except Exception:
            continue
        df = pd.read_csv(p); df.insert(0, "sra_year", y)
        save(df.astype({c: str for c in df.columns if df[c].dtype == object}), f"ewp_sra_{y}")
        got.append(y)
        vintage("Early Warning Project SRA", str(y), f"{y-1}-12-01", f"{y-2}-12",
                "assessment for year Y published late Y-1 using data to end Y-2 (Confirmed for 2025-26; Guess for others)", "Guess")
    status("Early Warning Project SRA", bool(got), f"{got[0] if got else ''}..{got[-1] if got else ''}",
           "Public; redistribution licence not established (Astra entry) -> scores used, not republished", len(got),
           "https://earlywarningproject.ushmm.org/api/sra/download/csv/",
           "2014-2017 assessments are on GitHub repos only (github.com API blocked from this cloud; raw files reachable) -> job 5")


# ----------------------------------------------------------------- 11. Borders (Grok 'bad neighbourhood')
def borders():
    url = "https://raw.githubusercontent.com/geodatasource/country-borders/master/GEODATASOURCE-COUNTRY-BORDERS.CSV"
    p = fetch(url, "country_borders.csv")
    df = pd.read_csv(p, keep_default_na=False)
    df.columns = ["iso2", "name", "border_iso2", "border_name"][:len(df.columns)]
    import country_converter as coco
    cc = coco.CountryConverter()
    df["iso3"] = cc.convert(df.iso2.tolist(), src="ISO2", to="ISO3", not_found=None)
    df["border_iso3"] = cc.convert(df.border_iso2.replace("", "XX").tolist(), src="ISO2", to="ISO3", not_found=None)
    save(df, "borders")
    status("Country borders (GeoDataSource)", True, "GitHub master", "MIT licence per repository (Likely: check LICENSE file)", len(df), url,
           "today's borders applied to all years (Guess; historical borders not modelled)", sha256_file(p))


# ----------------------------------------------------------------- 12. Food prices (Astra 'necessities'; Claude candidate C1)
def wfp():
    js = requests.get("https://data.humdata.org/api/3/action/package_show?id=global-wfp-food-prices", headers=UA, timeout=120).json()["result"]
    res = [r for r in js["resources"] if re.search(r"wfp_food_prices_global_\d{4}\.csv", r["url"])]
    parts = []
    for r in res:
        yr = re.search(r"(\d{4})\.csv", r["url"]).group(1)
        p = fetch(r["url"], f"wfp_{yr}.csv", timeout=900)
        with open(p, encoding="utf-8", errors="replace") as fh:
            fh.readline(); second = fh.readline()
        df = pd.read_csv(p, low_memory=False, skiprows=[1] if second.startswith("#") else None)
        cols = {c.lower(): c for c in df.columns}
        need = ["countryiso3", "date", "market_id", "commodity_id", "unit", "pricetype", "currency", "price"]
        if not all(k in cols for k in need):
            status("WFP food prices", False, yr, note=f"columns {list(df.columns)[:15]}"); continue
        d = df[[cols[k] for k in need]].copy(); d.columns = need
        d["price"] = pd.to_numeric(d.price, errors="coerce")
        d["date"] = pd.to_datetime(d.date, errors="coerce").dt.to_period("M")
        parts.append(d[d.pricetype.astype(str).str.lower().eq("retail")])
    d = pd.concat(parts, ignore_index=True)
    key = ["countryiso3", "market_id", "commodity_id", "unit", "currency"]
    d = d.groupby(key + ["date"], as_index=False).price.median()
    d = d[d.price > 0]
    d["lp"] = np.log(d.price)
    lag = d[key + ["date", "lp"]].copy(); lag["date"] = lag.date + 12
    d = d.merge(lag.rename(columns={"lp": "lp12"}), on=key + ["date"], how="left")
    d["yoy"] = d.lp - d.lp12
    cm = (d.dropna(subset=["yoy"]).groupby(["countryiso3", "date"])
            .agg(food_yoy_median=("yoy", "median"), n_series=("yoy", "size")).reset_index())
    cm["year"] = cm.date.dt.year; cm["month"] = cm.date.dt.month
    cm = cm.rename(columns={"countryiso3": "iso3"}).drop(columns="date")
    save(cm, "wfp_food_yoy_cm")
    status("WFP food prices (HDX)", True, f"{len(res)} yearly files", "CC BY-IGO (HDX)", len(cm),
           "https://data.humdata.org/dataset/global-wfp-food-prices",
           "country-month median of year-on-year log price change across matched market-commodity-unit retail series (Astra spec)")
    vintage("WFP food prices", "current", dt.date.today().isoformat(), "", "month m usable end of month m+1 (Guess; no vintages, updated weekly)", "Guess")


# ----------------------------------------------------------------- 13. Weather (Astra) - open replacement for ERA5
def cru():
    base = "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/crucy.2503061057.v4.09/countries/"
    rows = []
    for var in ("pre", "tmx"):
        idx = requests.get(base + var + "/", headers=UA, timeout=120).text
        files = sorted(set(re.findall(r'href="(crucy\.[^"]+\.per)"', idx)))
        for fn in files:
            try:
                p = fetch(base + var + "/" + fn, "cru_" + fn, tries=2, timeout=60)
                t = pd.read_fwf(p, skiprows=3, widths=[5] + [8] * 17)
            except Exception as e:
                continue
            parts = fn.split(".")          # crucy.v4.09.1901.2024.<Country>.<var>.per (fixed 22 Oct: was index 4 = the end year)
            country = parts[-3].replace("_", " ")
            t = t.iloc[:, :13]; t.columns = ["year"] + list(range(1, 13))
            t = t.melt(id_vars="year", var_name="month", value_name="value")
            t["value"] = pd.to_numeric(t.value, errors="coerce").replace(-999.0, np.nan)
            t["country"] = country; t["var"] = var
            rows.append(t)
    df = pd.concat(rows, ignore_index=True)
    names = df.country.unique().tolist()
    conv = dict(zip(names, name_to_iso3(names)))
    df["iso3"] = df.country.map(conv)
    df = df[df.year >= 1946]
    save(df, "cru_country_monthly")
    status("CRU TS 4.09 country averages (weather)", True, "4.09 (to 2024)", "Open Government Licence v3", len(df), base,
           "Open replacement for ERA5 (needs an account). Guess. 2025 not yet released (CRU TS 4.10 expected 2026) -> 2025 months missing")
    for v, rel, cov in [("4.09", "2025-03-06", "2024-12")]:
        vintage("CRU TS country", v, rel, cov, "yearly release; earlier years use lag rule: year y usable 1 April y+2 (Guess, matches release cadence)", "Guess")


# ----------------------------------------------------------------- main
BLOCKED = [
    ("ACLED", "Banned by Exam Pack s.9 (licence bars benchmarking/redistribution)."),
    ("EM-DAT", "Not used: terms bar redistribution and derivative databases. C5 uses NOAA NCEI + DFO floods + one global pandemic row. Storms, droughts, heat, epidemics other than COVID are a known gap (never read as calm)."),
    ("Powell & Thyne coups", "BLOCKED from this cloud: www.uky.edu does not answer (TLS times out after proxy connect). Replacement used: CSP Coups list v2021 (to 2021) + V-Dem regime-end type (to 2025)."),
    ("Laeven & Valencia banking crises", "BLOCKED from this cloud: imf.org returns 403 for the data file. C4 banking-crisis arm missing; C4 still uses defaults (BoC-BoE), currency crash (IMF NEER) and inflation >=25% (IMF CPI / WDI)."),
    ("ERA5 weather (Astra)", "Needs a Copernicus account/API key. Replaced by CRU TS 4.09 country averages (Open Government Licence), Guess replacement; country-level only."),
    ("VIEWS archive, prediction markets", "Benchmarks: fetched in job 5."),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintages", action="store_true", help="also fetch every archived vintage (GED, V-Dem, WDI)")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    jobs = [("ucdp_ged", lambda: ucdp_ged(a.vintages)), ("ucdp_candidate", ucdp_candidate), ("ucdp_small", ucdp_small),
            ("vdem", lambda: vdem(a.vintages)), ("wdi_current", wdi_current), ("wdi_vintages", wdi_vintages),
            ("unhcr", unhcr), ("imf", imf), ("boc_boe", boc_boe), ("pitf_csp", pitf_csp), ("mmp", mmp),
            ("disasters", disasters), ("ewp", ewp), ("borders", borders), ("wfp", wfp), ("cru", cru)]
    only = set(filter(None, a.only.split(",")))
    for nm, fn in jobs:
        if only and nm not in only:
            continue
        print(f"--- {nm}", flush=True)
        run(nm, fn)
    if not only:
        for s, why in BLOCKED:
            status(s, False, note=why)
    st = pd.DataFrame(STATUS)
    mode = "a" if only and os.path.exists(os.path.join(CLEAN, "source_status.csv")) else "w"
    st.to_csv(os.path.join(CLEAN, "source_status.csv"), index=False, mode=mode, header=(mode == "w"))
    vt = pd.DataFrame(VINTAGES)
    vt.to_csv(os.path.join(CLEAN, "vintage_table.csv"), index=False, mode=mode, header=(mode == "w"))
    print("done", len(st), "status rows,", len(vt), "vintage rows")

if __name__ == "__main__":
    main()
