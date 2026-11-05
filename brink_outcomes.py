#!/usr/bin/env python3
"""
Brink Model - Round 1 build - JOB 1 (part 2): crisis outcome tables C1-C7, both lenses, sealed split.

Plain words
  Reads the clean input tables written by brink_data.py (data/clean/) and builds, for every
  country-month 1946-2025, whether a crisis of each type STARTED (onset) or got WORSE
  (escalation), following Exam Pack v2.2 (SHA-256 bd3d64f1...ebaec1) and the written readings in
  briefs/brink-build/brink-outcome-readings.md (every reading labelled Guess).

  Outputs (data/outcomes/):
    outcomes_long_open.csv.gz     one row per country-month per crisis type, 1946-01 .. 2014-12
    outcomes_long_sealed.csv.gz   the same for 2015-01 .. 2025-12  (SEALED: built, hashed, never printed)
    targets_open.csv.gz           one row per country-month: GLOBAL / DOMESTIC event flags (open years)
    targets_sealed.csv.gz         the same for sealed years (SEALED)
    onset_counts_open.csv         onsets per type per year, OPEN years only (safe to print)
    outcome_hashes.json           SHA-256 of every file above + this script
  Nothing about sealed years is ever printed: only file hashes.

Columns of the long table
  iso3, ym (YYYY-MM), ctype (C1..C7), observed (1 = data exists for this type here, 0 = missing),
  in_crisis (0/1), onset (0/1), escalation (0/1), band (0 none, 1 onset band, 2.. higher bands),
  near (1 = within +-20% of a threshold: reported, not scored), source, source_version, confidence.
  Missing data: observed=0 and onset/escalation empty (never zero).

Usage: python3 brink_outcomes.py
"""
import gzip, hashlib, io, json, os, re, sys, zipfile
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(ROOT, "data", "clean")
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "outcomes")
os.makedirs(OUT, exist_ok=True)

FIRST, LAST = pd.Period("1946-01", "M"), pd.Period("2025-12", "M")
SEAL_FROM = pd.Period("2015-01", "M")
NEW_EPISODE_GAP = 12          # Guess R3: a type must be out of crisis 12 months before a new onset counts
NEAR = 0.20                   # Exam Pack: +-20% dead zone
DOMESTIC_LOOKBACK = 120       # Exam Pack: previous 10 years
DOMESTIC_MIN_HISTORY = 60     # Guess R10: need 5 years of observed history, else DOMESTIC is missing


def P(y, m):
    return pd.Period(year=int(y), month=int(m), freq="M")

def rd(name):
    return pd.read_parquet(os.path.join(CLEAN, name + ".parquet"))


# --------------------------------------------------------------------- universe (Guess R1)
def universe():
    """Country-months that exist: V-Dem v16 country-years 1946-2025 (183 places), 12 months each."""
    v = rd("vdem_cy_v16")
    v = v[(v.year >= 1946) & (v.year <= 2025)][["country_text_id", "year"]].drop_duplicates()
    rows = [(c, P(y, m)) for c, y in v.itertuples(index=False) for m in range(1, 13)]
    u = pd.DataFrame(rows, columns=["iso3", "ym"])
    return u

ALIAS = {"PSE": "PSE", "PSG": "PSG"}   # V-Dem splits Palestine (West Bank PSE, Gaza PSG); UCDP books both to Israel (gw 666)


# --------------------------------------------------------------------- helpers
def level_episodes(lv, thr, bands, observed, near_on=True):
    """
    lv: level series per month for one place (float, NaN = missing); thr: onset threshold;
    bands: increasing thresholds for escalation bands (band 1 = thr).
    Returns arrays: in_crisis, onset, escalation, band, near.
    Onset = crossing thr after >= NEW_EPISODE_GAP months not in crisis (Guess R3).
    Escalation = during an episode, crossing into a band higher than any reached so far in it.
    Near = onset or non-crisis month whose level is within +-20% of thr (dead zone).
    """
    n = len(lv)
    inc = np.zeros(n); on = np.zeros(n); esc = np.zeros(n); band = np.zeros(n); near = np.zeros(n)
    last_in = -10 ** 6; ep_max = 0; prev_zone = False
    for i in range(n):
        x = lv[i]
        if not observed[i] or np.isnan(x):
            inc[i] = on[i] = esc[i] = band[i] = np.nan; near[i] = 0; prev_zone = False
            continue
        b = 0
        for k, t in enumerate(bands):
            if x >= t:
                b = k + 1
        band[i] = b
        if b >= 1:
            inc[i] = 1
            if i - last_in > NEW_EPISODE_GAP:
                on[i] = 1; ep_max = b
                if near_on and x <= thr * (1 + NEAR):
                    near[i] = 1
            elif b > ep_max:
                esc[i] = 1; ep_max = b
            last_in = i
            prev_zone = False
        else:
            zone = x >= thr * (1 - NEAR)
            if near_on and zone and not prev_zone and i - last_in > NEW_EPISODE_GAP:
                near[i] = 1          # near-miss on ENTERING the dead zone: reported, not scored (Guess R3)
            prev_zone = zone
    return inc, on, esc, band, near


def frame_for(u, iso, months, inc, on, esc, band, near, ctype, src, ver, conf, observed):
    return pd.DataFrame({"iso3": iso, "ym": months, "ctype": ctype, "observed": observed.astype(int),
                         "in_crisis": inc, "onset": on, "escalation": esc, "band": band, "near": near,
                         "source": src, "source_version": ver, "confidence": conf})


# --------------------------------------------------------------------- C1 war / armed conflict
def c1(u):
    """
    Level = organised-violence deaths (UCDP GED 26.1, all three violence types) in the rolling 12 months.
    Thresholds/bands: 1,000 (onset band), 10,000, 100,000 (tenfold steps = escalation; Guess R5).
    Plus: a NEW conflict episode crossing 25 deaths in a year (UCDP/PRIO ACD 26.1 episode start,
    month of start_date2) is an onset even when the place is already in C1 (Exam Pack: new conflict).
    Before 1989 (no GED): level proxy = 1,000 if ACD intensity_level 2 (war) that year, else 25 if
    any active conflict, else 0 (Guess R4, yearly resolution, confidence Guess).
    """
    cm = rd("ucdp_ged_cm_26_1")
    cm = cm[cm.year.notna()]
    cm["ym"] = [P(y, m) for y, m in zip(cm.year, cm.month)]
    d = cm.groupby(["iso3", "ym"]).deaths_best.sum()
    acd = rd("ucdp_ucdp_prio_armed_conflict")
    acd["year"] = acd.year.astype(int); acd["intensity_level"] = acd.intensity_level.astype(float)
    # explode multi-location rows
    acd["loc"] = acd.gwno_loc.astype(str).str.split(",")
    acd = acd.explode("loc"); acd["loc"] = acd["loc"].str.strip()
    sys.path.insert(0, ROOT)
    from brink_data import gw_to_iso3
    acd["iso3"] = acd["loc"].map(lambda g: gw_to_iso3(float(g)) if g not in ("", "nan") else None)
    war = acd.groupby(["iso3", "year"]).intensity_level.max()
    # new episodes: first row of each (conflict_id, start_date2)
    acd["sd2"] = pd.to_datetime(acd.start_date2, errors="coerce")
    ep = acd.sort_values("year").drop_duplicates(["conflict_id", "sd2", "iso3"])
    ep = ep[ep.sd2.notna()]
    newc = {}
    for r in ep.itertuples():
        newc.setdefault((r.iso3, P(r.sd2.year, r.sd2.month)), []).append(str(r.conflict_id))
    # near flag for new conflicts: battle deaths in start year within 25 +- 20% (BRD 26.1, 1989+)
    brd = rd("ucdp_ucdp_brd_conflict"); brd["year"] = brd.year.astype(int); brd["bd_best"] = brd.bd_best.astype(float)
    bd = brd.groupby(["conflict_id", "year"]).bd_best.sum()
    frames = []
    for iso, g in u.groupby("iso3"):
        months = g.ym.sort_values().tolist()
        n = len(months)
        lv = np.full(n, np.nan); obs = np.zeros(n, bool); conf = []
        isoalias = "ISR" if iso in ("PSE", "PSG") else iso
        for i, m in enumerate(months):
            if m.year >= 1989:
                # rolling 12 months of GED deaths (0 where GED has no event: GED is a full census of
                # UCDP-coded organised violence, so no event = no coded death; Guess R2)
                s = sum(d.get((isoalias, m - k), 0.0) for k in range(12))
                lv[i] = s; obs[i] = True; conf.append("Likely")
            else:
                il = war.get((isoalias, m.year), np.nan)
                lv[i] = 1000.0 if il == 2 else (25.0 if il == 1 else 0.0); obs[i] = True; conf.append("Guess")
        inc, on, esc, band, near = level_episodes(lv, 1000.0, [1000.0, 10000.0, 100000.0], obs)
        near = np.where(np.array([m.year < 1989 for m in months]), 0.0, near)   # proxy level is not a count
        # new conflict onsets
        for i, m in enumerate(months):
            ids = newc.get((isoalias, m))
            if ids:
                on[i] = 1; inc[i] = 1 if np.isnan(inc[i]) or inc[i] == 0 else inc[i]
                if band[i] == 0:
                    band[i] = 0.5      # band 0.5 = active conflict below 1,000 (25-999)
                nears = [bd.get((int(c), m.year), np.nan) for c in ids]
                if all(not np.isnan(x) and x <= 25 * (1 + NEAR) for x in nears):
                    near[i] = 1
        src = ["UCDP GED" if m.year >= 1989 else "UCDP/PRIO ACD" for m in months]
        frames.append(frame_for(u, iso, months, inc, on, esc, band, near, "C1", src, "26.1", conf, obs))
    return pd.concat(frames)


# --------------------------------------------------------------------- C2 mass violence against civilians
def c2(u):
    """
    One-sided violence by ONE group (UCDP GED 26.1 type 3, grouped by side_a actor), rolling 12 months.
    Onset band: >= 25 by one group; escalation band: >= 1,000 by one group (EWP definition).
    A new perpetrator crossing 25 in a place counts as a new onset (like a new conflict; Guess R6).
    Before 1989: no event data -> PITF genocide/politicide episodes (1955-2018) mark band 2 at their
    start month; other months MISSING (not zero) for C2 before 1989 (Guess R6).
    """
    z = zipfile.ZipFile(os.path.join(RAW, "ged261-csv.zip"))
    n = [x for x in z.namelist() if x.lower().endswith(".csv")][0]
    ev = pd.read_csv(z.open(n), usecols=["type_of_violence", "side_a_new_id", "side_a", "country_id", "date_start", "best"],
                     low_memory=False)
    ev = ev[ev.type_of_violence == 3].copy()
    sys.path.insert(0, ROOT)
    from brink_data import gw_to_iso3
    ev["iso3"] = ev.country_id.map(gw_to_iso3)
    ev["ds"] = pd.to_datetime(ev.date_start, errors="coerce")
    ev["ym"] = [P(t.year, t.month) for t in ev.ds]
    a = ev.groupby(["iso3", "side_a_new_id", "ym"]).best.sum()
    actors = a.reset_index().groupby("iso3").side_a_new_id.unique().to_dict()
    pg = rd("pitf_genocide_politicide")
    pitf_starts = {}
    for r in pg[["CCODE", "YRBEGIN", "MOBEGIN"]].drop_duplicates().itertuples(index=False):
        try:
            iso_ = gw_to_iso3(float(r.CCODE)); y = int(float(r.YRBEGIN)); mo = int(float(r.MOBEGIN))
        except Exception:
            continue
        if iso_ and 1 <= mo <= 12:
            pitf_starts.setdefault(iso_, []).append((y, mo))
    frames = []
    for iso, g in u.groupby("iso3"):
        months = g.ym.sort_values().tolist(); nm = len(months)
        isoalias = "ISR" if iso in ("PSE", "PSG") else iso
        inc = np.full(nm, np.nan); on = np.full(nm, np.nan); esc = np.full(nm, np.nan)
        band = np.full(nm, np.nan); near = np.zeros(nm); obs = np.array([m.year >= 1989 for m in months])
        conf = ["Likely" if o else "missing" for o in obs]
        idx = {m: i for i, m in enumerate(months)}
        for i in range(nm):
            if obs[i]:
                inc[i] = on[i] = esc[i] = band[i] = 0
        for act in actors.get(isoalias, []):
            lv = np.array([sum(a.get((isoalias, act, m - k), 0.0) for k in range(12)) if months[i].year >= 1989 else np.nan
                           for i, m in enumerate(months)])
            ai, ao, ae, ab, an = level_episodes(lv, 25.0, [25.0, 1000.0], obs)
            for i in range(nm):
                if not obs[i]:
                    continue
                inc[i] = max(inc[i], ai[i]); band[i] = max(band[i], ab[i])
                if ao[i] == 1:
                    on[i] = 1
                    if an[i] == 1 and near[i] == 0:
                        near[i] = 1
                    elif an[i] == 0:
                        near[i] = -1     # a clean onset overrides a near flag from another actor
                if ae[i] == 1:
                    esc[i] = 1
                if ao[i] == 0 and an[i] == 1 and inc[i] == 0 and near[i] == 0:
                    near[i] = 1
        near = np.where(near < 0, 0, near)
        # pre-1989: PITF genocide/politicide episode starts mark an escalation-band onset (Guess R6)
        for (y, mo) in pitf_starts.get(iso, []):
            if y < 1989 and (P(y, mo) in idx):
                i = idx[P(y, mo)]
                obs[i] = True; inc[i] = 1; on[i] = 1; esc[i] = 0; band[i] = 2; conf[i] = "Guess"
        frames.append(frame_for(u, iso, months, inc, on, esc, band, near, "C2",
                                ["UCDP GED type 3" if m.year >= 1989 else ("PITF genocide/politicide 2018" if o else "")
                                 for m, o in zip(months, obs)], "26.1 / PITF 2018", conf, obs))
    return pd.concat(frames)


# --------------------------------------------------------------------- event-type helper (C3, C5)
def point_events(months, events, dedupe):
    """events: dict month -> (band, near). Onset if no event of this type in the previous `dedupe` months."""
    n = len(months); inc = np.zeros(n); on = np.zeros(n); esc = np.zeros(n); band = np.zeros(n); near = np.zeros(n)
    last = -10 ** 6
    for i, m in enumerate(months):
        if m in events:
            b, nr = events[m]
            inc[i] = 1; band[i] = b
            if i - last > dedupe:
                on[i] = 1; near[i] = nr
            last = i
        elif m in NEARMISS_C5.get("_cur", {}):
            near[i] = 1
    return inc, on, esc, band, near

NEARMISS_C5 = {}


# --------------------------------------------------------------------- C3 state breakdown
IRREGULAR_END = {0, 1, 2, 3, 5, 6, 7, 8}   # V-Dem v2regendtype: coups (0,1), self-coup (2), assassination (3),
                                            # loss in civil war (5) / inter-state war (6), foreign intervention (7),
                                            # popular uprising (8). Not: natural death, managed transitions (Guess R7)

def c3(u):
    """
    Point events: (a) end of a regime by an irregular process (V-Dem v16 v2regendtype + end date parsed from
    v2reginfo), 1946-2025; (b) PITF adverse regime change episode start month (to 2018);
    (c) CSP successful coup (success == 1), to 2021 (replacement for Powell & Thyne, which is unreachable).
    Events from different sources within 3 months are one event (Guess R7). No escalation band.
    """
    sys.path.insert(0, ROOT)
    from brink_data import gw_to_iso3
    ev = {}
    v = rd("vdem_cy_v16")
    v = v[v.v2regendtype.isin(IRREGULAR_END) & v.v2reginfo.notna()][["country_text_id", "v2reginfo", "v2regendtype"]].drop_duplicates()
    for iso, info, t in v.itertuples(index=False):
        mm = re.search(r"-\s*(\d{2})/(\d{2})/(\d{4})\)", str(info))
        if mm:
            y, mo = int(mm.group(3)), int(mm.group(2))
            if 1946 <= y <= 2025:
                ev.setdefault(iso, {}).setdefault(P(y, mo), set()).add("V-Dem v16 regime end")
    arc = rd("pitf_adverse_regime_change")
    for r in arc[["CCODE", "YRBEGIN", "MOBEGIN"]].drop_duplicates().itertuples(index=False):
        try:
            iso = gw_to_iso3(float(r.CCODE)); y = int(float(r.YRBEGIN)); mo = int(float(r.MOBEGIN))
        except Exception:
            continue
        if iso and 1 <= mo <= 12 and 1946 <= y:
            ev.setdefault(iso, {}).setdefault(P(y, mo), set()).add("PITF ARC 2018")
    ann = rd("csp_coups_annual")
    sc2cc = dict(zip(ann.scode.astype(str), ann.ccode.astype(str)))
    cl = rd("csp_coups_list")
    for r in cl.itertuples(index=False):
        try:
            if int(float(r.success)) != 1:
                continue
            iso = gw_to_iso3(float(sc2cc.get(str(r.scode), "nan"))); y = int(float(r.year)); mo = int(float(r.mth))
        except Exception:
            continue
        if iso and 1 <= mo <= 12:
            ev.setdefault(iso, {}).setdefault(P(y, mo), set()).add("CSP coups v2021")
    frames = []
    for iso, g in u.groupby("iso3"):
        months = g.ym.sort_values().tolist(); obs = np.ones(len(months), bool)
        e = ev.get(iso, {})
        inc, on, esc, band, near = point_events(months, {m: (1, 0) for m in e}, 3)
        src = ["; ".join(sorted(e[m])) if m in e else ("V-Dem v16 + PITF ARC + CSP" if m.year <= 2018 else
               ("V-Dem v16 + CSP" if m.year <= 2021 else "V-Dem v16 only")) for m in months]
        conf = ["Likely" if m.year <= 2021 else "Guess" for m in months]
        frames.append(frame_for(u, iso, months, inc, on, esc, band, near, "C3", src, "v16 / 2018 / v2021", conf, obs))
    return pd.concat(frames)


# --------------------------------------------------------------------- C4 economic collapse
def _annual_to_month7(series_by_year, years):
    """Annual value placed in July of its year; other months of an observed year carry 0 (Guess R8)."""
    out = {}
    for y in years:
        val = series_by_year.get(y, np.nan)
        for mo in range(1, 13):
            out[P(y, mo)] = (val if mo == 7 else (0.0 if not np.isnan(val) else np.nan))
    return out

def c4(u):
    """
    Arms (each its own episodes; C4 onset = any arm onset):
      inflation >= 25% a year: IMF CPI monthly YoY (1990+); WDI annual CPI inflation before that or where
        IMF is missing (placed in July). Bands 25 / 100 / 1,000 % (Guess R8).
      currency crash >= 30% fall in a year vs a trade-weighted basket: IMF NEER (1990+, ~100 countries);
        else WDI official rate vs US dollar, annual (placed in July). Bands 30 / 60 / 90 % (Guess R8).
      sovereign default: BoC-BoE 2026, debt in default to private creditors (FC bonds, FC bank loans, local
        currency debt) or the Paris Club > 0; annual, placed in July (Guess R8).
      systemic banking crisis: MISSING (Laeven & Valencia unreachable from this cloud).
    """
    w = rd("wdi_current")
    infl_a = w[w.indicator == "inflation_cpi"].dropna(subset=["value"]).set_index(["iso3", "year"]).value.to_dict()
    fx = w[w.indicator == "fx_lcu_per_usd"].dropna(subset=["value"]).set_index(["iso3", "year"]).value.to_dict()
    cpi = rd("imf_cpi_yoy_m")
    cpi["ym"] = [pd.Period(p.replace("-M", "-"), "M") for p in cpi.period]
    cpi = cpi.dropna(subset=["cpi_yoy"]).set_index(["iso3", "ym"]).cpi_yoy.to_dict()
    ne = rd("imf_neer_m")
    ne = ne.sort_values("indicator").drop_duplicates(["iso3", "period"])   # ACW before AEW
    ne["ym"] = [pd.Period(p.replace("-M", "-"), "M") for p in ne.period]
    ne = ne.set_index(["iso3", "ym"]).value.to_dict()
    # defaults
    b = rd("bocboe_2026_Debt_2026.csv")
    c = b.iloc[:, 2].astype(str)
    h = c[c == "k2"].index[0]
    tab = b.iloc[h + 1:, 2:].copy(); tab.columns = [str(x) for x in b.iloc[h, 2:]]
    tab = tab[tab.DEBT_YEAR.astype(str).str.fullmatch(r"\d{4}")]
    num = lambda col: pd.to_numeric(tab[[x for x in tab.columns if x.startswith(col)][0]], errors="coerce").fillna(0)
    tab["priv"] = num("DEBT_FC_BANK_LOANS") + num("DEBT_FC_BONDS") + num("DEBT_LC_DEBT") + num("DEBT_PARIS_CLUB")
    ccol = [x for x in tab.columns if x.startswith("DEBT_COUNTRY")][0]
    names = tab[ccol].astype(str).unique().tolist()
    import country_converter as coco
    conv = dict(zip(names, coco.convert(names, to="ISO3", not_found=None)))
    tab["iso3"] = tab[ccol].map(conv)
    dflt = tab.dropna(subset=["iso3"]).groupby(["iso3", tab.DEBT_YEAR.astype(int)]).priv.sum().to_dict()
    dflt_years = set(int(y) for y in tab.DEBT_YEAR.astype(int).unique())
    frames = []
    for iso, g in u.groupby("iso3"):
        months = g.ym.sort_values().tolist(); n = len(months)
        years = sorted(set(m.year for m in months))
        ia = _annual_to_month7({y: infl_a.get((iso, y), np.nan) for y in years}, years)
        fa = {}
        for y in years:
            a0, a1 = fx.get((iso, y - 1), np.nan), fx.get((iso, y), np.nan)
            fa[y] = 100 * (1 - a0 / a1) if (a0 and a1 and not np.isnan(a0) and not np.isnan(a1) and a1 > 0) else np.nan
        fa = _annual_to_month7(fa, years)
        da = _annual_to_month7({y: (float(dflt.get((iso, y), 0.0) > 0) if y in dflt_years else np.nan) for y in years}, years)
        lv_i = np.array([cpi.get((iso, m), ia.get(m, np.nan)) for m in months])
        src_i = ["IMF CPI" if (iso, m) in cpi else "WDI CPI (annual)" for m in months]
        lv_f = []
        src_f = []
        for m in months:
            a, b0 = ne.get((iso, m)), ne.get((iso, m - 12))
            if a is not None and b0 is not None and b0 > 0:
                lv_f.append(max(0.0, 100 * (1 - a / b0))); src_f.append("IMF NEER")
            else:
                lv_f.append(fa.get(m, np.nan)); src_f.append("WDI USD rate (annual)")
        lv_f = np.array(lv_f, float)
        lv_d = np.array([da.get(m, np.nan) for m in months], float)
        arms = []
        for lv, thr, bands, nf in [(lv_i, 25.0, [25.0, 100.0, 1000.0], True), (lv_f, 30.0, [30.0, 60.0, 90.0], True), (lv_d, 1.0, [1.0], False)]:
            obs = ~np.isnan(lv)
            arms.append(level_episodes(lv, thr, bands, obs, near_on=nf) + (obs,))
        obs_any = arms[0][5] | arms[1][5] | arms[2][5]
        inc = np.full(n, np.nan); on = np.full(n, np.nan); esc = np.full(n, np.nan); band = np.full(n, np.nan); near = np.zeros(n)
        for i in range(n):
            if not obs_any[i]:
                continue
            vals = [a for a in arms if a[5][i]]
            inc[i] = max(a[0][i] for a in vals); band[i] = max(a[3][i] for a in vals)
            on[i] = float(any(a[1][i] == 1 for a in vals)); esc[i] = float(any(a[2][i] == 1 for a in vals))
            clean_on = any(a[1][i] == 1 and a[4][i] == 0 for a in vals)
            near[i] = 1.0 if (any(a[4][i] == 1 for a in vals) and not clean_on) else 0.0
        src = [f"{src_i[i]} | {src_f[i]} | BoC-BoE 2026 | banking arm MISSING" for i in range(n)]
        conf = ["Likely" if (months[i].year >= 1990) else "Guess" for i in range(n)]
        frames.append(frame_for(u, iso, months, inc, on, esc, band, near, "C4", src, "IMF current / WDI current / BoC-BoE 2026", conf, obs_any))
    return pd.concat(frames)


# --------------------------------------------------------------------- C5 major disaster
def c5(u):
    """
    One event qualifies if deaths >= 1,000, or deaths >= 100 per million people, or displaced >= 1% of
    population (DFO floods only; 'affected' read as displaced, Guess R9). Sources: NOAA NCEI earthquakes,
    tsunamis, volcanoes (1946-2026) + Dartmouth Flood Observatory (to Oct 2021). No EM-DAT (licence).
    Each qualifying event is an onset. The COVID pandemic is ONE global event (iso3 WLD), kept out of the
    country table and listed separately (Guess R9). Coverage is PARTIAL (no storms, droughts, heat, other
    epidemics; no floods after 2021-10): months with no qualifying event are coded 0 with confidence Guess
    and source note 'partial coverage' - this is the one place a no-event month is 0; see reading R9.
    """
    d = rd("disasters_public")
    d = d[d.iso3.notna() & (d.iso3 != "WLD")].copy()
    for c in ("deaths", "year", "month"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["displaced"] = pd.to_numeric(d.get("displaced"), errors="coerce")
    d = d.dropna(subset=["year", "month"])
    d = d[(d.month >= 1) & (d.month <= 12)]
    w = rd("wdi_current"); pop = w[w.indicator == "population"].dropna(subset=["value"])
    first = pop.sort_values("year").groupby("iso3").first().value.to_dict()
    pop = pop.set_index(["iso3", "year"]).value.to_dict()
    ev = {}; nearmiss = {}
    for r in d.itertuples(index=False):
        y, mo = int(r.year), int(r.month)
        pp = pop.get((r.iso3, y), first.get(r.iso3, np.nan))
        de = r.deaths if not np.isnan(r.deaths) else np.nan
        ratios = [de / 1000.0 if not np.isnan(de) else np.nan,
                  (de / (pp / 1e6)) / 100.0 if (not np.isnan(de) and pp and not np.isnan(pp)) else np.nan,
                  (r.displaced / pp) / 0.01 if (not np.isnan(r.displaced) and pp and not np.isnan(pp)) else np.nan]
        ratios = [x for x in ratios if not np.isnan(x)]
        if not ratios:
            continue
        mx = max(ratios)
        m = P(y, mo)
        if mx >= 1.0:
            b = 1 + (1 if (not np.isnan(de) and de >= 10000) else 0) + (1 if (not np.isnan(de) and de >= 100000) else 0)
            nr = 1 if mx <= 1 + NEAR else 0
            pb, pn = ev.get((r.iso3, m), (0, 1))
            ev[(r.iso3, m)] = (max(pb, b), min(pn, nr))
        elif mx >= 1 - NEAR:
            nearmiss[(r.iso3, m)] = 1
    frames = []
    for iso, g in u.groupby("iso3"):
        months = g.ym.sort_values().tolist(); obs = np.ones(len(months), bool)
        e = {m: ev[(iso, m)] for m in months if (iso, m) in ev}
        NEARMISS_C5["_cur"] = {m: 1 for m in months if (iso, m) in nearmiss and m not in e}
        inc, on, esc, band, near = point_events(months, e, 0)
        src = ["NOAA NCEI + DFO floods (partial coverage)" if m <= P(2021, 10) else "NOAA NCEI only (partial coverage; floods missing)" for m in months]
        frames.append(frame_for(u, iso, months, inc, on, esc, band, near, "C5", src, "NOAA API v1 / DFO current", ["Guess"] * len(months), obs))
    NEARMISS_C5.clear()
    return pd.concat(frames)


# --------------------------------------------------------------------- C6 restriction collapse
def c6(u):
    """
    Fall of 0.15+ in one year on V-Dem v16 physical-integrity (v2x_clphy) OR civil-liberties (v2x_civlib).
    Yearly change y-1 -> y, placed in July of y (Guess R10). Bands: fall 0.15 (onset), 0.30 (escalation).
    """
    v = rd("vdem_cy_v16")[["country_text_id", "year", "v2x_clphy", "v2x_civlib"]].sort_values(["country_text_id", "year"])
    for c in ("v2x_clphy", "v2x_civlib"):
        v[c + "_d"] = v.groupby("country_text_id")[c].diff()
        prev_year = v.groupby("country_text_id").year.shift()
        v.loc[prev_year != v.year - 1, c + "_d"] = np.nan
    v["fall"] = (-v[["v2x_clphy_d", "v2x_civlib_d"]].min(axis=1, skipna=True)).clip(lower=0)
    v.loc[v[["v2x_clphy_d", "v2x_civlib_d"]].isna().all(axis=1), "fall"] = np.nan
    fall = v.set_index(["country_text_id", "year"]).fall.to_dict()
    frames = []
    for iso, g in u.groupby("iso3"):
        months = g.ym.sort_values().tolist()
        years = sorted(set(m.year for m in months))
        a = _annual_to_month7({y: fall.get((iso, y), np.nan) for y in years}, years)
        lv = np.array([a.get(m, np.nan) for m in months], float)
        obs = ~np.isnan(lv)
        inc, on, esc, band, near = level_episodes(lv, 0.15, [0.15, 0.30], obs)
        frames.append(frame_for(u, iso, months, inc, on, esc, band, near, "C6", "V-Dem v16 v2x_clphy / v2x_civlib", "v16",
                                ["Likely"] * len(months), obs))
    return pd.concat(frames)


# --------------------------------------------------------------------- C7 mass unrest
def c7(u):
    """
    Proxy (Guess R12; no ACLED, no injury counts exist in open data): Mass Mobilization Project (CC0),
    1990-01 .. 2020-03, 162 countries. A month has unrest if one protest event has
      (participants >= 2,000, or 'thousands') AND (protester violence OR state response of beatings,
      shootings or killings), OR its notes mention a curfew or a state of emergency.
    New onset after 12 quiet months. Countries/months outside MMP coverage are MISSING (never zero).
    """
    m = rd("mmp_protests")
    m = m[m.protest.astype(str) == "1"].copy()
    def npart(r):
        cat = str(r.participants_category)
        if cat in (">10000", "5000-10000", "2000-4999"):
            return True
        if cat in ("100-999", "50-99", "1000-1999"):
            return False
        t = str(r.participants).lower().replace(",", "")
        if "thousand" in t or re.search(r"\d000s", t):
            return True
        nums = [int(x) for x in re.findall(r"\d+", t)]
        return bool(nums) and max(nums) >= 2000
    big = m.apply(npart, axis=1)
    sr = m[[c for c in m.columns if c.startswith("stateresponse")]].astype(str).apply(lambda col: col.str.lower())
    harsh = sr.isin(["beatings", "shootings", "killings"]).any(axis=1)
    pv = pd.to_numeric(m.protesterviolence, errors="coerce").fillna(0) == 1
    curfew = m.notes.astype(str).str.contains("curfew|state of emergency", case=False)
    m["hit"] = (big & (pv | harsh)) | curfew
    m["ym"] = [P(y, mo) if str(mo).replace(".0", "").isdigit() and 1 <= int(float(mo)) <= 12 else None
               for y, mo in zip(pd.to_numeric(m.startyear, errors="coerce").fillna(0).astype(int), m.startmonth)]
    hits = set(zip(m[m.hit & m.ym.notna()].iso3, m[m.hit & m.ym.notna()].ym))
    covered = set(m.iso3.dropna())
    frames = []
    lo, hi = P(1990, 1), P(2020, 3)
    for iso, g in u.groupby("iso3"):
        months = g.ym.sort_values().tolist()
        obs = np.array([(iso in covered) and lo <= mm <= hi for mm in months])
        lv = np.array([(1.0 if (iso, mm) in hits else 0.0) if o else np.nan for mm, o in zip(months, obs)])
        inc, on, esc, band, near = level_episodes(lv, 1.0, [1.0], obs, near_on=False)
        frames.append(frame_for(u, iso, months, inc, on, esc, band, near, "C7",
                                ["Mass Mobilization Project (proxy)" if o else "MISSING (no ACLED-free source)" for o in obs],
                                "mmALL_073120", ["Guess" if o else "missing" for o in obs], obs))
    return pd.concat(frames)


# --------------------------------------------------------------------- lenses (Guess R11)
def lenses(L):
    """
    Per type: GLOBAL event = onset or escalation (not in the near dead zone).
    DOMESTIC event = onset with no onset of that type in the previous 120 months, or escalation to a band
    not reached in the previous 120 months; needs >= 60 observed months of that type in the window, else
    missing. Combined per country-month: event = 1 if any type has an event; 'near' if no event but any
    near flag (excluded from scoring); missing if no type is observed.
    """
    L = L.sort_values(["iso3", "ctype", "ym"]).reset_index(drop=True)
    L["g_event"] = np.where(L.observed == 1, ((L.onset == 1) | (L.escalation == 1)).astype(float), np.nan)
    L.loc[(L.g_event == 1) & (L.near == 1) & (L.escalation != 1), "g_event"] = np.nan   # onset in dead zone
    dom = np.full(len(L), np.nan)
    for (iso, ct), g in L.groupby(["iso3", "ctype"], sort=False):
        ix = g.index.values; on = g.onset.values; es = g.escalation.values; bd = g.band.values; ob = g.observed.values
        for k in range(len(ix)):
            lo = max(0, k - DOMESTIC_LOOKBACK)
            if ob[k] != 1:
                continue
            if ob[lo:k].sum() < DOMESTIC_MIN_HISTORY:
                continue
            prior_on = np.nansum(on[lo:k]) > 0
            prior_band = np.nanmax(bd[lo:k]) if np.any(~np.isnan(bd[lo:k])) else 0
            ev = (on[k] == 1 and not prior_on) or (es[k] == 1 and bd[k] > prior_band)
            dom[ix[k]] = 1.0 if ev else 0.0
    L["d_event"] = dom
    L.loc[(L.d_event == 1) & (L.near == 1) & (L.escalation != 1), "d_event"] = np.nan
    def comb(col):
        g = L.groupby(["iso3", "ym"])[col]
        anyev = g.max()
        return anyev
    T = pd.DataFrame({
        "global_event": L.groupby(["iso3", "ym"]).g_event.max(),
        "domestic_event": L.groupby(["iso3", "ym"]).d_event.max(),
        "near_any": L.groupby(["iso3", "ym"]).near.max(),
        "types_observed": L.groupby(["iso3", "ym"])["observed"].sum(),
        "types_list": L[L.observed == 1].groupby(["iso3", "ym"]).ctype.apply(lambda s: "".join(x[1] for x in sorted(s))),
    }).reset_index()
    # a month whose only signal is a dead-zone onset is excluded (NaN), not calm
    nz = L[(L.near == 1)].groupby(["iso3", "ym"]).size()
    for col in ("global_event", "domestic_event"):
        k = T.set_index(["iso3", "ym"]).index
        is_near = k.isin(nz.index)
        T.loc[is_near & (T[col].fillna(0).values == 0), col] = np.nan
    T.loc[T.types_observed == 0, ["global_event", "domestic_event"]] = np.nan
    return L, T


# --------------------------------------------------------------------- main
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def write_gz(df, path):
    # deterministic gzip (mtime=0) so the same table always gives the same fingerprint
    buf = df.to_csv(index=False).encode()
    with open(path, "wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
            gz.write(buf)

def main():
    u = universe()
    parts = []
    cache = os.path.join(OUT, "_build_cache.pkl")
    if "--reuse" in sys.argv and os.path.exists(cache):
        L = pd.read_pickle(cache); L, T = lenses(L); return finish(L, T)
    for nm, fn in [("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5), ("C6", c6), ("C7", c7)]:
        print("building", nm, flush=True)
        parts.append(fn(u))
    L = pd.concat(parts, ignore_index=True)
    L.to_pickle(os.path.join(OUT, "_build_cache.pkl"))   # local cache only, never published
    L, T = lenses(L)
    return finish(L, T)

def finish(L, T):
    L["ym"] = L.ym.astype(str); T["ym"] = T.ym.astype(str)
    for c in ("in_crisis", "onset", "escalation", "near", "g_event", "d_event"):
        L[c] = L[c].astype("Float64")
    L = L.sort_values(["iso3", "ym", "ctype"]).reset_index(drop=True)
    T = T.sort_values(["iso3", "ym"]).reset_index(drop=True)
    sealed = L.ym >= str(SEAL_FROM); sealedT = T.ym >= str(SEAL_FROM)
    files = {
        "outcomes_long_open.csv.gz": L[~sealed], "outcomes_long_sealed.csv.gz": L[sealed],
        "targets_open.csv.gz": T[~sealedT], "targets_sealed.csv.gz": T[sealedT],
    }
    hashes = {}
    for f, df in files.items():
        p = os.path.join(OUT, f); write_gz(df, p); hashes[f] = sha(p)
    # open-years onset counts (safe to print)
    Lo = L[~sealed].copy(); Lo["year"] = Lo.ym.str[:4].astype(int)
    cnt = Lo[Lo.onset == 1].groupby(["year", "ctype"]).size().unstack(fill_value=0)
    esc = Lo[Lo.escalation == 1].groupby(["year", "ctype"]).size().unstack(fill_value=0).add_suffix("_esc")
    near = Lo[Lo.near == 1].groupby(["year", "ctype"]).size().unstack(fill_value=0).add_suffix("_near")
    cnt = cnt.join(esc, how="outer").join(near, how="outer").fillna(0).astype(int)
    To = T[~sealedT].copy(); To["year"] = To.ym.str[:4].astype(int)
    cnt["GLOBAL_event_months"] = To.groupby("year").global_event.sum().astype(int)
    cnt["DOMESTIC_event_months"] = To.groupby("year").domestic_event.sum().astype(int)
    cnt.to_csv(os.path.join(OUT, "onset_counts_open.csv"))
    hashes["onset_counts_open.csv"] = sha(os.path.join(OUT, "onset_counts_open.csv"))
    hashes["brink_outcomes.py"] = sha(os.path.abspath(__file__))
    # outside-table record: the one global pandemic event (Exam Pack rule)
    hashes["note"] = "COVID-19 pandemic = one GLOBAL event (WLD, 2020-03, date Guess), held outside the country table."
    json.dump(hashes, open(os.path.join(OUT, "outcome_hashes.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in hashes.items()}, indent=1))
    print("rows open:", int((~sealed).sum()), " countries:", L.iso3.nunique())

if __name__ == "__main__":
    main()
