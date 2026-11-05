#!/usr/bin/env python3
"""
Brink Model - Round 1 build - shared AS-OF feature machinery (used by Jobs 2-6).

Plain words
  For every country and every forecast month t (1946-01 .. 2025-12) this builds what was PUBLIC at the end
  of month t, never later revisions (Exam Pack v2.2 s.3, 'no hindsight'):
    - UCDP deaths: final GED release published by t (vintages 4.0 .. 26.1, from June 2015) plus every
      monthly Candidate release published by t (2018 on). Before June 2015 no vintage exists: GED 4.0 is
      used with the lag rule 'events of year y usable from 1 June y+1' (Guess, hindsight-risk flag).
    - World Bank WDI: the archived edition current at t (May 2009 on); before that the May 2009 edition
      with the lag rule 'year y usable from 1 July y+1' (Guess, hindsight-risk flag).
    - V-Dem: the version released by t (v8 April 2018 .. v16 March 2026; v12 missing so v11.1 runs to
      March 2023). Before April 2018: v8 with 'year y usable from 1 April y+1' (Guess, hindsight-risk flag).
    - UNHCR, UCDP/PRIO conflict list: no vintages; lag rules from the vintage table (Guess).
  Outcome-history features (already in crisis, months since last crisis) are NOT built here: they come
  from the outcome tables, so the sealed years are only touched by the Job 6 scoring run.

  Output: data/features/asof_panel.parquet (one row per iso3 x month), plus a hindsight_flag column
  (1 = at least one input came from a lag rule rather than a real vintage).
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(ROOT, "data", "clean")
FEAT = os.path.join(ROOT, "data", "features")
os.makedirs(FEAT, exist_ok=True)
sys.path.insert(0, ROOT)

FIRST, LAST = pd.Period("1946-01", "M"), pd.Period("2025-12", "M")
MONTHS = pd.period_range(FIRST, LAST, freq="M")

def rd(n):
    return pd.read_parquet(os.path.join(CLEAN, n + ".parquet"))

def universe():
    v = rd("vdem_cy_v16")
    v = v[(v.year >= 1946) & (v.year <= 2025)][["country_text_id", "year"]].drop_duplicates()
    return sorted(v.country_text_id.unique()), set(zip(v.country_text_id, v.year))

def ucdp_iso(iso):
    return "ISR" if iso in ("PSE", "PSG") else iso


# ------------------------------------------------------------------ UCDP deaths as-of
GED_REL = {"4.0": "2015-06", "5.0": "2016-06", "17.1": "2017-06", "18.1": "2018-06", "19.1": "2019-06",
           "20.1": "2020-06", "21.1": "2021-06", "22.1": "2022-06", "23.1": "2023-06", "24.1": "2024-06",
           "25.1": "2025-06", "26.1": "2026-06"}

def _ged_matrix(version, isos, tv=None):
    cm = rd("ucdp_ged_cm_" + version.replace(".", "_"))
    cm = cm[cm.year.notna()]
    if tv is not None:
        cm = cm[cm.tv == tv]
    cm = cm.assign(ym=[pd.Period(year=int(y), month=int(m), freq="M") for y, m in zip(cm.year, cm.month)])
    s = cm.groupby(["iso3", "ym"]).deaths_best.sum()
    M = pd.DataFrame(0.0, index=isos, columns=MONTHS)
    for (iso, ym), v in s.items():
        if iso in M.index and ym in M.columns:
            M.at[iso, ym] = v
    # A country with NO events at all in this release, but events in the latest release for the same period,
    # was left out of this release (e.g. Syria before GED 18.1): missing, never zero (Guess).
    if version != "26.1":
        present = set(cm.iso3.dropna())
        lat = rd("ucdp_ged_cm_26_1"); lat = lat[lat.year.notna() & (lat.year <= cm.year.max())]
        if tv is not None:
            lat = lat[lat.tv == tv]
        absent = (set(lat[lat.deaths_best > 0].iso3.dropna()) - present) & set(isos)
        if absent:
            M.loc[sorted(absent), :] = np.nan
    return M

def deaths_asof(isos, tv=None):
    """
    Returns dict: origin t -> (avail_through, composite matrix view) is too big; instead returns
    two DataFrames (iso x origin): d3 = deaths in the last 3 available months, d12 = last 12 available months,
    plus avail (months of lag between t and the newest available month) and hindsight flag per origin.
    tv: None = all organised violence; 3 = one-sided only.
    """
    uis = [ucdp_iso(i) for i in isos]
    base = sorted(set(uis))
    out3 = pd.DataFrame(np.nan, index=base, columns=MONTHS)
    out12 = out3.copy(); out1 = out3.copy()
    availm = pd.Series(np.nan, index=MONTHS); hind = pd.Series(0, index=MONTHS)
    g40 = _ged_matrix("4.0", base, tv)
    # --- before June 2015: GED 4.0 with the June y+1 lag rule
    for t in MONTHS[MONTHS < pd.Period("2015-06", "M")]:
        y_av = t.year - 1 if t.month >= 6 else t.year - 2
        A = pd.Period(year=y_av, month=12, freq="M")
        if A < pd.Period("1989-12", "M"):
            continue
        cols3 = [A - k for k in range(3)]; cols12 = [A - k for k in range(12)]
        out3[t] = g40[cols3].sum(axis=1); out12[t] = g40[cols12].sum(axis=1); out1[t] = g40[A]
        availm[t] = (t - A).n; hind[t] = 1
    # --- June 2015 on: final releases + Candidate releases, each used from its release month
    cand = rd("ucdp_candidate_cm_all_releases")
    if tv is not None:
        cand = cand[cand.tv == tv]
    cand = cand.assign(ym=[pd.Period(year=int(y), month=int(m), freq="M") for y, m in zip(cand.year, cand.month)],
                       rel=[pd.Period(str(r)[:7], "M") for r in cand.release_date])
    # Candidate releases are incremental (newest month + late additions): sum them per month (Guess).
    # A month counts as known only once a release has it as its newest month, or a final GED covers it.
    rel_list = []
    for v, g in cand.groupby("ged_version"):
        r = g.rel.iloc[0]
        sub = g.groupby(["iso3", "ym"]).deaths_best.sum()
        rel_list.append((r, g.ym.max(), sub))
    allrel = rd("ucdp_candidate_cm_all_releases")
    import country_converter as coco
    cont = dict(zip(base, coco.convert(base, to="continent", not_found=None)))
    africa = [i for i in base if cont.get(i) == "Africa"]
    GLOBAL_FROM = pd.Period("2020-07", "M")   # Guess: Candidate covered Africa only before mid-2020 (seen in the files)
    main_month = {}
    for v, g in allrel.groupby("ged_version"):
        r = pd.Period(str(g.release_date.iloc[0])[:7], "M")
        my = int(g.year.max()); mm = int(g[g.year == my].month.max())
        main_month.setdefault(r, set()).add(pd.Period(year=my, month=mm, freq="M"))
    comp = None; covered_final = None
    add = pd.DataFrame(0.0, index=base, columns=MONTHS)
    known = pd.DataFrame(False, index=base, columns=MONTHS)
    finals = {v: pd.Period(r, "M") for v, r in GED_REL.items()}
    for t in MONTHS[MONTHS >= pd.Period("2015-06", "M")]:
        newf = [v for v, r in finals.items() if r == t]
        if newf:
            v = newf[0]
            comp = _ged_matrix(v, base, tv) if v != "4.0" else g40.copy()
            covered_final = pd.Period(year=t.year - 1, month=12, freq="M")
        for r, mx, sub in rel_list:
            if r == t:
                for (iso, ym), val in sub.items():
                    if iso in add.index and ym in add.columns:
                        add.at[iso, ym] += val
        for m in main_month.get(t, set()):
            cov = base if t >= GLOBAL_FROM else africa
            known.loc[cov, m] = True
        if comp is None:
            continue
        for iso in base:
            kn = [m for m in known.columns[known.loc[iso].values] if covered_final < m <= t]
            A = max([covered_final] + kn)
            def window(k):
                cols = [A - j for j in range(k)]
                ok = [c for c in cols if c <= covered_final or known.at[iso, c]]
                if not ok:
                    return np.nan
                tot = sum((comp.at[iso, c] if c <= covered_final else add.at[iso, c]) for c in ok)
                return tot * (k / len(ok))
            out3.at[iso, t] = window(3); out12.at[iso, t] = window(12)
            out1.at[iso, t] = comp.at[iso, A] if A <= covered_final else add.at[iso, A]
        availm[t] = (t - covered_final).n; hind[t] = 0
    return out1, out3, out12, availm, hind


# ------------------------------------------------------------------ yearly sources as-of
def _lag_year(t, month_from):
    """Latest data year usable at origin t when year y becomes usable on month `month_from` of y+1."""
    return t.year - 1 if t.month >= month_from else t.year - 2

def wdi_asof(indicator, isos):
    """iso x origin: (value, year) of the latest observation in the edition current at t."""
    w = rd("wdi_vintages")
    w = w[w.indicator == indicator].dropna(subset=["value"])
    eds = sorted(w.edition.unique())
    ed_period = {e: pd.Period(f"{e[:4]}-{e[4:6]}", "M") for e in eds}
    val = pd.DataFrame(np.nan, index=isos, columns=MONTHS); yr = val.copy(); hind = pd.Series(0, index=MONTHS)
    cache = {}
    def latest(e, maxyear):
        k = (e, maxyear)
        if k not in cache:
            g = w[(w.edition == e) & (w.year <= maxyear)].sort_values("year").groupby("iso3").last()
            cache[k] = g
        return cache[k]
    first = eds[0]
    for t in MONTHS:
        cur = [e for e in eds if ed_period[e] <= t]
        if not cur:
            if t.year < 1961:
                continue
            g = latest(first, _lag_year(t, 7)); hind[t] = 1
        else:
            g = latest(cur[-1], 9999)
        idx = g.index.intersection(val.index)
        val.loc[idx, t] = g.loc[idx, "value"].values; yr.loc[idx, t] = g.loc[idx, "year"].values
    return val, yr, hind

def wdi_world_mean_asof(indicator, how="mean"):
    """Cross-country mean (or median) of the indicator per (edition, year): used for 'relative to world'."""
    w = rd("wdi_vintages"); w = w[(w.indicator == indicator) & w.iso3.str.len().eq(3)].dropna(subset=["value"])
    f = {"mean": "mean", "median": "median"}[how]
    return w.groupby(["edition", "year"]).value.agg(f)

VDEM_VERS = [("8", "2018-04"), ("9", "2019-04"), ("10", "2020-03"), ("111", "2021-03"), ("13", "2023-03"),
             ("14", "2024-03"), ("15", "2025-03"), ("16", "2026-03")]

def vdem_asof(var, isos):
    tabs = {v: rd("vdem_cy_v" + v) for v, _ in VDEM_VERS}
    rel = {v: pd.Period(r, "M") for v, r in VDEM_VERS}
    val = pd.DataFrame(np.nan, index=isos, columns=MONTHS); yr = val.copy(); hind = pd.Series(0, index=MONTHS)
    cache = {}
    def latest(v, maxyear):
        k = (v, maxyear)
        if k not in cache:
            d = tabs[v]
            if var not in d.columns:
                cache[k] = None
            else:
                g = d[(d.year <= maxyear)].dropna(subset=[var]).sort_values("year").groupby("country_text_id").last()
                cache[k] = g
        return cache[k]
    for t in MONTHS:
        cur = [v for v, _ in VDEM_VERS if rel[v] <= t]
        if not cur:
            g = None
            for v, _ in VDEM_VERS:             # earliest version that carries the variable (v8 lacks some)
                g = latest(v, _lag_year(t, 4))
                if g is not None:
                    break
            hind[t] = 1
        else:
            g = None
            for v in reversed(cur):          # newest version that has the variable (v8/v10 lack some)
                g = latest(v, 9999)
                if g is not None:
                    if v != cur[-1]:
                        hind[t] = 1
                    break
            if g is None:
                g = latest("16", t.year - 1); hind[t] = 1
        if g is None:
            continue
        idx = g.index.intersection(val.index)
        val.loc[idx, t] = g.loc[idx, var].values; yr.loc[idx, t] = g.loc[idx, "year"].values
    return val, yr, hind

def vdem_change_asof(var, isos, years_back=1):
    """Change in var between the latest published year and `years_back` years before, both from the SAME
    version current at t (so a later rewrite of history never enters)."""
    tabs = {v: rd("vdem_cy_v" + v) for v, _ in VDEM_VERS}
    rel = {v: pd.Period(r, "M") for v, r in VDEM_VERS}
    out = pd.DataFrame(np.nan, index=isos, columns=MONTHS); cache = {}
    def ch(v, maxyear):
        k = (v, maxyear)
        if k not in cache:
            d = tabs[v]
            if var not in d.columns:
                cache[k] = None
            else:
                d = d[d.year <= maxyear][["country_text_id", "year", var]].dropna()
                last = d.sort_values("year").groupby("country_text_id").last()
                prev = d.set_index(["country_text_id", "year"])[var]
                vals = {}
                for iso, r in last.iterrows():
                    p = prev.get((iso, r.year - years_back), np.nan)
                    vals[iso] = r[var] - p
                cache[k] = pd.Series(vals)
        return cache[k]
    for t in MONTHS:
        cur = [v for v, _ in VDEM_VERS if rel[v] <= t]
        s = (next((ch(v, _lag_year(t, 4)) for v, _ in VDEM_VERS if ch(v, _lag_year(t, 4)) is not None), None) if not cur
             else next((ch(v, 9999) for v in reversed(cur) if ch(v, 9999) is not None), None))
        if s is None:
            continue
        idx = s.index.intersection(out.index)
        out.loc[idx, t] = s.loc[idx].values
    return out

def yearly_lagged(series_by_iso_year, isos, month_from):
    """For sources with no vintages: value of the latest year usable at t under the lag rule."""
    out = pd.DataFrame(np.nan, index=isos, columns=MONTHS); yr = out.copy()
    s = series_by_iso_year.dropna()
    for t in MONTHS:
        ly = _lag_year(t, month_from)
        g = s[s.index.get_level_values(1) <= ly]
        if len(g) == 0:
            continue
        g = g.reset_index().sort_values(g.index.names[1]).groupby(g.index.names[0]).last()
        idx = g.index.intersection(out.index)
        out.loc[idx, t] = g.loc[idx].iloc[:, -1].values; yr.loc[idx, t] = g.loc[idx].iloc[:, 0].values
    return out, yr


# ------------------------------------------------------------------ conflict list and borders
def acd_active_by_year():
    from brink_data import gw_to_iso3
    a = rd("ucdp_ucdp_prio_armed_conflict")
    a["year"] = a.year.astype(int); a["intensity_level"] = a.intensity_level.astype(float)
    a["loc"] = a.gwno_loc.astype(str).str.split(","); a = a.explode("loc")
    a["iso3"] = a["loc"].str.strip().map(lambda g: gw_to_iso3(float(g)) if g not in ("", "nan") else None)
    return a.groupby(["iso3", "year"]).intensity_level.max()

def borders():
    b = rd("borders").dropna(subset=["iso3", "border_iso3"])
    nb = b.groupby("iso3").border_iso3.apply(list).to_dict()
    # V-Dem-only codes share their parent's borders (Guess)
    for k, p in {"PSG": "PSE", "SML": "SOM", "ZZB": "TZA"}.items():
        nb.setdefault(k, nb.get(p, []))
    return nb


def to_long(df, name):
    s = df.stack(future_stack=True)
    s.name = name
    return s


if __name__ == "__main__":
    print("this module is imported by the design programs")
