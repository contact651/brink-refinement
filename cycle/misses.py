#!/usr/bin/env python3
"""Miss list + one plain cause per miss, for the Working Model's practice-year forecasts (2000-2014)."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import brink_features as F

TYPE_NAME = {"C1": "war / armed conflict", "C2": "mass killing of civilians", "C3": "state breakdown / coup",
             "C4": "economic collapse", "C5": "major disaster", "C6": "sudden loss of freedoms", "C7": "mass unrest"}

def thresholds(f, Ys, joined):
    thr = {}
    for h in (3, 6, 12):
        d = joined(f, Ys, "GLOBAL", h)
        p = np.sort(d.p.values[d.y.values == 1])[::-1]
        thr[h] = float(p[int(np.ceil(0.7 * len(p))) - 1])        # cut that catches 70% of events
    return thr

def miss_list(f, L, P, thr):
    ev = L[(L.g_event == 1) & (L.ym >= pd.Period("2000-01", "M")) & (L.ym <= pd.Period("2014-12", "M"))][["iso3", "ym", "ctype", "band", "onset", "escalation"]].copy()
    fp = f.set_index(["iso3", "ym"])
    def p_at(iso, m, h):
        try: return float(fp.at[(iso, m - h), f"p_GLOBAL_{h}"])
        except KeyError: return np.nan
    for h in (12, 6, 3):
        ev[f"p{h}"] = [p_at(i, m, h) for i, m in zip(ev.iso3, ev.ym)]
    ev = ev.dropna(subset=["p12", "p6", "p3"], how="all")
    ev["flag12"] = ev.p12 >= thr[12]; ev["flag6"] = ev.p6 >= thr[6]; ev["flag3"] = ev.p3 >= thr[3]
    ev["status"] = np.where(ev.flag12 | ev.flag6, "seen in time", np.where(ev.flag3, "seen late (3 months)", "missed"))
    # human cost: UCDP GED 26.1 deaths in that country in the 12 months from the event + disaster deaths that month
    cm = F.rd("ucdp_ged_cm_26_1"); cm = cm[cm.year.notna()]
    cm["ym"] = [pd.Period(year=int(y), month=int(mo), freq="M") for y, mo in zip(cm.year, cm.month)]
    dd = cm.groupby(["iso3", "ym"]).deaths_best.sum()
    dis = F.rd("disasters_public"); dis = dis[dis.year.notna() & dis.month.notna() & dis.iso3.notna()]
    dis = dis[(pd.to_numeric(dis.month, errors="coerce") >= 1) & (pd.to_numeric(dis.month, errors="coerce") <= 12)]
    dis["ym"] = [pd.Period(year=int(float(y)), month=int(float(mo)), freq="M") for y, mo in zip(dis.year, dis.month)]
    ds = dis.groupby(["iso3", "ym"]).deaths.sum()
    def ali(i): return "ISR" if i in ("PSE", "PSG") else i
    ev["deaths_next12"] = [sum(dd.get((ali(i), m + k), 0.0) for k in range(12)) for i, m in zip(ev.iso3, ev.ym)]
    ev["deaths_prior12_actual"] = [sum(dd.get((ali(i), m - k), 0.0) for k in range(1, 13)) for i, m in zip(ev.iso3, ev.ym)]
    ev["disaster_deaths"] = [float(ds.get((i, m), 0.0)) for i, m in zip(ev.iso3, ev.ym)]
    ev["human_cost"] = np.where(ev.ctype.isin(["C1", "C2"]), ev.deaths_next12, np.where(ev.ctype == "C5", ev.disaster_deaths, 0.0))
    # what the model could see at the 12-month forecast date
    pv = P.set_index(["iso3", "ym"])
    ev["seen_violence_12mo_before"] = [float(pv.f1b.get((i, m - 12), np.nan)) if (i, m - 12) in pv.index else np.nan for i, m in zip(ev.iso3, ev.ym)]
    return ev

def cause(r):
    """One plain cause per miss (rule written before ideas were drafted)."""
    if r.ctype == "C5":
        return "true surprise (no public warning signal for earthquakes, floods, volcanoes)"
    if r.ctype in ("C1", "C2"):
        seen = r.seen_violence_12mo_before
        if (np.isnan(seen) or seen == 0) and r.deaths_prior12_actual > 0:
            return "data late or hidden (deaths already happening were not yet published at forecast time)"
        if seen and seen > 0:
            return "weighed too low or too slow (the model saw violence but did not raise the alarm enough)"
        return "true surprise (no violence or other public signal beforehand)"
    return {"C3": "signal not read (the model has no signal for coup risk or fragile rulers)",
            "C4": "signal not read (the model has no economic signal: prices, currency, debt)",
            "C6": "signal not read (the model does not watch freedoms being squeezed)",
            "C7": "signal not read (the model has no signal for street pressure: prices, cuts, protests)"}[r.ctype]
