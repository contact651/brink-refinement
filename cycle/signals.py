"""Candidate signals for idea tests (cycle 1). Every value is what was public at the end of month t, using the same
as-of rules as brink_features.py (World Bank archived editions; V-Dem versions; outcome history up to t only)."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import brink_features as F

def _long(df, n):
    s = F.to_long(df, n).reset_index(); s.columns = ["iso3", "ym", n]; return s

def wdi_asof_extra(indicator, isos):
    orig = F.rd
    w = pd.read_parquet(os.path.join(F.CLEAN, "wdi_vintages_extra.parquet"))
    F.rd = lambda n: w if n == "wdi_vintages" else orig(n)
    try:
        return F.wdi_asof(indicator, isos)
    finally:
        F.rd = orig

def economic(isos):
    infl, _, _ = F.wdi_asof("inflation_cpi", isos)
    g, _, _ = F.wdi_asof("gdp_pc_growth", isos)
    a = _long(np.log1p(infl.clip(lower=0)), "x_infl")
    b = _long((-g).clip(lower=0), "x_gdpfall")
    return a.merge(b, on=["iso3", "ym"])

def austerity(isos):
    gc, _, _ = wdi_asof_extra("govcons_real_growth", isos)
    return _long(np.log1p((-gc).clip(lower=0)), "x_cut")

def freedoms(isos):
    ch = F.vdem_change_asof("v2x_civlib", isos, years_back=2)
    return _long((-ch).clip(lower=0), "x_libfall")

def type_memory(L, types=("C3", "C4", "C6", "C7")):
    out = None
    for ct in types:
        M = L[L.ctype == ct].set_index(["iso3", "ym"]).onset.unstack("ym")
        M = M.reindex(columns=pd.period_range(pd.Period("1946-01", "M"), M.columns.max(), freq="M"))
        A = (M.fillna(0).values == 1); res = np.zeros(A.shape); last = np.full(A.shape[0], -1)
        for k in range(A.shape[1]):
            last = np.where(A[:, k], k, last)
            res[:, k] = np.where(last >= 0, k - last, k + 1)
        df = pd.DataFrame(np.log1p(np.minimum(res, 240)), index=M.index, columns=M.columns)
        s = _long(df, f"x_since_{ct}")
        out = s if out is None else out.merge(s, on=["iso3", "ym"], how="outer")
    return out

def speed(P):
    d3 = np.expm1(P.f1); d12 = np.expm1(P.f1b)
    return pd.DataFrame({"iso3": P.iso3, "ym": P.ym, "x_speed": np.log1p(4 * d3) - np.log1p(d12)})
