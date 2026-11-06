#!/usr/bin/env python3
"""
Weekly loop, step 1 of 3 (trusted, reviewed code on main, network allowed).

Builds the PRACTICE INPUTS only: everything the isolated practice test needs, for forecast months up to
Dec 2014. No outcome for 2015 or later is ever computed here: the crisis tables are built on a universe
that stops at 2014-12 (every outcome rule looks backwards only, so the open rows are the same as in a full
build). Inputs that cover later years (deaths, disasters) are cut at 2014-12 before they are saved.

Output folder (given with --out):
  panel.pkl                 (P, L, T, Ys): Working Model signals + every allowed extra signal, outcomes, targets
  clean/ucdp_ged_cm_26_1.parquet, clean/disasters_public.parquet   cut at 2014-12 (for the miss list)
  manifest.json             SHA-256 of every file above, package versions, source status, outcome check
"""
import argparse, hashlib, json, os, pickle, sys, importlib.metadata as md
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cycle"))
LAST = pd.Period("2014-12", "M")
ROUND1_OPEN = {"outcomes_long_open.csv.gz": "d87721cf5d927308407053beb3fed926fce74d9b5f0c786c03cd99d137581324",
               "targets_open.csv.gz": "73b9a8c9499d267e780b8d96df98aeb41bd7ca51388f438a7babd700923ce585"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def open_outcomes(out):
    """Crisis tables C1-C7 and both lenses, on country-months up to 2014-12 only."""
    import brink_outcomes as BO
    u = BO.universe()
    u = u[u.ym <= LAST].reset_index(drop=True)
    parts = [fn(u) for fn in (BO.c1, BO.c2, BO.c3, BO.c4, BO.c5, BO.c6, BO.c7)]
    L, T = BO.lenses(pd.concat(parts, ignore_index=True))
    L["ym"] = L.ym.astype(str); T["ym"] = T.ym.astype(str)
    for c in ("in_crisis", "onset", "escalation", "near", "g_event", "d_event"):
        L[c] = L[c].astype("Float64")
    L = L.sort_values(["iso3", "ym", "ctype"]).reset_index(drop=True)
    T = T.sort_values(["iso3", "ym"]).reset_index(drop=True)
    assert L.ym.max() <= str(LAST) and T.ym.max() <= str(LAST)
    paths = {}
    for name, df in (("outcomes_long_open.csv.gz", L), ("targets_open.csv.gz", T)):
        p = os.path.join(out, name); BO.write_gz(df, p); paths[name] = p
    return paths


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = os.path.abspath(a.out); os.makedirs(os.path.join(out, "clean"), exist_ok=True)
    op = open_outcomes(out)
    check = {k: {"sha256": sha(p), "round1": ROUND1_OPEN[k], "same_as_round1": sha(p) == ROUND1_OPEN[k]} for k, p in op.items()}

    import brink_claude as BC, brink_benchmarks as BM, brink_features as F, signals as SG
    from brink_grok import targets, H, LENSES
    BC.build_features()
    P, L, T = BC.prep([op["outcomes_long_open.csv.gz"]], [op["targets_open.csv.gz"]])
    P = P[P.ym <= LAST].reset_index(drop=True)
    isos = sorted(P.iso3.unique())
    for add in (SG.economic(isos), SG.austerity(isos), SG.freedoms(isos), SG.type_memory(L), SG.speed(P)):
        add["ym"] = pd.PeriodIndex(add.ym, freq="M")
        P = P.merge(add[add.ym <= LAST], on=["iso3", "ym"], how="left")
    br = BM.base_rate(P[["iso3", "ym"]], T)
    for lens in LENSES:
        for h in H:
            P[f"br_{lens}_{h}"] = br[f"p_{lens}_{h}"].values
    Ys = {h: P[["iso3", "ym"]].merge(targets(T, h), on=["iso3", "ym"], how="left") for h in H}
    assert P.ym.max() <= LAST and L.ym.max() <= LAST and T.ym.max() <= LAST
    with open(os.path.join(out, "panel.pkl"), "wb") as fh:
        pickle.dump((P, L, T, Ys), fh)

    cm = F.rd("ucdp_ged_cm_26_1"); cm = cm[cm.year.notna() & (cm.year <= LAST.year)]
    cm.to_parquet(os.path.join(out, "clean", "ucdp_ged_cm_26_1.parquet"), index=False)
    ds = F.rd("disasters_public"); y = pd.to_numeric(ds.year, errors="coerce")
    ds[y.notna() & (y <= LAST.year)].to_parquet(os.path.join(out, "clean", "disasters_public.parquet"), index=False)

    files = {}
    for dp, _, fs in os.walk(out):
        for f in fs:
            p = os.path.join(dp, f); files[os.path.relpath(p, out)] = sha(p)
    status = os.path.join(ROOT, "data", "clean", "source_status.csv")
    man = {"last_month": str(LAST), "files": files, "open_outcomes_check": check,
           "rows": int(len(P)), "countries": int(P.iso3.nunique()),
           "packages": {k: md.version(k) for k in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "requests", "country_converter")},
           "sources": pd.read_csv(status)[["source", "ok", "version", "rows"]].astype(str).to_dict("records") if os.path.exists(status) else []}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "sources"}, indent=1))


if __name__ == "__main__":
    main()
