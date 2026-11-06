#!/usr/bin/env python3
"""
Weekly loop, step 2 of 3: the ONE reviewed practice runner. Runs inside the isolated step (no network, no
repository history, empty environment) on the practice inputs only (forecast months up to Dec 2014).

An experiment is a DATA file in queue/ (JSON). It can only pick signals from ALLOWED_SIGNALS and a lens.
Nothing in it is ever executed: no code, paths, web addresses or commands are accepted, and any unknown key
or value refuses the file. A refused file is reported as REFUSED, never as DROP.

Every run: re-fits the Working Model on the practice years, scores it, lists misses and false alarms; then
tests at most MAX_PER_RUN queued experiments under the keep rule in cycle/practice.py (K1-K3, written before
any idea was tested). Writes one result file. Queue empty = "monitoring only, no experiment ran".
"""
import argparse, hashlib, json, os, pickle, re, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "cycle"))

ALLOWED_SIGNALS = {
    "x_infl": "inflation, as published (log)", "x_gdpfall": "fall in income per person",
    "x_cut": "cut in real government spending (austerity)", "x_libfall": "fall in civil liberties over 2 years",
    "x_since_C3": "time since last state breakdown", "x_since_C4": "time since last economic collapse",
    "x_since_C6": "time since last loss of freedoms", "x_since_C7": "time since last mass unrest",
    "x_speed": "violence speeding up (last 3 months vs last 12)", "x_own": "the place's own past crisis rate",
}
KEYS = {"id", "title", "added", "lens", "add_signals", "note"}
REQUIRED = {"id", "title", "added", "lens", "add_signals"}
LENS_VALUES = {"both", "GLOBAL", "DOMESTIC"}
BAD_TEXT = re.compile(r"(://|\\|\.\.|`|\$\(|\$\{|<script|import |exec\(|eval\(|__)", re.I)
MAX_PER_RUN = 2


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def check_experiment(raw):
    """Return (spec, None) if the file is a valid data-only experiment, else (None, reason)."""
    try:
        spec = json.loads(raw.decode("utf-8"))
    except Exception:
        return None, "not valid JSON"
    if not isinstance(spec, dict):
        return None, "not a JSON object"
    extra = set(spec) - KEYS
    if extra:
        return None, f"unknown keys {sorted(extra)}"
    if not REQUIRED <= set(spec):
        return None, f"missing keys {sorted(REQUIRED - set(spec))}"
    if not (isinstance(spec["id"], str) and re.fullmatch(r"[A-Za-z0-9_]{1,40}", spec["id"])):
        return None, "id must be 1-40 letters, digits or _"
    for k in ("title", "note"):
        v = spec.get(k, "")
        if not isinstance(v, str) or len(v) > 400 or BAD_TEXT.search(v):
            return None, f"{k} must be plain text up to 400 characters"
    if not (isinstance(spec["added"], str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", spec["added"])):
        return None, "added must be a date YYYY-MM-DD"
    if spec["lens"] not in LENS_VALUES:
        return None, f"lens must be one of {sorted(LENS_VALUES)}"
    s = spec["add_signals"]
    if not (isinstance(s, list) and 1 <= len(s) <= 6 and all(isinstance(x, str) for x in s) and len(set(s)) == len(s)):
        return None, "add_signals must be a list of 1-6 different names"
    bad = [x for x in s if x not in ALLOWED_SIGNALS]
    if bad:
        return None, f"signals not in the reviewed list: {bad}"
    return spec, None


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=os.path.join(HERE, "inputs"))
    ap.add_argument("--queue", default=os.path.join(HERE, "queue"))
    ap.add_argument("--done", default=os.path.join(HERE, "done_ids.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "result.json"))
    a = ap.parse_args()
    t0 = time.time()
    import practice as PR, misses as MS
    from brink_grok import H, LENSES

    wm = json.load(open(os.path.join(HERE, "loop", "working_model.json")))
    wm_extra = {lens: list(wm["extra_signals"][lens]) for lens in LENSES}
    with open(os.path.join(a.inputs, "panel.pkl"), "rb") as fh:
        P, L, T, Ys = pickle.load(fh)
    assert P.ym.max() <= pd.Period("2014-12", "M"), "practice inputs reach past 2014"

    def forecast(extra):
        """Working Model (or a variant) fitted per lens with that lens's extra signals."""
        f = PR.fit_predict(P, Ys, extra["GLOBAL"])
        if extra["DOMESTIC"] != extra["GLOBAL"]:
            fd = PR.fit_predict(P, Ys, extra["DOMESTIC"])
            assert (fd.iso3.values == f.iso3.values).all() and (fd.ym.values == f.ym.values).all()
            for c in [c for c in f.columns if c.startswith("p_DOMESTIC_")]:
                f[c] = fd[c].values
        return f

    # --- Working Model on the practice years
    base = forecast(wm_extra)
    scores = {k: {m: r4(v) if m in ("prauc", "brier") else v for m, v in d.items()} for k, d in PR.summary_scores(base, Ys).items()}
    thr = MS.thresholds(base, Ys, PR.joined)
    ev = MS.miss_list(base, L, P, thr)
    ev["cause"] = ev.apply(MS.cause, axis=1)
    d12 = PR.joined(base, Ys, "GLOBAL", 12)
    fa = d12[(d12.p >= thr[12]) & (d12.y == 0)]
    top = (ev[ev.status != "seen in time"].sort_values("human_cost", ascending=False).head(10)
           [["iso3", "ym", "ctype", "status", "human_cost", "cause"]])
    misses = {"thresholds_70pct_catch": {str(k): r4(v) for k, v in thr.items()},
              "events": int(len(ev)), "status_counts": ev.status.value_counts().to_dict(),
              "by_cause": ev[ev.status != "seen in time"].groupby("cause").size().to_dict(),
              "false_alarm_country_months_12m": int(len(fa)), "calm_country_months_12m": int((d12.y == 0).sum()),
              "top_misses_by_human_cost": [{"iso3": r.iso3, "month": str(r.ym), "type": r.ctype, "status": r.status,
                                            "human_cost": r4(r.human_cost), "cause": r.cause} for r in top.itertuples()]}

    # --- queue: data-only experiment files, oldest first, skipping ids already tested
    done = set(json.load(open(a.done))) if os.path.exists(a.done) else set()
    files = sorted(f for f in os.listdir(a.queue) if f.endswith(".json")) if os.path.isdir(a.queue) else []
    refused, todo = [], []
    for f in files:
        raw = open(os.path.join(a.queue, f), "rb").read()
        spec, why = check_experiment(raw)
        if spec is None:
            refused.append({"file": f, "sha256": sha_bytes(raw), "verdict": "REFUSED", "reason": why}); continue
        if spec["id"] in done:
            continue
        todo.append((f, spec, sha_bytes(raw)))
    todo.sort(key=lambda x: (x[1]["added"], x[0]))
    results = []
    for f, spec, h in todo[:MAX_PER_RUN]:
        lenses = LENSES if spec["lens"] == "both" else [spec["lens"]]
        extra = {lens: wm_extra[lens] + ([s for s in spec["add_signals"] if s not in wm_extra[lens]] if lens in lenses else [])
                 for lens in LENSES}
        var = forecast(extra)
        Tb, R, v = PR.compare(var, base, Ys, spec["id"])
        cells = [{"lens": r.lens, "h": int(r.h), "metric": r.metric, "working_model": r4(r.model), "variant": r4(r.variant_score),
                  "better_by": r4(r.diff_better), "p_better": r4(r.p_better), "p_worse": r4(r.p_worse),
                  "sig_better": bool(r.sig_better) if r.sig_better == r.sig_better else False, "sig_worse": bool(r.sig_worse)}
                 for r in Tb.itertuples()]
        results.append({"file": f, "id": spec["id"], "title": spec["title"], "lens": spec["lens"], "variant_extra_signals": extra,
                        "add_signals": spec["add_signals"], "sha256": h, "verdict": "KEEP" if v["KEEP"] else "DROP",
                        "rules": {k: bool(x) for k, x in v.items() if k != "KEEP"}, "cells": cells,
                        "continents_worse": R[R.sig_worse].continent.tolist() if len(R) else [],
                        "coefficients": var.attrs.get("coefs", {})})
    status = "experiment ran" if results else "monitoring only, no experiment ran"
    res = {"schema": 1, "status": status, "working_model": wm.get("version"), "practice_years": "2000-01..2014-12",
           "keep_rule": "cycle/practice.py K1-K3 (B=500, seed 20261026, Holm alpha 0.05)",
           "working_model_scores": scores, "misses": misses, "experiments": results, "refused": refused,
           "queue_left": max(0, len(todo) - MAX_PER_RUN), "seconds": int(time.time() - t0)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(status, [(r["id"], r["verdict"]) for r in results], "refused", len(refused))


if __name__ == "__main__":
    main()
