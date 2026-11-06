#!/usr/bin/env python3
"""
Exam wall: score ONE reserved experiment on the exam years 2015-2025 (trusted, reviewed code on main).
Runs only inside the exam workflow, after the reservation is on the ledger and every fingerprint matched.

Same method as exam check 1 (cycle/exam_check.py, 26 Oct 2026), made general for any data-only experiment file:
  - Working Model = locked v1.0 (claude_params.json) on every lens with no extra signals in loop/working_model.json;
    a lens WITH extra signals is re-fitted exactly the Round 1 way on open years only (everything known by Dec 2014;
    beta calibration on 2010-2014 forecasts from a model fitted to Dec 2009; penalty = v1.0's locked values).
  - Variant = Working Model + the experiment's signals on its lens(es), fitted the same way.
  - Scored with the frozen Round 1 code (brink_score.py): the 24 pass gates against Base rate and PITF-style for both,
    48 'significantly worse' checks for both, and the direct variant vs Working Model test on every changed lens.
No decision is taken here: the result goes to the council.

--rehearsal --outcomes-dir DIR: a code check BEFORE any reservation, with NO exam data. Uses only the open outcome tables
(built on a universe cut at Dec 2014), shifts every date 15 years back (fit to Dec 1999, calibrate on 1995-1999,
score 2000-2014) and uses 50 bootstrap draws. Its numbers are not evidence; it only proves the scorer runs end to end.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "cycle")); sys.path.insert(0, os.path.join(ROOT, "loop"))
import brink_claude as BC, brink_score as S, brink_benchmarks as BM
import signals as SG, practice as PR
from brink_grok import targets, H, LENSES
from runner import check_experiment
BC.enet = lambda C: LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=C, solver="saga", max_iter=500, tol=1e-3, random_state=0)

O = os.path.join(ROOT, "data", "outcomes"); MODELS = os.path.join(ROOT, "data", "models")
LONG = [os.path.join(O, "outcomes_long_open.csv.gz"), os.path.join(O, "outcomes_long_sealed.csv.gz")]
TARG = [os.path.join(O, "targets_open.csv.gz"), os.path.join(O, "targets_sealed.csv.gz")]
FIT_END, CAL_FROM, CAL_TRAIN_END = pd.Period("2014-12", "M"), pd.Period("2010-01", "M"), pd.Period("2009-12", "M")


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment", required=True); ap.add_argument("--working-model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rehearsal", action="store_true"); ap.add_argument("--outcomes-dir")
    a = ap.parse_args()
    t0 = time.time()
    global LONG, TARG, FIT_END, CAL_FROM, CAL_TRAIN_END
    alarm_years = range(2015, 2026)
    if a.rehearsal:
        LONG = [os.path.join(a.outcomes_dir, "outcomes_long_open.csv.gz")]; TARG = [os.path.join(a.outcomes_dir, "targets_open.csv.gz")]
        FIT_END, CAL_FROM, CAL_TRAIN_END = pd.Period("1999-12", "M"), pd.Period("1995-01", "M"), pd.Period("1994-12", "M")
        S.SCORE_FROM, S.SCORE_TO, S.B = pd.Period("2000-01", "M"), pd.Period("2014-12", "M"), 50
        alarm_years = range(2000, 2015)
        assert all("sealed" not in os.path.basename(p) for p in LONG + TARG)
    spec, why = check_experiment(open(a.experiment, "rb").read())
    if spec is None:
        sys.exit(f"STOP: experiment file refused: {why}")
    wm = json.load(open(a.working_model))
    wm_extra = {lens: list(wm["extra_signals"][lens]) for lens in LENSES}
    lenses = LENSES if spec["lens"] == "both" else [spec["lens"]]
    var_extra = {lens: wm_extra[lens] + ([s for s in spec["add_signals"] if s not in wm_extra[lens]] if lens in lenses else [])
                 for lens in LENSES}

    P, L, T = BC.prep(LONG, TARG)
    if a.rehearsal:
        assert L.ym.max() <= pd.Period("2014-12", "M") and T.ym.max() <= pd.Period("2014-12", "M"), "rehearsal saw exam years"
    isos = sorted(P.iso3.unique()); n0 = len(P)
    for add in (SG.economic(isos), SG.austerity(isos), SG.freedoms(isos), SG.type_memory(L), SG.speed(P)):
        add["ym"] = pd.PeriodIndex(add.ym, freq="M")
        P = P.merge(add, on=["iso3", "ym"], how="left")
    assert len(P) == n0, "a signal table duplicated rows"
    br = BM.base_rate(P[["iso3", "ym"]], T)
    for lens in LENSES:
        for h in H:
            P[f"br_{lens}_{h}"] = br[f"p_{lens}_{h}"].values
    ym = P.ym.values; wt = BC.wts(P)
    Ys = {h: P[["iso3", "ym"]].merge(targets(T, h), on=["iso3", "ym"], how="left") for h in H}

    locked = json.load(open(os.path.join(MODELS, "claude_params.json")))
    v10, alarm = BC.predict(locked, LONG, TARG, None)
    v10["ym"] = pd.PeriodIndex(v10.ym, freq="M") if not isinstance(v10.ym.dtype, pd.PeriodDtype) else v10.ym
    assert (v10.iso3.values == P.iso3.values).all() and (v10.ym.astype(str).values == P.ym.astype(str).values).all()

    def build(extra):
        """Locked v1.0 on lenses without extra signals; Round 1-way refit on lenses with them."""
        f = v10.copy(); coefs = {}
        for lens in LENSES:
            if not extra[lens]:
                continue
            mats = []
            for h in H:
                Px = P
                if "x_own" in extra[lens]:
                    b = P[f"br_{lens}_{h}"].clip(1e-4, 1 - 1e-4)
                    Px = P.assign(x_own=np.log(b / (1 - b)))
                yv = Ys[h][f"Y_{lens}_{h}"].values
                tr0 = (ym <= CAL_TRAIN_END - h) & ~np.isnan(yv)
                ca = (ym >= CAL_FROM) & (ym <= FIT_END - h) & ~np.isnan(yv)
                X0, s0 = PR.design(Px[tr0], lens, extra[lens])
                m0 = BC.enet(PR.C_LOCK[h]).fit(X0.values, yv[tr0], sample_weight=wt[tr0])
                Xc, _ = PR.design(Px[ca], lens, extra[lens], s0)
                cal = BC.beta_cal_fit(m0.predict_proba(Xc.values)[:, 1], yv[ca])
                tr = (ym <= FIT_END - h) & ~np.isnan(yv)
                X, s = PR.design(Px[tr], lens, extra[lens])
                m = BC.enet(PR.C_LOCK[h]).fit(X.values, yv[tr], sample_weight=wt[tr])
                coefs[f"{lens}_{h}"] = {k: round(float(v), 4) for k, v in zip(X.columns, m.coef_[0]) if k in extra[lens]}
                Xa, _ = PR.design(Px, lens, extra[lens], s)
                mats.append(BC.beta_cal_apply(m.predict_proba(Xa.values)[:, 1], cal))
            M = BC.pav_rows(np.vstack(mats).T)
            for k, h in enumerate(H):
                f[f"p_{lens}_{h}"] = np.clip(M[:, k], 1e-4, 1 - 1e-4)
        return f, coefs

    base, base_coefs = build(wm_extra)
    var, var_coefs = build(var_extra)
    changed = [lens for lens in LENSES if var_extra[lens] != wm_extra[lens]]

    D = BM.pitf_frame()
    pp = json.load(open(os.path.join(MODELS, "pitf_style_params.json")))["pitf_style"]
    benches = {"Base rate": BM.base_rate(D[["iso3", "ym"]], T), "PITF-style": BM.pitf_predict(D, pp)}
    tab = S.score_table({"working_model": base, "variant": var, **benches}, T)
    gates = {}
    for nm, f in (("working_model", base), ("variant", var)):
        Mg, Wg, _ = S.gates(nm, f, benches, T, None)
        gates[nm] = {"main_gates_passed_of_24": int(Mg["pass"].sum()), "significantly_worse_of_48": int(Wg.significantly_worse.sum()),
                     "passed": Mg[Mg["pass"]][["vs", "lens", "h", "metric"]].to_dict("records"),
                     "worse": Wg[Wg.significantly_worse][["vs", "lens", "h", "metric"]].to_dict("records")}
    rng = np.random.default_rng(S.SEED); direct = []
    for lens in changed:
        for h in H:
            d = S.pair(var, base, T, lens, h); bl = S.blocks_index(d)
            for met in ["prauc", "brier", "far70"]:
                obs, pb = S.boot_test(d.y.values, d.pa.values, d.pb.values, met, bl, "better", rng)
                _, pw = S.boot_test(d.y.values, d.pa.values, d.pb.values, met, bl, "worse", rng)
                direct.append({"lens": lens, "h": h, "metric": met, "working_model": r4(S.METRICS[met][0](d.y.values, d.pb.values)),
                               "variant": r4(S.METRICS[met][0](d.y.values, d.pa.values)), "better_by": r4(obs),
                               "p_better": r4(pb), "p_worse": r4(pw), "n": int(len(d))})
    res = {"schema": 1, "experiment_id": spec["id"], "title": spec["title"], "lens": spec["lens"], "add_signals": spec["add_signals"],
           "working_model": wm.get("version"), "variant_extra_signals": var_extra, "changed_lenses": changed,
           "exam_years": "2000-01..2014-12 (REHEARSAL, not evidence)" if a.rehearsal else "2015-01..2025-12", "scoring": "brink_score.py (Round 1, frozen), B=1000, seed 20261022, Holm alpha 0.05",
           "scores": [{k: (r4(v) if isinstance(v, float) else v) for k, v in r.items()} for r in tab.to_dict("records")],
           "gates": gates, "direct_variant_vs_working_model": direct,
           "direct_cells_better_p05": sum(1 for x in direct if x["p_better"] is not None and x["p_better"] < 0.05),
           "direct_cells_worse_p05": sum(1 for x in direct if x["p_worse"] is not None and x["p_worse"] < 0.05),
           "coefficients": {"working_model": base_coefs, "variant": var_coefs},
           "atrocity_alarm_unchanged_v1_0": S.alarm_score(alarm, L, alarm_years), "seconds": int(time.time() - t0),
           "rehearsal": bool(a.rehearsal)}
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    summ = {k: res[k] for k in ("experiment_id", "changed_lenses", "direct_cells_better_p05", "direct_cells_worse_p05")}
    summ["gates_passed_of_24"] = {k: v["main_gates_passed_of_24"] for k, v in gates.items()}
    summ["significantly_worse_of_48"] = {k: v["significantly_worse_of_48"] for k, v in gates.items()}
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
