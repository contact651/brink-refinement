#!/usr/bin/env python3
"""
Weekly loop, step 3 of 3 (trusted, reviewed code on main; runs after the isolated step).
  done-ids : list the experiment ids the ledger already holds, so the runner never tests one twice.
  append   : check the isolated step's single result file against a fixed shape, then APPEND one line to
             ledger/results.jsonl, the full result to ledger/runs/, and one line to ledger/runs.jsonl.
             Existing lines are never edited. A kept idea gets a proposed Working Model file (for a candidate
             branch); nothing reaches main without the council verdict and a member's approval.
  failed   : append a 'failed' line to ledger/runs.jsonl (a failed run never counts as an idea dropped); it says
             whether this run's result line was already recorded before the failure.
Standard library only.
"""
import argparse, datetime as dt, hashlib, json, os, re, sys

STATUS = {"experiment ran", "monitoring only, no experiment ran"}
TOP = {"schema", "status", "working_model", "practice_years", "keep_rule", "working_model_scores", "misses",
       "experiments", "refused", "queue_left", "seconds"}
EXP = {"file", "id", "title", "lens", "variant_extra_signals", "add_signals", "sha256", "verdict", "rules", "cells",
       "continents_worse", "coefficients"}
SIGNAL = re.compile(r"x_[a-z0-9_]{1,30}")


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fail(msg):
    print("RESULT REFUSED:", msg); sys.exit(1)


def check(r):
    if set(r) != TOP: fail(f"top-level keys {sorted(set(r) ^ TOP)}")
    if r["schema"] != 1 or r["status"] not in STATUS: fail("schema or status")
    if not isinstance(r["experiments"], list) or not isinstance(r["refused"], list): fail("lists")
    if (r["status"] == "experiment ran") != bool(r["experiments"]): fail("status does not match experiments")
    for e in r["experiments"]:
        if set(e) != EXP: fail(f"experiment keys {sorted(set(e) ^ EXP)}")
        if not re.fullmatch(r"[A-Za-z0-9_]{1,40}", e["id"]) or e["verdict"] not in ("KEEP", "DROP"): fail("experiment id/verdict")
        if not re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) or e["lens"] not in ("both", "GLOBAL", "DOMESTIC"): fail("sha/lens")
        if not all(isinstance(v, bool) for v in e["rules"].values()) or len(e["rules"]) != 3: fail("rules")
        if (e["verdict"] == "KEEP") != all(e["rules"].values()): fail("verdict does not follow the keep rule")
        for lens, sig in e["variant_extra_signals"].items():
            if lens not in ("GLOBAL", "DOMESTIC") or not all(SIGNAL.fullmatch(s) for s in sig): fail("signals")
    for x in r["refused"]:
        if x.get("verdict") != "REFUSED" or not re.fullmatch(r"[0-9a-f]{64}", x.get("sha256", "")): fail("refused entry")
    for k in ("GLOBAL_12", "DOMESTIC_12"):
        if k not in r["working_model_scores"]: fail("scores")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def append(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["done-ids", "append", "failed"])
    ap.add_argument("--ledger", required=True); ap.add_argument("--out"); ap.add_argument("--result"); ap.add_argument("--manifest")
    ap.add_argument("--working-model"); ap.add_argument("--proposals")
    a = ap.parse_args()
    run = {"run_id": os.environ.get("GITHUB_RUN_ID", ""), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
           "commit": os.environ.get("GITHUB_SHA", ""), "workflow": os.environ.get("GITHUB_WORKFLOW", ""),
           "run_url": f"{os.environ.get('GITHUB_SERVER_URL', '')}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"}
    res_path = os.path.join(a.ledger, "results.jsonl")
    if a.cmd == "done-ids":
        ids = set()
        if os.path.exists(res_path):
            for line in open(res_path, encoding="utf-8"):
                if line.strip():
                    ids.update(e["id"] for e in json.loads(line).get("experiments", []))
        json.dump(sorted(ids), open(a.out, "w")); print("already tested:", sorted(ids)); return
    if a.cmd == "failed":
        have = os.path.exists(res_path) and any(line.strip() and json.loads(line).get("run_id") == run["run_id"] and
                                                  json.loads(line).get("run_attempt") == run["run_attempt"] for line in open(res_path, encoding="utf-8"))
        why = "failed after its result was recorded (a later step broke; see the run)" if have else "failed (no result recorded)"
        append(os.path.join(a.ledger, "runs.jsonl"), {"date": now(), **run, "status": why}); return
    r = json.load(open(a.result)); check(r)
    man = json.load(open(a.manifest))
    day = now()[:10]
    os.makedirs(os.path.join(a.ledger, "runs"), exist_ok=True)
    full = os.path.join(a.ledger, "runs", f"{day}_{run['run_id']}.json")
    json.dump({"run": run, "inputs_manifest": man, "result": r}, open(full, "w"), indent=1, sort_keys=True)
    line = {"date": now(), **run, "status": r["status"], "working_model": r["working_model"],
            "experiments": [{"id": e["id"], "title": e["title"], "lens": e["lens"], "add_signals": e["add_signals"],
                             "verdict": e["verdict"], "rules": e["rules"], "sha256": e["sha256"]} for e in r["experiments"]],
            "refused": [{"file": x["file"], "reason": x.get("reason", "")} for x in r["refused"]],
            "working_model_12m": {k: r["working_model_scores"][k] for k in ("GLOBAL_12", "DOMESTIC_12")},
            "misses_12m_false_alarms": r["misses"].get("false_alarm_country_months_12m"),
            "queue_left": r["queue_left"], "full_record": os.path.relpath(full, a.ledger), "full_record_sha256": sha_file(full)}
    append(res_path, line)
    append(os.path.join(a.ledger, "runs.jsonl"), {"date": now(), **run, "status": "ok", "result": r["status"],
                                                  "inputs_sha256": sha_file(a.manifest)})
    if a.proposals:
        os.makedirs(a.proposals, exist_ok=True)
        wm = json.load(open(a.working_model))
        for e in r["experiments"]:
            if e["verdict"] == "KEEP":
                p = dict(wm); p["extra_signals"] = e["variant_extra_signals"]
                p["version"] = f"{wm['version']} + {e['id']} (candidate, awaiting council)"
                json.dump(p, open(os.path.join(a.proposals, f"{e['id']}.json"), "w"), indent=1)
    print(json.dumps(line, indent=1))


if __name__ == "__main__":
    main()
