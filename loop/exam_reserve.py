#!/usr/bin/env python3
"""
Exam wall: the reservation counter (trusted, reviewed code on main). Standard library only.

The exam years (2015-2025) may be checked at most 3 times per calendar quarter. Before any exam outcome is built or
read, the exam workflow appends ONE reservation line to ledger/exam_ledger.jsonl and pushes it with a normal push.
If that push fails, the workflow stops and nothing is built. A reservation is never refunded: a crash after it still
counts. Lines are only ever added; this file has no command that edits or removes one.

  reserve : (with --dry-run: only say whether it WOULD reserve; writes nothing)
            refuse (and log a 'refused' line) unless the experiment file is byte-identical to one the weekly practice
            run KEPT, and the quarter still has a free check; otherwise append a 'reservation' line.
  verify  : after the push, confirm the newest reservation of this run names this exact experiment file and commit.
  outcome : append the outcome of a reserved check (scored / fingerprint mismatch / failed after reservation).
  used    : print the checks used per quarter (for people and for the self-test).

Checks used before the ledger existed are listed, on main, in pub/prior_exam_checks.json and count in full.
"""
import argparse, datetime as dt, hashlib, json, os, re, sys

LIMIT = 3
ID = re.compile(r"[A-Za-z0-9_]{1,40}")


def now_utc(s=None):
    if s:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    return dt.datetime.now(dt.timezone.utc)


def quarter(t):
    return f"{t.year}Q{(t.month - 1) // 3 + 1}"


def lines(path):
    if not os.path.exists(path):
        return []
    out = []
    for k, line in enumerate(open(path, encoding="utf-8"), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                sys.exit(f"STOP: line {k} of {path} is not valid JSON; the ledger must be checked by a person")
    return out


def used(ledger, prior):
    n = {}
    for x in (json.load(open(prior)) if prior and os.path.exists(prior) else []):
        n[x["quarter"]] = n.get(x["quarter"], 0) + 1
    for x in lines(os.path.join(ledger, "exam_ledger.jsonl")):
        if x.get("type") == "reservation":
            n[x["quarter"]] = n.get(x["quarter"], 0) + 1
    return n


def kept_on_practice(ledger, exp_id, sha):
    """True only if a weekly practice result line KEPT this id with this exact file hash."""
    for r in lines(os.path.join(ledger, "results.jsonl")):
        for e in r.get("experiments", []):
            if e.get("id") == exp_id and e.get("sha256") == sha and e.get("verdict") == "KEEP":
                return True
    return False


DRY = False


def append(path, obj):
    if DRY:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def run_info():
    return {"run_id": os.environ.get("GITHUB_RUN_ID", ""), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
            "workflow": os.environ.get("GITHUB_WORKFLOW", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reserve", "verify", "outcome", "used"])
    ap.add_argument("--ledger", required=True); ap.add_argument("--prior")
    ap.add_argument("--experiment"); ap.add_argument("--queue"); ap.add_argument("--commit")
    ap.add_argument("--now"); ap.add_argument("--status"); ap.add_argument("--detail", default="")
    ap.add_argument("--result-file", default=""); ap.add_argument("--summary"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    global DRY
    DRY = a.dry_run and a.cmd == "reserve"
    path = os.path.join(a.ledger, "exam_ledger.jsonl")
    t = now_utc(a.now); q = quarter(t); stamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")

    if a.cmd == "used":
        print(json.dumps(used(a.ledger, a.prior), sort_keys=True)); return

    if a.cmd == "outcome":
        if a.status not in ("scored", "fingerprint mismatch", "failed after reservation (not refunded)"):
            sys.exit("STOP: unknown outcome status")
        summ = json.load(open(a.summary)) if a.summary and os.path.exists(a.summary) else {}
        append(path, {"type": "outcome", "date": stamp, **run_info(), "status": a.status, "detail": a.detail[:400],
                      "result_file": a.result_file, "summary": summ})
        print("outcome logged:", a.status); return

    if not (a.experiment and ID.fullmatch(a.experiment)):
        sys.exit("STOP: experiment id must be 1-40 letters, digits or _")
    exp_path = os.path.join(a.queue, a.experiment + ".json")
    if not (a.commit and re.fullmatch(r"[0-9a-f]{40}", a.commit)):
        sys.exit("STOP: commit must be a full 40-character revision")

    if a.cmd == "reserve":
        base = {"date": stamp, **run_info(), "quarter": q, "experiment_id": a.experiment, "scorer_commit": a.commit}
        if not os.path.isfile(exp_path):
            append(path, {"type": "refused", **base, "reason": "no such experiment file on main"})
            print("REFUSED: no such experiment file on main"); sys.exit(3)
        sha = hashlib.sha256(open(exp_path, "rb").read()).hexdigest()
        base["experiment_sha256"] = sha
        if not kept_on_practice(a.ledger, a.experiment, sha):
            append(path, {"type": "refused", **base, "reason": "this exact experiment file was not KEPT by a weekly practice run"})
            print("REFUSED: not kept on practice with this exact file"); sys.exit(3)
        n = used(a.ledger, a.prior).get(q, 0)
        if n >= LIMIT:
            append(path, {"type": "refused", **base, "reason": f"quarter limit reached ({n} of {LIMIT} checks used in {q})"})
            print(f"REFUSED: {q} already used {n} of {LIMIT} checks"); sys.exit(3)
        append(path, {"type": "reservation", **base, "check_no": n + 1, "limit": LIMIT})
        print(f"{'WOULD RESERVE' if DRY else 'RESERVED'}: check {n + 1} of {LIMIT} for {q}"); return

    if a.cmd == "verify":
        sha = hashlib.sha256(open(exp_path, "rb").read()).hexdigest()
        mine = [x for x in lines(path) if x.get("type") == "reservation" and x.get("run_id") == os.environ.get("GITHUB_RUN_ID", "")
                and x.get("run_attempt") == os.environ.get("GITHUB_RUN_ATTEMPT", "")]
        if not mine:
            sys.exit("STOP: no reservation for this run is on the ledger")
        r = mine[-1]
        if r["experiment_sha256"] != sha or r["scorer_commit"] != a.commit or r["experiment_id"] != a.experiment:
            sys.exit("STOP: the reservation does not match this experiment file or this code revision")
        print(f"VERIFIED: check {r['check_no']} of {r['limit']} for {r['quarter']}, experiment {a.experiment} {sha[:12]}, code {a.commit[:12]}")


if __name__ == "__main__":
    main()
