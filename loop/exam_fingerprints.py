#!/usr/bin/env python3
"""
Exam wall: fingerprint check (trusted, reviewed code on main). Standard library only.
Every named file must exist and its SHA-256 must equal the pinned part in pub/exam_fingerprint.json (Round 1, frozen).
Any mismatch or missing file: prints which, exits 1, and nothing is scored (fail closed). Prints no table contents.
Usage: exam_fingerprints.py --manifest pub/exam_fingerprint.json --out check.json NAME=PATH [NAME=PATH ...]
"""
import argparse, hashlib, json, os, sys


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+"); a = ap.parse_args()
    parts = json.load(open(a.manifest))["parts"]
    rows, bad = [], []
    for item in a.files:
        name, _, path = item.partition("=")
        if name not in parts:
            sys.exit(f"STOP: {name} has no pinned fingerprint")
        got = sha(path) if os.path.isfile(path) else None
        ok = got == parts[name]
        rows.append({"file": name, "expected": parts[name], "got": got, "match": ok})
        if not ok:
            bad.append(name)
    json.dump({"all_match": not bad, "mismatches": bad, "files": rows}, open(a.out, "w"), indent=1)
    for r in rows:
        print(("MATCH    " if r["match"] else "MISMATCH ") + r["file"])
    if bad:
        print("FINGERPRINT MISMATCH: nothing will be scored. Files:", ", ".join(bad)); sys.exit(1)
    print("all fingerprints match")


if __name__ == "__main__":
    main()
