#!/usr/bin/env python3
"""
Weekly loop: canary test, run INSIDE the isolated step before every practice run, under the same user and limits.
Every probe below tries something the practice step must never be able to do. Every probe must FAIL.
If any probe succeeds, this exits 1: the run stops, no result is written, and the missed-week alarm fires.
  1. open the exam recipe and any exam outcome path (in the old checkout and in this folder)
  2. run git log, git cat-file and git show (here and in the old checkout)
  3. reach the internet: the exam sources, github.com, and plain addresses
  4. find a token or credential in the environment or in the usual credential files
"""
import argparse, glob, os, socket, subprocess, sys

ap = argparse.ArgumentParser(); ap.add_argument("--ws", required=True); a = ap.parse_args()
here = os.getcwd()
leaks = []


def must_fail(name, fn):
    try:
        ok = fn()
    except Exception:
        return
    if ok is not False:
        leaks.append(name)


def read(p):
    with open(p, "rb") as f:
        f.read(1)
    return True


# 1. exam recipe and exam outcomes
for base in (a.ws, os.path.join(a.ws, "src"), here):
    for rel in ("brink_outcomes.py", "cycle/exam_check.py", "run_round1.py", "data/outcomes/outcomes_long_sealed.csv.gz",
                "data/outcomes/targets_sealed.csv.gz", "data/raw/ged261-csv.zip", "data/clean/vdem_cy_v16.parquet"):
        must_fail(f"read {os.path.join(base, rel)}", lambda p=os.path.join(base, rel): read(p))
must_fail("any sealed or exam file in this folder",
          lambda: bool([p for p in glob.glob(os.path.join(here, "**", "*"), recursive=True)
                        if any(k in os.path.basename(p).lower() for k in ("sealed", "brink_outcomes", "exam_check", ".b64"))]) or False)

# 2. repository history
for cwd in (here, a.ws, os.path.join(a.ws, "src")):
    for cmd in (["git", "log", "-1"], ["git", "cat-file", "-p", "HEAD"], ["git", "show", "--stat", "HEAD"]):
        must_fail(f"{' '.join(cmd)} in {cwd}",
                  lambda c=cmd, d=cwd: subprocess.run(c, cwd=d, capture_output=True, timeout=20).returncode == 0 or False)

# 3. network
for host in ("ucdp.uu.se", "www.v-dem.net", "api.worldbank.org", "api.imf.org", "www.systemicpeace.org",
             "github.com", "api.github.com", "raw.githubusercontent.com", "pypi.org"):
    must_fail(f"resolve {host}", lambda h=host: bool(socket.getaddrinfo(h, 443)))
for ip in ("1.1.1.1", "8.8.8.8", "140.82.112.3", "185.199.108.133"):
    must_fail(f"connect {ip}:443", lambda i=ip: bool(socket.create_connection((i, 443), timeout=5)))

# 4. credentials
words = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL", "GITHUB_", "ACTIONS_", "RUNNER_")
must_fail("credential-like variables in the environment", lambda: bool([k for k in os.environ if any(w in k.upper() for w in words)]) or False)
for p in ("~/.git-credentials", "~/.config/gh/hosts.yml", "/home/runner/.git-credentials", "/home/runner/.config/gh/hosts.yml"):
    must_fail(f"read {p}", lambda q=os.path.expanduser(p): read(q))
for cfg in (os.path.join(a.ws, ".git", "config"), os.path.join(a.ws, "src", ".git", "config")):
    must_fail(f"read {cfg}", lambda q=cfg: read(q))

n = 0
if leaks:
    print("CANARY FAILED: these must be impossible but worked:\n  " + "\n  ".join(leaks))
    sys.exit(1)
print("canary: every probe was blocked (exam paths, git history, network, credentials)")
