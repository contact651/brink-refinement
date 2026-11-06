#!/usr/bin/env python3
"""
Fake-exam tests for the exam wall (no real exam data is built, read or touched; the real ledger is never written).
Runs the SAME reservation script the exam workflow runs (loop/exam_reserve_push.sh) against a throw-away git ledger.

  T1 binding        : only a file the weekly practice run KEPT, byte for byte, can be reserved; bad ids stop early
  T2 fourth refused : with 1 check already used in 2026 Q4 (as on the real record), checks 2 and 3 pass, the 4th is refused
  T3 quarter edge   : 31 Dec 23:59:59 UTC still counts as Q4 (refused); 1 Jan 00:00:00 UTC is a new quarter (check 1)
  T4 no refunds     : a crash after a reservation leaves it counted; logging the crash adds a line, never removes one;
                      every ledger state is the previous state plus new lines; the code has no way to rewrite the file
  T5 concurrency    : two requests from the same ledger state at the same moment: exactly one reservation lands,
                      the other push is refused and that request stops before building anything
  T6 binding check  : after reserving, a different code revision or an edited experiment file is refused;
                      the early dry-run check never writes anything
  T7 fingerprints   : matching files pass; one changed byte or a missing file fails closed
Exit 0 only if every test passes. Standard library + git only.
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile, threading

HERE = os.path.dirname(os.path.abspath(__file__))
PUSH = os.path.join(HERE, "exam_reserve_push.sh"); RES = os.path.join(HERE, "exam_reserve.py"); FPR = os.path.join(HERE, "exam_fingerprints.py")
COMMIT, OTHER = "a" * 40, "b" * 40
fails = []


def run(cmd, cwd=None, env=None):
    e = dict(os.environ, GIT_TERMINAL_PROMPT="0", **(env or {}))
    return subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True)


def ok(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info and not cond else ""))
    if not cond:
        fails.append(name)


def git(*a, cwd=None):
    r = run(["git", *a], cwd=cwd)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(a)}: {r.stderr}")
    return r.stdout.strip()


def sha(b):
    return hashlib.sha256(b).hexdigest()


T = tempfile.mkdtemp(prefix="examtest_")
try:
    remote = os.path.join(T, "remote.git"); git("init", "-q", "--bare", "-b", "ledger", remote)
    q = os.path.join(T, "queue"); os.makedirs(q)
    files = {"FAKE_KEEP": b'{"id": "FAKE_KEEP", "title": "fake kept idea", "added": "2026-11-06", "lens": "GLOBAL", "add_signals": ["x_own"]}\n',
             "FAKE_DROP": b'{"id": "FAKE_DROP", "title": "fake dropped idea", "added": "2026-11-06", "lens": "GLOBAL", "add_signals": ["x_cut"]}\n',
             "FAKE_EDITED": b'{"id": "FAKE_EDITED", "title": "fake idea edited after practice", "added": "2026-11-06", "lens": "GLOBAL", "add_signals": ["x_infl"]}\n'}
    for k, b in files.items():
        open(os.path.join(q, k + ".json"), "wb").write(b)
    practice = {"experiments": [{"id": "FAKE_KEEP", "sha256": sha(files["FAKE_KEEP"]), "verdict": "KEEP"},
                                {"id": "FAKE_DROP", "sha256": sha(files["FAKE_DROP"]), "verdict": "DROP"},
                                {"id": "FAKE_EDITED", "sha256": sha(b"the version that was tested"), "verdict": "KEEP"}]}
    seed = os.path.join(T, "seed"); git("init", "-q", "-b", "ledger", seed)
    open(os.path.join(seed, "results.jsonl"), "w").write(json.dumps(practice) + "\n")
    open(os.path.join(seed, "exam_ledger.jsonl"), "w").write("")
    git("add", "-A", cwd=seed); git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "seed", cwd=seed)
    git("remote", "add", "origin", remote, cwd=seed); git("push", "-q", "origin", "ledger", cwd=seed)
    prior = os.path.join(T, "prior.json"); json.dump([{"quarter": "2026Q4", "check_no": 1}], open(prior, "w"))
    n_clone = [0]

    def clone():
        n_clone[0] += 1; d = os.path.join(T, f"c{n_clone[0]}"); git("clone", "-q", "-b", "ledger", remote, d); return d

    def remote_ledger():
        d = clone(); s = open(os.path.join(d, "exam_ledger.jsonl")).read(); shutil.rmtree(d); return s

    history = [remote_ledger()]

    def attempt(exp, now, run_id, d=None, commit=COMMIT):
        d = d or clone()
        r = run(["bash", PUSH, d, prior, exp, q, commit, now], env={"GITHUB_RUN_ID": run_id, "GITHUB_RUN_ATTEMPT": "1"})
        return r.returncode, r.stdout + r.stderr, d

    def used():
        d = clone(); r = run([sys.executable, RES, "used", "--ledger", d, "--prior", prior]); shutil.rmtree(d)
        return json.loads(r.stdout)

    def appended_only():
        s = remote_ledger(); good = s.startswith(history[-1]); history.append(s); return good

    # T1 binding
    c, out, _ = attempt("FAKE_DROP", "2026-11-10T00:00:00Z", "101")
    ok("T1 a dropped idea cannot be reserved", c == 3 and "not kept" in out, out)
    c, out, _ = attempt("FAKE_EDITED", "2026-11-10T00:00:00Z", "102")
    ok("T1 an idea edited after its practice test cannot be reserved", c == 3 and "not kept" in out, out)
    c, out, _ = attempt("NO_SUCH_IDEA", "2026-11-10T00:00:00Z", "103")
    ok("T1 a missing idea file cannot be reserved", c == 3 and "no such" in out, out)
    c, out, _ = attempt("../../etc/passwd", "2026-11-10T00:00:00Z", "104")
    ok("T1 a path instead of an id stops before writing anything", c not in (0, 3) and "STOP" in out, out)
    ok("T1 refusals do not use up checks", used().get("2026Q4") == 1)
    ok("T1 ledger only grew", appended_only())

    # T2 fourth check refused
    c2, out2, _ = attempt("FAKE_KEEP", "2026-11-10T00:00:00Z", "201")
    c3, out3, _ = attempt("FAKE_KEEP", "2026-11-11T00:00:00Z", "202")
    c4, out4, _ = attempt("FAKE_KEEP", "2026-11-12T00:00:00Z", "203")
    ok("T2 check 2 of 3 reserved", c2 == 0 and "check 2 of 3" in out2, out2)
    ok("T2 check 3 of 3 reserved", c3 == 0 and "check 3 of 3" in out3, out3)
    ok("T2 a fourth check in the quarter is refused", c4 == 3 and "3 of 3" in out4, out4)
    ok("T2 counter shows 3 used in 2026 Q4", used().get("2026Q4") == 3)
    ok("T2 ledger only grew", appended_only())

    # T3 quarter boundary (UTC)
    c, out, _ = attempt("FAKE_KEEP", "2026-12-31T23:59:59Z", "301")
    ok("T3 31 Dec 23:59:59 UTC is still 2026 Q4 (refused)", c == 3 and "2026Q4" in out, out)
    c, out, _ = attempt("FAKE_KEEP", "2027-01-01T00:00:00Z", "302")
    ok("T3 1 Jan 00:00:00 UTC starts 2027 Q1 (check 1 of 3)", c == 0 and "check 1 of 3 for 2027Q1" in out, out)
    ok("T3 ledger only grew", appended_only())

    # T4 crash after reservation: no refund
    c, out, d = attempt("FAKE_KEEP", "2027-01-05T00:00:00Z", "401")
    ok("T4 reservation made (check 2 of 2027 Q1)", c == 0 and "check 2 of 3" in out, out)
    # the job 'crashes' here: nothing else happens; the reservation must still count
    ok("T4 a crashed check still counts", used().get("2027Q1") == 2)
    r = run([sys.executable, RES, "outcome", "--ledger", d, "--status", "failed after reservation (not refunded)", "--detail", "simulated crash"],
            env={"GITHUB_RUN_ID": "401", "GITHUB_RUN_ATTEMPT": "1"})
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "outcome", cwd=d); git("push", "-q", "origin", "HEAD:ledger", cwd=d)
    ok("T4 logging the crash does not give the check back", used().get("2027Q1") == 2)
    ok("T4 ledger only grew", appended_only())
    src = open(RES).read()
    ok("T4 the counter code can only append to the ledger", '"a", encoding' in src and '"w"' not in src and "truncate" not in src)
    r = run([sys.executable, RES, "refund", "--ledger", d])
    ok("T4 there is no refund command", r.returncode != 0)

    # T5 concurrency: two requests from the same ledger state at the same moment
    a_dir, b_dir = clone(), clone()
    res = {}
    def go(name, d):
        res[name] = attempt("FAKE_KEEP", "2027-04-02T00:00:00Z", "50" + name, d=d)
    th = [threading.Thread(target=go, args=(n, d)) for n, d in (("1", a_dir), ("2", b_dir))]
    [t.start() for t in th]; [t.join() for t in th]
    codes = sorted(v[0] for v in res.values())
    ok("T5 exactly one of two simultaneous requests reserves; the other stops at the push", codes == [0, 2], str(res))
    ok("T5 exactly one reservation landed in 2027 Q2", used().get("2027Q2") == 1)
    ok("T5 ledger only grew", appended_only())

    # T6 the reservation is bound to this file and this code revision
    c, out, d = attempt("FAKE_KEEP", "2027-04-03T00:00:00Z", "601")
    env = {"GITHUB_RUN_ID": "601", "GITHUB_RUN_ATTEMPT": "1"}
    r1 = run([sys.executable, RES, "verify", "--ledger", d, "--experiment", "FAKE_KEEP", "--queue", q, "--commit", COMMIT], env=env)
    r2 = run([sys.executable, RES, "verify", "--ledger", d, "--experiment", "FAKE_KEEP", "--queue", q, "--commit", OTHER], env=env)
    ok("T6 the reserved file and revision verify", r1.returncode == 0, r1.stdout + r1.stderr)
    ok("T6 a different code revision is refused", r2.returncode != 0)
    open(os.path.join(q, "FAKE_KEEP.json"), "ab").write(b" ")
    r3 = run([sys.executable, RES, "verify", "--ledger", d, "--experiment", "FAKE_KEEP", "--queue", q, "--commit", COMMIT], env=env)
    ok("T6 an experiment file edited after reserving is refused", r3.returncode != 0)
    r4 = run([sys.executable, RES, "verify", "--ledger", d, "--experiment", "FAKE_KEEP", "--queue", q, "--commit", COMMIT],
             env={"GITHUB_RUN_ID": "999", "GITHUB_RUN_ATTEMPT": "1"})
    ok("T6 a run without its own reservation is refused", r4.returncode != 0)

    # T6b a dry run (the early check before the long build) never writes
    before = remote_ledger(); d = clone()
    r = run([sys.executable, RES, "reserve", "--dry-run", "--ledger", d, "--prior", prior, "--experiment", "FAKE_DROP", "--queue", q, "--commit", COMMIT])
    r2 = run(["git", "status", "--porcelain"], cwd=d)
    ok("T6b a dry run refuses a dropped idea and writes nothing", r.returncode == 3 and r2.stdout.strip() == "", r.stdout + r2.stdout)
    open(os.path.join(q, "FAKE_KEEP.json"), "wb").write(files["FAKE_KEEP"])
    r = run([sys.executable, RES, "reserve", "--dry-run", "--ledger", d, "--prior", prior, "--experiment", "FAKE_KEEP", "--queue", q, "--commit", COMMIT,
             "--now", "2027-04-04T00:00:00Z"])
    r2 = run(["git", "status", "--porcelain"], cwd=d)
    ok("T6b a dry run says it would reserve and writes nothing", r.returncode == 0 and "WOULD RESERVE" in r.stdout and r2.stdout.strip() == "", r.stdout)
    ok("T6b the shared ledger is unchanged by dry runs", remote_ledger() == before)

    # T7 fingerprints fail closed
    fdir = os.path.join(T, "fp"); os.makedirs(fdir)
    fake = {"outcomes_long_sealed.csv.gz": b"fake sealed table", "targets_sealed.csv.gz": b"fake targets"}
    for k, b in fake.items():
        open(os.path.join(fdir, k), "wb").write(b)
    man = os.path.join(T, "manifest.json"); json.dump({"parts": {k: sha(b) for k, b in fake.items()}}, open(man, "w"))
    args = [f"{k}={os.path.join(fdir, k)}" for k in fake]
    r = run([sys.executable, FPR, "--manifest", man, "--out", os.path.join(T, "fp1.json"), *args])
    ok("T7 matching files pass", r.returncode == 0, r.stdout)
    open(os.path.join(fdir, "targets_sealed.csv.gz"), "ab").write(b"!")
    r = run([sys.executable, FPR, "--manifest", man, "--out", os.path.join(T, "fp2.json"), *args])
    ok("T7 one changed byte fails closed", r.returncode == 1 and "MISMATCH targets_sealed" in r.stdout and
       json.load(open(os.path.join(T, "fp2.json")))["all_match"] is False, r.stdout)
    os.remove(os.path.join(fdir, "outcomes_long_sealed.csv.gz"))
    r = run([sys.executable, FPR, "--manifest", man, "--out", os.path.join(T, "fp3.json"), *args])
    ok("T7 a missing file fails closed", r.returncode == 1)
    r = run([sys.executable, FPR, "--manifest", man, "--out", os.path.join(T, "fp4.json"), f"unpinned.csv={man}"])
    ok("T7 a file with no pinned fingerprint stops", r.returncode != 0)
finally:
    shutil.rmtree(T, ignore_errors=True)

print(f"\n{'ALL FAKE-EXAM TESTS PASSED' if not fails else 'FAILED: ' + '; '.join(fails)}")
sys.exit(1 if fails else 0)
