"""
P0.1 — behavioural check for the bridge's auth handling.

Run: `python3 scripts/check_bridge_auth.py scripts/nexus_bridge.py`

A standalone script rather than a pytest case because this repository has no
test runner yet; the Python test floor arrives with P0.3 (testcontainers +
pytest), and this should become a test case then rather than staying a script.

Extracts the functions by AST and executes them in an isolated namespace, so
nothing at module scope in nexus_bridge.py runs (it reads env, builds clients
and would reach the network).
"""
import ast
import io
import sys
import contextlib

PATH = sys.argv[1] if len(sys.argv) > 1 else "scripts/nexus_bridge.py"
src = ast.parse(open(PATH).read())

wanted = {"_report_auth_failure", "_require_agent_credential"}
funcs = {
    n.name: ast.get_source_segment(open(PATH).read(), n)
    for n in src.body
    if isinstance(n, ast.FunctionDef) and n.name in wanted
}

missing = wanted - funcs.keys()
assert not missing, f"functions not found: {missing}"

failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        failures.append(f"{label}: {detail}")


# ── _report_auth_failure ───────────────────────────────────────────────────
ns = {}
exec(funcs["_report_auth_failure"], ns)
report = ns["_report_auth_failure"]

for code in (401, 403):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rv = report(code, "a test call")
    out = buf.getvalue()
    check(f"{code} is reported as an auth failure", rv is True, f"returned {rv}")
    check(f"{code} names the env var", "NEXUS_AGENT_API_KEY" in out, "message omits it")
    check(f"{code} says it is a configuration fault", "CONFIGURATION" in out, "not stated")

for code in (200, 404, 500):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rv = report(code, "a test call")
    check(f"{code} is NOT reported as an auth failure", rv is False, f"returned {rv}")
    check(f"{code} prints nothing", buf.getvalue() == "", "printed noise")

# ── _require_agent_credential ──────────────────────────────────────────────
for key, should_exit in (("", True), ("   ", True), ("a-real-key", False)):
    ns = {"NEXUS_AGENT_API_KEY": key.strip()}
    exec(funcs["_require_agent_credential"], ns)
    require = ns["_require_agent_credential"]
    buf = io.StringIO()
    exited = False
    code = None
    try:
        with contextlib.redirect_stdout(buf):
            require()
    except SystemExit as e:
        exited, code = True, e.code
    label = f"key={key!r}"
    check(f"{label} → {'refuses to start' if should_exit else 'starts'}", exited == should_exit,
          f"exited={exited}")
    if should_exit:
        check(f"{label} exits non-zero", code == 1, f"exit code {code}")
        check(f"{label} names the variable", "NEXUS_AGENT_API_KEY" in buf.getvalue(), "not named")

print()
if failures:
    print(f"FAILED — {len(failures)}")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("OK — bridge additions behave as intended.")
