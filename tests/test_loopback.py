#!/usr/bin/env python3
"""The Loopback read (`delta_attnres`) against the paper's equations, in both implementations (the training model in
trainer/ and looped_transformer.py), on CPU in fp32 with torch reference kernels:

  check_coefficients.py      with zero queries the coefficients recovered by least squares from the hidden states are the
                             decay prior c0 (the paper's schedule) at every read and the input coefficient is 1; with
                             learned queries the true coefficient of every stored update equals the tracked one at every
                             read, their sum is B_s and the input coefficient stays 1; iteration-end reads decay once;
                             beta = 0 with zero queries == Plain Loop bitwise.
  check_equivalence_init.py  the same Plain Loop equivalence on the released 191M configs (logits, loss and every shared
                             gradient bitwise), the (KR-1) x d query (11,520 entries at 191M), zero at init and on the
                             trainer's meta-device path, with a live gradient at zero.

    python tests/test_loopback.py            # runs both; exits non-zero on any FAIL
"""
import os
import subprocess
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loopback_audit")
rc = 0
for script in ("check_coefficients.py", "check_equivalence_init.py"):
    print(f"==== {script}", flush=True)
    out = subprocess.run([sys.executable, os.path.join(HERE, script)], capture_output=True, text=True)
    lines = [l for l in (out.stdout + out.stderr).splitlines() if l.startswith(("[PASS]", "[FAIL]", "    "))]
    print("\n".join(lines))
    n_fail = sum(l.startswith("[FAIL]") for l in lines)
    n_pass = sum(l.startswith("[PASS]") for l in lines)
    print(f"---- {script}: {n_pass} PASS, {n_fail} FAIL (rc {out.returncode})", flush=True)
    if out.returncode and not lines:
        print(out.stderr[-2000:])
    rc |= bool(n_fail or out.returncode)
print("ALL PASSED" if rc == 0 else "FAILED")
sys.exit(rc)
