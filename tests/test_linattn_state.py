"""Tests for loop_linattn_state.py: the readers over the stored states (the paper's state interface).

The load-bearing test is `test_matches_materialised_state`. Every reader keeps its memory FACTORED,
S_r = sum_j v_j w_j^T with only the write vectors stored; factoring is an algebraic identity, but one
that is easy to get subtly wrong (transpose the transport, apply the gate on the wrong side of the
correction, drop a write coefficient) while still producing plausible numbers, so the factored kernel
is checked against a naive implementation that builds S explicitly at toy sizes.

The remaining tests pin the contracts a reader of the paper relies on: initialization equals Plain
Loop (and the dtype contract), the grid identities that make the readers comparable (`kda` with a
constant per-channel gate MUST equal `gdn`, `gdn` with the gate open MUST equal `deltanet`, ...),
the parameter rows and activations, and numerical hygiene at the edges.

    python tests/test_linattn_state.py
    python tests/test_linattn_state.py --compile  # also check torch.compile agrees with eager
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loop_linattn_state import (  # noqa: E402
    LINATTN_KINDS, LoopLinAttn,
    _LACT_NORM_EPS as LACT_EPS,
    _LACT_ETA_MAX as LACT_ETA_MAX,
    _RIDGE_MIN as RIDGE_MIN,
)

B, T, D, H, LOOPS = 2, 3, 32, 4, 3


def make(kind, seed=0):
    torch.manual_seed(seed)
    return LoopLinAttn(D, LOOPS, kind=kind, num_heads=H)


def states(n, scale=1.0, seed=1, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, T, D, generator=g, dtype=torch.float32).to(dtype) * scale
            for _ in range(n)]


def randomise_(mod, seed=2):
    """Move every parameter off its init, so the tests exercise real gate values.

    Deliberately modest magnitudes: the point is to leave the init, not to drive the
    sigmoids into saturation where every implementation agrees trivially.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in mod.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.7)
    return mod


# --------------------------------------------------------------------------------
# The reference: build S explicitly. O(dh^2) memory per (b, t, head) -- toy sizes only.
# --------------------------------------------------------------------------------

def naive_read(mod, st):
    n = len(st)
    hfs = [h.float() for h in st]
    ks = [mod._l2(mod._split(h)) for h in hfs]
    vs = [mod._split(h) for h in hfs]
    q = mod._l2(mod._split(hfs[-1]))
    b, t, nh, dh = q.shape
    S = torch.zeros(b, t, nh, dh, dh)
    eye = torch.eye(dh).view(1, 1, 1, dh, dh)

    for r in range(n):
        beta = mod._beta(r) if mod.uses_delta else None
        dt = mod._dt(r) if mod.kind == "mamba2" else None
        gate = mod._gate(r, dt=dt) if mod.decay is not None else None
        er = (mod._erase_dir(hfs, ks, r)
              if ((beta is not None or mod.additive_dplr) and r > 0) else ks[r])

        if r > 0:
            if mod.additive_dplr:
                # A_r = Diag(w) - ehat (rem * ehat)^T : diagonal PLUS rank one, which
                # is a different map from diagonal TIMES (I - rank one).
                rem = mod._removal(r)
                A = (torch.diag_embed(gate.expand(*gate.shape[:-1], dh))
                     - er.unsqueeze(-1) * (rem * er).unsqueeze(-2))
            else:
                # A_r = Diag(a) (I - beta ehat ehat^T)
                A = eye.expand(b, t, nh, dh, dh).clone()
                if beta is not None:
                    A = A - beta.unsqueeze(-1) * (er.unsqueeze(-1) * er.unsqueeze(-2))
                if gate is not None:
                    a = gate if gate.shape[-1] == dh else gate.expand(
                        *gate.shape[:-1], dh)
                    A = a.unsqueeze(-1) * A      # Diag(a) on the LEFT scales rows
            S = S @ A

        w = mod._write_vec(r, ks[r], beta=beta, dt=dt, gate=gate)
        S = S + vs[r].unsqueeze(-1) * w.unsqueeze(-2)

    return (S * q.unsqueeze(-2)).sum(-1).reshape(st[-1].shape)


def mesa_matrix_read(mod, st):
    """`mesa` by building the dh x dh solution, instead of the n x n Woodbury form.

    S = V^T Gamma K (K^T Gamma K + lambda I)^-1. This is the definition; `_read_mesa`
    is an identity away from it. Toy sizes only -- the whole point of the n x n form is
    that this object cannot exist at production shapes.
    """
    hfs = [h.float() for h in st]
    K = torch.stack([mod._l2(mod._split(h)) for h in hfs], dim=-2)      # (b,t,H,n,dh)
    V = torch.stack([mod._split(h) for h in hfs], dim=-2)
    q = mod._l2(mod._split(hfs[-1]))
    dh = q.shape[-1]
    # no gamma clamp, and the ridge FLOOR, exactly as the implementation has them
    gam = torch.sigmoid(torch.stack(
        [mod._head_scalar(mod.b_gamma, j).squeeze(-1)
         for j in range(len(st))], dim=-1)).expand(*V.shape[:-1])
    lam = (RIDGE_MIN + torch.nn.functional.softplus(
        mod.ridge.float())).view(1, 1, -1, 1, 1)
    G = K.transpose(-1, -2) @ (gam.unsqueeze(-1) * K)                   # (b,t,H,dh,dh)
    G = G + lam * torch.eye(dh).view(1, 1, 1, dh, dh)
    S = V.transpose(-1, -2) @ (gam.unsqueeze(-1) * K) @ torch.linalg.inv(G)
    return (S @ q.unsqueeze(-1)).squeeze(-1).reshape(st[-1].shape)


# --------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------

def lact_matrix_read(mod, st):
    """`lact` by materialising the dh x dh fast weight and stepping it explicitly.

    Written from the update rule -- accumulate the chunk gradient over all visible
    sources, take one step, Frobenius-normalise -- rather than from the factored
    identity, so it does not inherit the factorisation being checked.
    """
    hfs = [h.float() for h in st]
    K = [mod._l2(mod._split(h)) for h in hfs]
    V = [mod._split(h) for h in hfs]
    q = mod._l2(mod._split(hfs[-1]))
    b, t, nh, dh = q.shape
    W = torch.zeros(b, t, nh, dh, dh)                 # [value, key]
    for r in range(len(st)):
        g = torch.zeros_like(W)
        for j in range(r + 1):
            eta = torch.nn.functional.softplus(
                mod._head_scalar(mod.b_eta, j)).clamp(max=LACT_ETA_MAX)   # (..,H,1)
            resid = (W @ K[j].unsqueeze(-1)).squeeze(-1) - V[j]        # (b,t,H,dh)
            g = g + eta.unsqueeze(-1) * (resid.unsqueeze(-1) * K[j].unsqueeze(-2))
        W = W - g
        # the same zero-safe form as the implementation: eps INSIDE the sqrt. It matters
        # here because ||W|| can land within an order of magnitude of eps when eta
        # underflows; the convention itself is covered by
        # test_lact_zero_state_stays_finite_in_backward.
        nrm2 = W.flatten(-2).pow(2).sum(-1) + LACT_EPS ** 2
        W = W / nrm2.sqrt()[..., None, None]
    return (W @ q.unsqueeze(-1)).squeeze(-1).reshape(st[-1].shape)


def test_lact_zero_state_stays_finite_in_backward():
    """An all-zero history must not NaN the backward through LaCT's normalisation.

    d/dx sqrt(x) is infinite at 0, so putting eps after the sqrt guards the division
    and not the gradient: an all-zero state would give a finite forward -- and, at
    alpha=0, a forward that looks like a clean init-equivalence pass -- while b_eta
    and every input receive NaN.
    """
    for scale in (0.0, 1e-20):
        mod = make("lact")
        # build the leaf AFTER the offset, or `.grad` lands on the wrong node
        st = [(torch.zeros(B, T, D) + scale).requires_grad_(True)
              for _ in range(LOOPS + 1)]
        out = mod.combine(st)
        assert torch.isfinite(out).all(), f"lact scale={scale}: forward"
        out.sum().backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), \
                f"lact scale={scale}: {name} grad is not finite"
        for i, x in enumerate(st):
            assert torch.isfinite(x.grad).all(), \
                f"lact scale={scale}: input {i} grad is not finite"
    print("PASS lact survives an all-zero history in the backward pass")


def test_lact_matches_the_materialised_fast_weight():
    """`lact`'s factored chunk step == the same step taken on an explicit dh x dh W."""
    mod = randomise_(make("lact"))
    for n in range(1, LOOPS + 2):
        for scale in (1.0, 300.0):
            st = states(n, scale=scale)
            torch.testing.assert_close(
                mod._read(st), lact_matrix_read(mod, st), rtol=2e-4, atol=2e-4,
                msg=lambda m, n=n, s=scale:
                    f"lact n={n} scale={s}: factored != materialised\n{m}")
    print("PASS lact: the factored chunk step == the explicit fast-weight step")


def _lact_chunk_of_one(mod, st):
    """The same update restricted to the newest source -- LaCT's chunk size 1."""
    hfs = [h.float() for h in st]
    K = [mod._l2(mod._split(h)) for h in hfs]
    V = [mod._split(h) for h in hfs]
    q = mod._l2(mod._split(hfs[-1]))
    b, t, nh, dh = q.shape
    W = torch.zeros(b, t, nh, dh, dh)
    for r in range(len(st)):
        eta = torch.nn.functional.softplus(
            mod._head_scalar(mod.b_eta, r)).clamp(max=LACT_ETA_MAX)
        resid = (W @ K[r].unsqueeze(-1)).squeeze(-1) - V[r]
        W = W - eta.unsqueeze(-1) * (resid.unsqueeze(-1) * K[r].unsqueeze(-2))
        nrm2 = W.flatten(-2).pow(2).sum(-1) + LACT_EPS ** 2
        W = W / nrm2.sqrt()[..., None, None]
    return (W @ q.unsqueeze(-1)).squeeze(-1).reshape(st[-1].shape)


def test_lact_chunk_size_changes_the_answer():
    """chunk = whole history is genuinely different from chunk = newest source.

    The LaCT-inspired reader's distinguishing feature is the step over the whole
    stored prefix, so the arm is worthless if it silently behaves like a chunk of 1.
    Measured across collinearity levels of the sources, and relative to the output's
    own scale (the output of a Frobenius-normalised W is ~1 while the input states
    are ~300; a ratio against the input would always look negligible).
    """
    mod = randomise_(make("lact"))
    seen = {}
    for pert, label in ((0.3, "0.95"), (0.7, "0.78"), (2.0, "0.35")):
        g = torch.Generator().manual_seed(11)
        base = torch.randn(B, T, D, generator=g) * 300.0
        st = [base + pert * 300.0 * torch.randn(B, T, D, generator=g)
              for _ in range(LOOPS + 1)]
        f = torch.stack([x.reshape(-1) for x in st])
        f = f / f.norm(dim=-1, keepdim=True)
        cos = float((f[0] * f[1]).sum())
        allc = mod._read(st).detach()
        one = _lact_chunk_of_one(mod, st).detach()
        rel = float((allc - one).abs().max()) / float(allc.abs().max())
        seen[label] = (cos, rel)
        assert rel > 0.05, (
            f"at cos {cos:.2f}, chunk-of-all and chunk-of-one differ by only "
            f"{rel:.3e} relative to the output -- the aggregation is not happening")
    body = ", ".join(f"cos {c:.2f} -> {r:.2f}" for c, r in seen.values())
    print(f"PASS lact's chunk really is the whole history ({body})")


def test_lact_global_eta_scale_is_weak_only_at_the_stream_scale():
    """A COMMON eta scale barely moves `lact` at the observed stream norm -- only there.

    Varying one common scale cannot distinguish "this parameter does nothing" from
    "this parameter's overall magnitude does nothing". The insensitivity is not
    structural: after the first retraction the update is W - c*g(W, V), whose
    direction depends on the common scale c, and normalising afterwards cannot undo a
    direction change. It is an artefact of the residual stream's norm (~300) making
    the write term dominate the carried one -- at unit scale the same sweep moves the
    read a great deal, and tilting eta ACROSS sources moves it at either scale.
    """
    seen = {}
    for scale in (1.0, 300.0):
        stx = states(LOOPS + 1, scale=scale)
        outs = []
        for raw in (-3.0, 0.0, 1.0, 3.0):      # same value everywhere: pure scale
            mod = make("lact")
            with torch.no_grad():
                mod.b_eta.fill_(raw)
            outs.append(mod._read(stx).detach())
        ref = outs[0]
        seen[scale] = max(float((o - ref).abs().max())
                          for o in outs[1:]) / float(ref.abs().max())
    assert seen[300.0] < 0.05, (
        f"at the stream scale a common eta no longer washes out ({seen[300.0]:.3f}); "
        f"the docstring is stale")
    assert seen[1.0] > 0.2, (
        f"at unit scale a common eta only moves the read {seen[1.0]:.3f}; if it is now "
        f"small everywhere, the insensitivity may be structural after all")

    st = states(LOOPS + 1, scale=300.0)

    # ... and the control: relative changes DO move it, so the parameters are not inert
    mod = make("lact")
    with torch.no_grad():
        mod.b_eta.zero_()
    flat = mod._read(st).detach()
    with torch.no_grad():
        mod.b_eta.copy_(torch.linspace(-2.0, 2.0, mod.b_eta.numel()).view_as(mod.b_eta))
    tilted = mod._read(st).detach()
    rel_spread = float((tilted - flat).abs().max()) / float(flat.abs().max())
    assert rel_spread > 0.05, (
        f"tilting eta across sources only moves the read by {rel_spread:.3e}; then "
        f"`lact` really would be a zero-parameter reader")
    body = ", ".join(f"input scale {s:g}: {v:.1%}" for s, v in seen.items())
    print(f"PASS lact: a common eta scale is weak ONLY at the stream scale ({body}); "
          f"tilting eta across sources moves the read {rel_spread:.0%}, so the "
          f"parameters are live and the weakness is not structural")


def test_mesa_matches_the_dense_solution():
    """`mesa`'s n x n Woodbury form == the dh x dh ridge-regression solution."""
    mod = randomise_(make("mesa"))
    for n in range(1, LOOPS + 2):
        for scale in (1.0, 300.0):
            st = states(n, scale=scale)
            torch.testing.assert_close(
                mod._read(st), mesa_matrix_read(mod, st),
                rtol=2e-4, atol=2e-4 * max(scale, 1.0),
                msg=lambda m, n=n, s=scale: f"mesa n={n} scale={s}: "
                                            f"n x n form != dense solution\n{m}")
    # the corner the randomisation never reaches: saturated gamma, floored ridge,
    # identical (perfectly collinear) sources
    deg = make("mesa")
    with torch.no_grad():
        deg.ridge.fill_(-20.0)
        deg.b_gamma.fill_(-20.0)
    same = [torch.ones(B, T, D) * 300.0 for _ in range(LOOPS + 1)]
    got, ref = deg._read(same), mesa_matrix_read(deg, same)
    assert torch.isfinite(got).all(), "mesa: degenerate corner is not finite"
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3,
                               msg="mesa: reference and implementation disagree at "
                                   "saturated gamma / floored ridge / collinear keys")
    # TRUE saturation, and the BACKWARD. sigmoid(-20) is 2e-9 -- small, not zero. At
    # -100 it underflows to exactly 0 in fp32, and building sqrt(gamma) as
    # sqrt(sigmoid(x)) then has an infinite derivative there: finite forward, NaN
    # gradients.
    for raw in (-100.0, -1e38):
        sat = make("mesa")
        with torch.no_grad():
            sat.alpha.fill_(0.5)
            sat.b_gamma.fill_(raw)
        out = sat.combine(states(LOOPS + 1, scale=300.0))
        assert torch.isfinite(out).all(), f"mesa b_gamma={raw}: forward"
        out.square().sum().backward()
        for name, p in sat.named_parameters():
            assert torch.isfinite(p.grad).all(), \
                f"mesa b_gamma={raw}: {name} grad is not finite"
    print("PASS mesa: the n x n Woodbury read == the dh x dh ridge solution (floored "
          "ridge, identical sources), and a fully saturated gamma keeps the BACKWARD "
          "finite")


def test_hgrn2_gate_telescopes():
    """hgrn2's KEY-CHANNEL weights are non-negative and telescope to 1 - prod(a).

    The tied input/forget gate buys per-CHANNEL leaky integration: every w_j is
    elementwise non-negative and, along each key channel, the weights over the history
    telescope to 1 - prod_r a_r. It is NOT a convex combination of the sources -- the
    source coefficient c_j = w_j . q is signed, because q is a signed vector. The
    weights are computed here from the closed form, prod_{r>j} a_r times (1 - a_j),
    from the gates alone, and the module's read is then checked against them, so
    nothing in the module's transition loop is assumed.
    """
    mod = randomise_(make("hgrn2"))
    st = states(LOOPS + 1, scale=300.0)
    hfs = [h.float() for h in st]
    n = len(st)
    a = [mod._gate(j) for j in range(n)]                        # (1,1,H,dh) each

    w = []
    for j in range(n):
        keep = torch.ones_like(a[j])
        for r in range(j + 1, n):
            keep = keep * a[r]
        w.append(keep * (1.0 - a[j]))

    stacked = torch.stack(w, 0)
    assert float(stacked.detach().min()) >= 0.0, \
        f"hgrn2 key weights must be non-negative, got {float(stacked.min()):.3e}"
    prod = torch.ones_like(a[0])
    for r in range(n):
        prod = prod * a[r]
    torch.testing.assert_close(stacked.sum(0), 1.0 - prod, rtol=1e-5, atol=1e-6,
                               msg="hgrn2 key weights do not telescope to 1 - prod(a)")

    # and the module reads exactly those coefficients back
    V = torch.stack([mod._split(h) for h in hfs], 0)
    q = mod._l2(mod._split(hfs[-1]))
    u = ((stacked * q.unsqueeze(0)).sum(-1, keepdim=True) * V).sum(0)
    torch.testing.assert_close(u.reshape(st[-1].shape), mod._read(st),
                               rtol=2e-4, atol=6e-2,
                               msg="hgrn2's read does not match its closed-form weights")

    # the SOURCE coefficients are signed, like every other mode here
    c = (stacked * q.unsqueeze(0)).sum(-1)
    assert float(c.min()) < 0.0, \
        "hgrn2 source coefficients came out non-negative on signed states; if that " \
        "has become true by construction, the docstring must say so"
    print(f"PASS hgrn2: key weights non-negative and telescope to 1-prod(a); source "
          f"coefficients are signed (min {float(c.min()):.3f}), as for every mode here")


def test_rwkv7_transition_is_additive_dplr():
    """rwkv7's transition is Diag(w) - khat (a*khat)^T, not Diag(w)(I - beta khat khat^T).

    Checked against a hand-computed 2x2 case rather than against anything built from the
    module's own helpers, because the two forms are both "a diagonal and a rank-one" and
    the materialised-state test shares `_removal`/`_erase_dir` with the implementation.
    """
    g = torch.tensor([0.8, 0.6])
    e = torch.tensor([1.0, 1.0]) / math.sqrt(2.0)
    rem = torch.tensor([0.2, 0.9])
    A = torch.diag(g) - torch.outer(e, rem * e)
    expect = torch.tensor([[0.70, -0.45], [-0.10, 0.15]])
    torch.testing.assert_close(A, expect, rtol=1e-4, atol=1e-4,
                               msg="the hand-computed reference itself is wrong")

    # A^T u for u = (1, 0) must be A's first ROW; the implementation computes
    # g * u - (rem * e) (e . u)
    u = torch.tensor([1.0, 0.0])
    got = g * u - (rem * e) * (e * u).sum()
    torch.testing.assert_close(got, A[0], rtol=1e-4, atol=1e-4,
                               msg="the additive-DPLR key-side update is not A^T")

    # and it is genuinely different from the multiplicative form kda uses
    beta = 0.5
    mult = (torch.eye(2) - beta * torch.outer(e, e)) @ torch.diag(g)
    gap = float((mult - A).abs().max())
    assert gap > 0.1, f"additive and multiplicative DPLR differ by only {gap:.3e}"
    print(f"PASS rwkv7 is additive DPLR (Diag(w) - khat (a*khat)^T), which differs "
          f"from kda's multiplicative form by {gap:.2f}")


def test_parameterisation_contract():
    """Hand-written expectations for every mode's gates: which ROW, which activation.

    THIS IS THE GAP THE OTHER TESTS LEAVE. Every reference in this file except
    `kda_paper_read` and the rwkv7 2x2 case builds its transition by calling
    `_beta`/`_gate`/`_write_vec`/`_removal`, so a bug in the PARAMETERISATION -- the
    wrong row, the wrong activation, a sign -- moves both sides together and is
    invisible to them. Everything below is computed by hand from the parameter values,
    with no module helper on the right-hand side.

    The row conventions being pinned:
      * beta, eta, gamma      -> row = SOURCE j          (they act when j is written)
      * decay gates           -> row = TRANSITION j-1    (there is one fewer)
      * mamba2 dt, hgrn2 gate -> row = SOURCE j          (they also appear in the write)
    """
    sig = torch.sigmoid
    sp = torch.nn.functional.softplus
    h = states(1, scale=300.0)[0].float()
    k = make("gdn")._l2(make("gdn")._split(h))        # a unit key, (B, T, H, dh)
    rows, grows = LOOPS + 1, LOOPS

    def rowvals(shape, lo=-1.5, hi=1.5):
        return torch.linspace(lo, hi, int(torch.tensor(shape).prod())).view(shape)

    # ---- beta: source row, sigmoid, and it is also the write coefficient ----
    for kind in ("deltanet", "gdn", "kda"):
        m = make(kind)
        with torch.no_grad():
            m.b_beta.copy_(rowvals(m.b_beta.shape))
        for j in (0, 1, rows - 1):
            want = sig(m.b_beta[j]).view(1, 1, -1, 1)
            torch.testing.assert_close(m._beta(j), want, rtol=0, atol=0,
                                       msg=f"{kind}: beta row {j}")
            torch.testing.assert_close(m._write_vec(j, k), want * k, rtol=0, atol=0,
                                       msg=f"{kind}: write != beta k at source {j}")

    # ---- decay gates: TRANSITION row (j-1), sigmoid ----
    for kind, per_channel in (("gdn", False), ("gla", True), ("kda", True)):
        m = make(kind)
        with torch.no_grad():
            m.b_gate.copy_(rowvals(m.b_gate.shape))
        assert m.b_gate.shape[0] == grows, \
            f"{kind}: gate has {m.b_gate.shape[0]} rows, want {grows} (one per transition)"
        for j in (1, 2, grows):
            want = sig(m.b_gate[min(j - 1, grows - 1)])
            want = want.view(1, 1, *want.shape) if per_channel else want.view(1, 1, -1, 1)
            torch.testing.assert_close(m._gate(j), want, rtol=0, atol=0,
                                       msg=f"{kind}: gate at source {j} is not row {j-1}")

    # ---- mamba2: SOURCE row; dt = softplus, decay = exp(-dt e^{A_log}) ----
    m = make("mamba2")
    with torch.no_grad():
        m.dt_bias.copy_(rowvals(m.dt_bias.shape, -1.0, 1.0))
        m.A_log.copy_(torch.linspace(-0.5, 0.5, H))
    assert m.dt_bias.shape[0] == rows, "mamba2 dt must have one row per SOURCE"
    for j in (0, 1, rows - 1):
        dt = sp(m.dt_bias[j]).view(1, 1, -1, 1)
        torch.testing.assert_close(m._dt(j), dt, rtol=0, atol=0)
        torch.testing.assert_close(m._write_vec(j, k), dt * k, rtol=0, atol=0,
                                   msg="mamba2 write must be the same dt as the decay")
        want = torch.exp(-dt * torch.exp(m.A_log).view(1, 1, -1, 1))
        torch.testing.assert_close(m._gate(j), want, rtol=1e-6, atol=1e-7,
                                   msg=f"mamba2 decay at source {j}")

    # ---- hgrn2: SOURCE row, and the write is 1 - gate ----
    m = make("hgrn2")
    with torch.no_grad():
        m.b_gate.copy_(rowvals(m.b_gate.shape))
    assert m.b_gate.shape[0] == rows, \
        "hgrn2's gate is also its write, so it needs one row per SOURCE"
    for j in (0, 1, rows - 1):
        g = sig(m.b_gate[j]).view(1, 1, *m.b_gate.shape[1:])
        torch.testing.assert_close(m._gate(j), g, rtol=0, atol=0)
        torch.testing.assert_close(m._write_vec(j, torch.zeros_like(g), gate=g),
                                   1.0 - g, rtol=0, atol=0,
                                   msg="hgrn2's write vector must be 1 - gate")

    # ---- retnet: frozen buffer, not a parameter, RetNet's own schedule ----
    m = make("retnet")
    assert "gamma" in dict(m.named_buffers()) and "gamma" not in dict(m.named_parameters())
    want = 1.0 - torch.pow(2.0, -5.0 - torch.arange(H, dtype=torch.float64)).float()
    torch.testing.assert_close(m.gamma.flatten(), want, rtol=0, atol=0,
                               msg="retnet gamma is not 1 - 2^-(5+h)")

    # ---- rwkv7: removal on the TRANSITION row, sigmoid, per channel ----
    m = make("rwkv7")
    with torch.no_grad():
        m.b_removal.copy_(rowvals(m.b_removal.shape))
    assert m.b_removal.shape[0] == grows, "rwkv7 removal acts on transitions"
    for j in (1, 2, grows):
        want = sig(m.b_removal[min(j - 1, grows - 1)])
        torch.testing.assert_close(m._removal(j),
                                   want.view(1, 1, *want.shape), rtol=0, atol=0,
                                   msg=f"rwkv7 removal at source {j} is not row {j-1}")
    assert torch.equal(m._write_vec(2, k), k), "rwkv7 writes v k^T with coefficient 1"

    # ---- mesa: gamma on the SOURCE row; the ridge carries a floor ----
    m = make("mesa")
    assert m.b_gamma.shape[0] == rows
    with torch.no_grad():
        m.ridge.fill_(-30.0)                 # softplus(-30) underflows to ~0
    lam = RIDGE_MIN + float(sp(m.ridge.detach().float())[0])
    assert lam >= RIDGE_MIN, f"the ridge floor is not applied: lambda = {lam}"

    # ---- lact: eta on the SOURCE row, softplus (not sigmoid: it is a step size) ----
    m = make("lact")
    assert m.b_eta.shape[0] == rows
    with torch.no_grad():
        m.b_eta.copy_(rowvals(m.b_eta.shape, -2.0, 2.0))
    for j in (0, rows - 1):
        want = sp(m.b_eta[j]).view(1, 1, -1, 1)
        got = sp(m._head_scalar(m.b_eta, j))
        torch.testing.assert_close(got, want, rtol=0, atol=0, msg=f"lact eta row {j}")

    # ---- linattn: writes with coefficient 1 and has no gate at all ----
    m = make("linattn")
    assert m._gate(1) is None, "linattn must have no decay"
    assert torch.equal(m._write_vec(1, k), k), "linattn writes v k^T with coefficient 1"

    # ---- key/query normalisation: exact unit norm (clamp-min, not additive eps) ----
    m = make("gdn")
    # a small-but-above-eps vector: clamp-min gives exactly 1, an additive eps gives
    # ||t||/(||t||+eps) = 0.999. That 1e-3 gap is the whole difference between
    # I - k k^T being a projection and merely resembling one.
    small = torch.tensor([[[1e-3] + [0.0] * (D - 1)]])
    n_small = float(m._l2(m._split(small)).norm(dim=-1).flatten()[0])
    assert abs(n_small - 1.0) < 1e-6, (
        f"||khat|| = {n_small:.6f} for a 1e-3 vector, not 1.0 -- the additive-eps form "
        f"is back and I - k k^T is no longer a projection")
    n_big = float(m._l2(m._split(states(1, scale=300.0)[0].float()))
                  .norm(dim=-1).flatten()[0])
    assert abs(n_big - 1.0) < 1e-6, f"||khat|| = {n_big:.6f} at production scale"
    # and BELOW eps the clamp deliberately does not normalise, which is what keeps a
    # zero state from producing 0/0
    tiny = torch.tensor([[[5e-7] + [0.0] * (D - 1)]])
    n_tiny = float(m._l2(m._split(tiny)).norm(dim=-1).flatten()[0])
    assert n_tiny < 1.0, f"the eps clamp is not engaging below eps (got {n_tiny})"

    # ---- erase direction: the write key everywhere except rwkv7, which uses the
    #      normalised difference of adjacent stored states h_r - h_{r-1} ----
    hs = [h * 1.0, h * 2.0, h * 2.0 + 7.0]
    hf3 = [z.float() for z in hs]
    ks3 = [make("gdn")._l2(make("gdn")._split(z)) for z in hf3]
    m = make("kda")
    torch.testing.assert_close(m._erase_dir(hf3, ks3, 2), ks3[2], rtol=0, atol=0,
                               msg="kda must erase along its own write key")
    m = make("rwkv7")
    want = m._l2(m._split(hf3[2] - hf3[1]))
    torch.testing.assert_close(m._erase_dir(hf3, ks3, 2), want, rtol=0, atol=0,
                               msg="rwkv7 must erase along the state difference, not the key")

    # ---- mesa gamma: SOURCE row, and sqrt(gamma) built stably from the logit ----
    m = make("mesa")
    with torch.no_grad():
        m.b_gamma.copy_(rowvals(m.b_gamma.shape, -3.0, 3.0))
    for j in (0, rows - 1):
        want = sig(m.b_gamma[j])
        got = sig(m._head_scalar(m.b_gamma, j).squeeze(-1)).squeeze()
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7,
                                   msg=f"mesa gamma row {j}")

    # ---- rwkv7 decay: TRANSITION row, sigmoid, per channel (its own tensor) ----
    m = make("rwkv7")
    with torch.no_grad():
        m.b_gate.copy_(rowvals(m.b_gate.shape))
    assert m.b_gate.shape[0] == grows, "rwkv7 decay acts on transitions"
    for j in (1, grows):
        want = sig(m.b_gate[min(j - 1, grows - 1)])
        torch.testing.assert_close(m._gate(j), want.view(1, 1, *want.shape),
                                   rtol=0, atol=0,
                                   msg=f"rwkv7 decay at source {j} is not row {j-1}")

    # ---- rows beyond the constructed depth CLAMP to the last one, not wrap ----
    m = make("gdn")
    with torch.no_grad():
        m.b_beta.copy_(rowvals(m.b_beta.shape))
    last = sig(m.b_beta[rows - 1]).view(1, 1, -1, 1)
    for j in (rows, rows + 5):
        torch.testing.assert_close(m._beta(j), last, rtol=0, atol=0,
                                   msg=f"row {j} beyond the constructed depth does not "
                                       f"clamp to the last row (wrapping?)")

    print("PASS parameterisation contract: row indices, activations, key normalisation "
          "and erase directions match hand-written values")


def test_parameter_effectiveness():
    """How many of each mode's NOMINAL parameters actually move the output?

    A tensor counts as EFFECTIVE if perturbing it by +-1.0 in raw (pre-activation)
    space moves the read by more than 1% of the read's own scale, in the best of four
    directions (a uniform +-1 probes only the all-ones direction, which can be a
    near-null direction of a tensor that is perfectly live otherwise: `lact.b_eta`
    barely moves uniformly, because LaCT's normalisation divides a common step size
    back out, and moves the read a great deal when tilted across sources). The
    expected-inert set is pinned: anything joining it is a finding, not a pass.
    """
    INERT_EXPECTED = set()          # nothing is inert
    st = states(LOOPS + 1, scale=300.0)
    rows = []
    inert = set()
    for kind in LINATTN_KINDS:
        mod = randomise_(make(kind))
        base = mod._read(st).detach()
        scale = float(base.abs().max())
        for name, p in mod.named_parameters():
            if name == "alpha":
                continue                  # alpha is applied in combine(), not _read
            tilt = torch.linspace(-1.0, 1.0, p.numel()).view_as(p)
            best = 0.0
            for delta in (torch.ones_like(p), -torch.ones_like(p), tilt, -tilt):
                with torch.no_grad():
                    keep = p.detach().clone()
                    p.add_(delta)
                    moved = mod._read(st).detach()
                    p.copy_(keep)
                best = max(best, float((moved - base).abs().max()))
            rel = best / max(scale, 1e-12)
            rows.append((kind, name, p.numel(), rel))
            if rel <= 0.01:
                inert.add((kind, name.split(".")[0]))

    assert inert == INERT_EXPECTED, (
        f"the set of inert parameter tensors changed.\n"
        f"  now inert but not expected: {sorted(inert - INERT_EXPECTED)}\n"
        f"  expected inert but is live: {sorted(INERT_EXPECTED - inert)}\n"
        f"An inert tensor is a parameter count that lies.")

    for kind in LINATTN_KINDS:
        live = [r for r in rows if r[0] == kind and r[3] > 0.01]
        assert live or kind in ("linattn", "retnet"), \
            f"{kind}: no parameter tensor moves the output at all"

    n_inert = sum(1 for r in rows if r[3] <= 0.01)
    worst = min((r for r in rows if r[3] > 0.01), key=lambda r: r[3])
    print(f"PASS parameter effectiveness: {len(rows) - n_inert}/{len(rows)} tensors "
          f"move the read (weakest live one: {worst[0]}.{worst[1]} at {worst[3]:.1%}); "
          f"inert = {sorted(inert)}  [effect-size smoke, not a degrees-of-freedom oracle]")


def test_matches_materialised_state():
    """The factored kernel == the explicit dh x dh state, for every mode and depth."""
    for kind in LINATTN_KINDS:
        if kind in ("mesa", "lact"):
            continue          # not DPLR recurrences; they have their own references
        mod = randomise_(make(kind))
        for n in range(1, LOOPS + 2):
            for scale in (1.0, 300.0):     # 300 ~ the real residual-stream norm
                st = states(n, scale=scale)
                got = mod._read(st)
                ref = naive_read(mod, st)
                torch.testing.assert_close(
                    got, ref, rtol=2e-4, atol=2e-4 * max(scale, 1.0),
                    msg=lambda m, k=kind, n=n, s=scale:
                        f"{k} n={n} scale={s}: factored != materialised\n{m}")
    print(f"PASS factored == materialised ({len(LINATTN_KINDS)} kinds x depths x scales)")


def test_init_equivalence():
    """alpha init 0 => combine() returns states[-1], in bf16 and fp32: every reader
    starts as Plain Loop.

    "Bitwise" would be too strong and the exception is worth pinning rather than
    papering over: the blend is arithmetic, so a -0.0 in the stream comes back as +0.0.
    `torch.equal` treats those as equal (and so does everything downstream here), so the
    contract is checked with `torch.equal` for ordinary values and the signed-zero case
    is checked separately, at the bit level, and asserted to be exactly that one
    difference.
    """
    for kind in LINATTN_KINDS:
        mod = make(kind)
        for dtype in (torch.float32, torch.bfloat16):
            for n in range(1, LOOPS + 2):
                st = states(n, scale=300.0, dtype=dtype)
                out = mod.combine(st)
                assert out.dtype == dtype, f"{kind}: {out.dtype} != {dtype}"
                assert torch.equal(out, st[-1]), \
                    f"{kind} {dtype} n={n}: not identity at init"

    # the one documented exception, pinned at the bit level
    mod = make("kda")
    neg = torch.tensor([[[-0.0, 1.0, 2.0, 3.0] * (D // 4)]])
    out = mod.combine([neg, neg])
    assert torch.equal(out, neg), "signed zero should still compare equal"
    zero = neg == 0
    assert torch.equal(out.view(torch.int32)[~zero], neg.view(torch.int32)[~zero]), \
        "every NONZERO value must be bit-identical at init"
    assert bool(torch.signbit(neg[zero]).all()) and not bool(torch.signbit(out[zero]).any()), \
        "the -0.0 -> +0.0 exception no longer happens; tighten the docstring to bitwise"
    print("PASS init-equivalence holds (fp32 + bf16); -0.0 -> +0.0 is the only bit "
          "difference, and the dtype contract holds")


def test_grid_collapses():
    """Each reader must reduce to its neighbour when its extra gate is removed.

    These are the identities that make a comparison across the readers an isolated
    variable. `kda` differing from `gdn` under a CONSTANT per-channel gate would mean
    the two differ in something other than gate granularity.
    """
    st = states(LOOPS + 1, scale=300.0)
    BIG = 30.0        # sigmoid(30) == 1.0 in fp32; the gate is off

    def read(kind, mut):
        mod = make(kind)
        with torch.no_grad():
            mut(mod)
        return mod._read(st)

    # kda (diagonal gate, all channels equal) == gdn (scalar gate), same beta.
    def kda_flat(m):
        m.b_beta.copy_(torch.linspace(-1.0, 1.0, m.b_beta.numel()).view_as(m.b_beta))
        m.b_gate.copy_(torch.full_like(m.b_gate, 0.3))

    def gdn_flat(m):
        m.b_beta.copy_(torch.linspace(-1.0, 1.0, m.b_beta.numel()).view_as(m.b_beta))
        m.b_gate.copy_(torch.full_like(m.b_gate, 0.3))

    torch.testing.assert_close(read("kda", kda_flat), read("gdn", gdn_flat),
                               rtol=1e-5, atol=1e-3,
                               msg="kda with a constant channel gate != gdn")

    # gdn with the gate saturated open == deltanet, same beta.
    def gdn_open(m):
        m.b_beta.copy_(torch.linspace(-1.0, 1.0, m.b_beta.numel()).view_as(m.b_beta))
        m.b_gate.fill_(BIG)

    def dn(m):
        m.b_beta.copy_(torch.linspace(-1.0, 1.0, m.b_beta.numel()).view_as(m.b_beta))

    torch.testing.assert_close(read("gdn", gdn_open), read("deltanet", dn),
                               rtol=1e-5, atol=1e-3,
                               msg="gdn with the gate open != deltanet")

    # gla with the gate saturated open == linattn (no decay, no delta).
    torch.testing.assert_close(read("gla", lambda m: m.b_gate.fill_(BIG)),
                               read("linattn", lambda m: None),
                               rtol=1e-5, atol=1e-3,
                               msg="gla with the gate open != linattn")

    # retnet's frozen gammas are 1 - 2^-(5+h) in (0.968, 0.999): over an R-step axis
    # that is nearly no decay.
    gam = make("retnet").gamma.flatten()
    assert torch.all(gam > 0.96) and torch.all(gam < 1.0), gam
    assert float(gam.pow(LOOPS).min()) > 0.90, \
        "retnet's total decay over the loop axis should be < 10%"
    print("PASS grid collapses: kda->gdn->deltanet, gla->linattn, retnet is ~undecayed")


def test_batch_invariance():
    """combine() on a batch == combine() per sample. No cross-position/sample state."""
    for kind in LINATTN_KINDS:
        mod = randomise_(make(kind))
        st = states(LOOPS + 1, scale=300.0)
        full = mod.combine(st)
        for i in range(B):
            one = mod.combine([h[i:i + 1] for h in st])
            torch.testing.assert_close(full[i:i + 1], one, rtol=1e-5, atol=1e-3,
                                       msg=f"{kind}: batch element {i} differs")
    print("PASS batch invariance (per-sample == batched) for every kind")


def test_depth_beyond_construction():
    """A history longer than the constructed depth reuses the last parameter row
    (clamped, not wrapped), and the factored read still matches the reference."""
    for kind in LINATTN_KINDS:
        mod = randomise_(make(kind))
        st = states(LOOPS + 5, scale=300.0)      # deeper than the constructed depth
        got = mod._read(st)
        if kind in ("mesa", "lact"):
            assert torch.isfinite(got).all(), f"{kind}: non-finite at extended depth"
            continue
        torch.testing.assert_close(got, naive_read(mod, st), rtol=2e-4, atol=6e-2,
                                   msg=f"{kind}: row clamping differs from reference")
        assert torch.isfinite(got).all(), f"{kind}: non-finite at extended depth"
    print("PASS extended depth (more sources than parameter rows) is clamped, not wrapped")


def test_gradients_reach_every_parameter():
    """No dead parameters: every parameter tensor has a finite, nonzero gradient at
    init and after one optimiser step."""
    for kind in LINATTN_KINDS:
        mod = make(kind)
        with torch.no_grad():
            mod.alpha.fill_(0.5)             # alpha=0 legitimately zeroes the read
        # unit-scale states here on purpose: the step below has to leave the
        # parameters in a sane place, and at the real 300-norm scale a squared loss
        # is ~9e4 and one SGD step drives dt into a regime where exp(-dt) underflows
        # -- which would make this test report a dead parameter that is only dead
        # because the test broke the model.
        st = states(LOOPS + 1, scale=1.0)
        opt = torch.optim.SGD(mod.parameters(), lr=1e-3)
        for step in (0, 1):
            opt.zero_grad(set_to_none=True)
            mod.combine(st).square().mean().backward()
            for name, p in mod.named_parameters():
                assert p.grad is not None, f"{kind}: {name} has no grad"
                assert torch.isfinite(p.grad).all(), f"{kind}: {name} grad non-finite"
                assert bool(p.grad.abs().max() > 0), \
                    f"{kind}: {name} has a zero gradient at step {step}"
            opt.step()
    print("PASS every parameter is gradient-reachable")


def test_param_counts():
    rows = LOOPS + 1          # one write per SOURCE
    grows = LOOPS             # one decay per TRANSITION -- there is one fewer
    dh = D // H
    expect = {
        "linattn": 1,
        "retnet": 1,
        "mamba2": 1 + H + rows * H,      # dt also sets the write scale, so it keeps rows
        "gla": 1 + grows * H * dh,
        "hgrn2": 1 + rows * H * dh,      # the gate IS the write, so it keeps rows too
        "deltanet": 1 + rows * H,
        "gdn": 1 + rows * H + grows * H,
        "kda": 1 + rows * H + grows * H * dh,
        "rwkv7": 1 + 2 * grows * H * dh,   # per-channel decay AND per-channel removal
        "lact": 1 + rows * H,              # one learning rate per source, nothing else
        "mesa": 1 + rows * H + H,        # gamma per (source, head) + one ridge per head
    }
    for kind in LINATTN_KINDS:
        got = sum(p.numel() for p in make(kind).parameters())
        assert got == expect[kind], f"{kind}: {got} params, expected {expect[kind]}"
    print("PASS parameter counts: " +
          ", ".join(f"{k}={expect[k]}" for k in LINATTN_KINDS))


def test_compile_agrees():
    """torch.compile agrees with eager for every kind."""
    import torch._dynamo
    st = states(LOOPS + 1, scale=300.0)
    for kind in LINATTN_KINDS:
        # without this the specialisations of one `combine` code object blow through
        # dynamo's recompile limit and the rest run EAGER -- the test would then print
        # "compiled agrees" having compiled nothing.
        torch._dynamo.reset()
        mod = randomise_(make(kind))
        eager = mod.combine(st)
        comp = torch.compile(mod.combine, fullgraph=True)(st)
        torch.testing.assert_close(comp, eager, rtol=1e-5, atol=1e-3,
                                   msg=f"{kind}: compiled != eager")
    print("PASS torch.compile agrees with eager for every kind")


def kda_paper_read(mod, st):
    """KDA in the PAPER's orientation, written without this file's transition helpers.

    KDA, as defined in Kimi Linear (Moonshot AI, 2025), keeps a key x value state and applies
        S_paper <- E D S_paper + beta k v^T,     E = I - beta k k^T,  D = Diag(a)
    and reads o = S_paper^T q. `naive_read` cannot check this: it builds the transition
    from the module's own `_gate`/`_beta` and hard-codes the same left/right convention
    the module uses, so it validates the FACTORISATION and inherits the ORDER. This one
    is written straight from the paper's recurrence and is therefore independent of it.
    """
    assert mod.kind == "kda"
    hfs = [h.float() for h in st]
    ks = [mod._l2(mod._split(h)) for h in hfs]
    vs = [mod._split(h) for h in hfs]
    q = mod._l2(mod._split(hfs[-1]))
    b, t, nh, dh = q.shape
    S = torch.zeros(b, t, nh, dh, dh)                  # [key, value]
    eye = torch.eye(dh).view(1, 1, 1, dh, dh)

    for r in range(len(st)):
        beta = mod._beta(r)
        kr = ks[r]
        if r > 0:
            a = mod._gate(r)
            E = eye - beta.unsqueeze(-1) * (kr.unsqueeze(-1) * kr.unsqueeze(-2))
            S = E @ (a.unsqueeze(-1) * S)              # E D S
        S = S + beta.unsqueeze(-1) * (kr.unsqueeze(-1) * vs[r].unsqueeze(-2))
    return (S * q.unsqueeze(-1)).sum(-2).reshape(st[-1].shape)


def test_kda_matches_the_paper_orientation():
    """`kda` is KDA, and the test can tell the two gate/erase orders apart."""
    mod = randomise_(make("kda"))
    for n in range(1, LOOPS + 2):
        st = states(n, scale=300.0)
        torch.testing.assert_close(
            mod._read(st), kda_paper_read(mod, st), rtol=2e-4, atol=6e-2,
            msg=lambda m, n=n: f"kda n={n}: does not match KDA's own recurrence\n{m}")

    # The order only has content when the gate is non-uniform across channels, so show
    # the alternative order actually DISAGREES here -- otherwise the test above would
    # pass for a gate-order bug and prove nothing.
    st = states(LOOPS + 1, scale=300.0)
    hfs = [h.float() for h in st]
    ks = [mod._l2(mod._split(h)) for h in hfs]
    vs = [mod._split(h) for h in hfs]
    q = mod._l2(mod._split(hfs[-1]))
    W = None
    for r in range(len(st)):
        beta = mod._beta(r)
        if W is not None:                              # ERASE first, then gate (wrong)
            kr = ks[r].unsqueeze(0)
            W = W - beta.unsqueeze(0) * (W * kr).sum(-1, keepdim=True) * kr
            W = mod._gate(r).unsqueeze(0) * W
        new = (beta * ks[r]).unsqueeze(0)
        W = new if W is None else torch.cat([W, new], 0)
    V = torch.stack(vs, 0)
    other = ((W * q.unsqueeze(0)).sum(-1, keepdim=True) * V).sum(0).reshape(st[-1].shape)
    gap = float((other - mod._read(st)).detach().abs().max())
    # The bar has to clear what the correctness assertion above tolerates, or a genuine
    # order bug could sit inside atol and pass BOTH checks. That assertion uses
    # atol=6e-2 with rtol=2e-4, so require an order of magnitude more than atol.
    assert gap > 6e-1, (
        f"the two gate/erase orders differ by only {gap:.3e} on this input, which is "
        f"within an order of magnitude of the correctness check's own atol (6e-2) -- "
        f"this test cannot reliably detect the order; pick a less commuting case")
    print(f"PASS kda == KDA's own recurrence; the wrong order differs by {gap:.3g}, "
          f"so the check has teeth")


def test_no_dead_gate_rows():
    """Every ROW of every gate parameter must be reachable.

    A whole-tensor "some element is nonzero" check passes with an entire row dead --
    which is what happens when a gate is sized by sources (R+1) instead of transitions
    (R): row 0 is never read, has zero gradient forever, and still counts as capacity
    in the parameter table.
    """
    for kind in LINATTN_KINDS:
        mod = randomise_(make(kind))
        with torch.no_grad():
            mod.alpha.fill_(0.5)
        mod.combine(states(LOOPS + 1, scale=300.0)).square().sum().backward()
        for name, p in mod.named_parameters():
            if p.dim() == 0 or name == "alpha":
                continue        # alpha is the mechanism's own gate, checked separately
            per_row = p.grad.reshape(p.shape[0], -1).abs().amax(dim=1)
            dead = (per_row == 0).nonzero().flatten().tolist()
            assert not dead, f"{kind}: {name} rows {dead} have zero gradient -- dead"
    print("PASS no dead rows: every gate/beta row reaches the loss")


def test_extreme_gates_stay_finite():
    """Large dt / A_log must not underflow the decay to 0 or NaN the backward."""
    st = states(LOOPS + 1, scale=300.0)
    for A_log, dt_bias in ((0.0, 104.0), (90.0, 0.0), (90.0, 104.0), (-90.0, -90.0)):
        mod = make("mamba2")
        with torch.no_grad():
            mod.alpha.fill_(0.5)
            mod.A_log.fill_(A_log)
            mod.dt_bias.fill_(dt_bias)
        out = mod.combine(st)
        assert torch.isfinite(out).all(), f"A_log={A_log} dt={dt_bias}: output not finite"
        gate = mod._gate(1)
        assert float(gate.min()) > 0.0, \
            f"A_log={A_log} dt={dt_bias}: decay underflowed to exactly 0"
        out.square().sum().backward()
        for name, p in mod.named_parameters():
            assert torch.isfinite(p.grad).all(), \
                f"A_log={A_log} dt={dt_bias}: {name} grad is not finite"
    # the write scale must reuse the SAME bounded dt as the decay, and lact's step
    # multiplies un-normalised values so it needs a bound too
    for kind, name, val in (("mamba2", "dt_bias", 1e38), ("lact", "b_eta", 1e38)):
        mod = make(kind)
        with torch.no_grad():
            mod.alpha.fill_(0.5)
            getattr(mod, name).fill_(val)
        out = mod.combine(st)
        assert torch.isfinite(out).all(), f"{kind}: {name}={val} gives a non-finite read"
        out.square().sum().backward()
        for pn, p in mod.named_parameters():
            assert torch.isfinite(p.grad).all(), f"{kind}: {name}={val} -> {pn} grad"
    print("PASS extreme dt / A_log / eta stay finite in forward AND backward, and the "
          "decay never reaches exactly 0")


def _saved_bytes(kind, L):
    mod = LoopLinAttn(8, L, kind=kind, num_heads=1)
    with torch.no_grad():
        mod.alpha.fill_(0.5)
    deltas = [torch.randn(1, 1, 8) for _ in range(L + 1)]
    total = []
    with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (total.append(t.numel() * t.element_size()), t)[1], lambda t: t):
        x = deltas[0]
        hist = [x]
        for l in range(L):
            x = x + deltas[l + 1] * 0.01
            hist.append(x)
            x = mod.combine(hist)
        x.square().sum()
    return sum(total)


def test_scaling_differs_by_path():
    """The three code paths have DIFFERENT complexity in the iteration count.

    The DPLR kernel is O(R^3) saved activations over a whole forward -- O(R^2) inside
    each junction, recomputed at each of R junctions. `lact` is worse: one read is
    already O(n^3) (an n x n projection against n keys at every one of n steps), so
    recomputing it per junction is O(R^4). `mesa` carries an n x n solve per junction.
    """
    ratios = {}
    for kind in ("kda", "lact", "mesa"):
        s = {L: _saved_bytes(kind, L) for L in (8, 16, 32)}
        ratios[kind] = (s[16] / s[8], s[32] / s[16])
    (k1, k2) = ratios["kda"]
    (l1, l2) = ratios["lact"]
    assert k2 > k1 > 4.0, f"kda growth {ratios['kda']} is not super-quadratic"
    assert k2 < 8.0, f"kda growth {ratios['kda']} exceeds cubic; recheck the docstring"
    (m1, m2) = ratios["mesa"]
    assert l2 > k2, (
        f"lact ({l2:.1f}x) should grow FASTER than kda ({k2:.1f}x) per doubling -- it "
        f"re-solves an O(n^3) problem at every junction")
    assert m2 > 4.0, f"mesa growth {ratios['mesa']} is not super-quadratic"
    assert m2 < k2, (
        f"mesa ({m2:.1f}x) should grow more slowly than kda ({k2:.1f}x): it carries one "
        f"n x n solve per junction, not an O(R^2) recurrence")
    body = ", ".join(f"{k}: x{a:.1f} then x{b:.1f}" for k, (a, b) in ratios.items())
    print(f"PASS saved activations scale differently per path ({body}) -- DPLR cubic, "
          f"lact steeper")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true",
                    help="also run the compiled-vs-eager check (slow)")
    a = ap.parse_args()

    torch.manual_seed(0)
    test_parameterisation_contract()
    test_parameter_effectiveness()
    test_matches_materialised_state()
    test_lact_zero_state_stays_finite_in_backward()
    test_lact_matches_the_materialised_fast_weight()
    test_lact_chunk_size_changes_the_answer()
    test_lact_global_eta_scale_is_weak_only_at_the_stream_scale()
    test_mesa_matches_the_dense_solution()
    test_kda_matches_the_paper_orientation()
    test_hgrn2_gate_telescopes()
    test_rwkv7_transition_is_additive_dplr()
    test_init_equivalence()
    test_grid_collapses()
    test_batch_invariance()
    test_depth_beyond_construction()
    test_gradients_reach_every_parameter()
    test_no_dead_gate_rows()
    test_extreme_gates_stay_finite()
    test_scaling_differs_by_path()
    test_param_counts()
    if a.compile:
        test_compile_agrees()
    print("\nALL LINATTN TESTS PASSED")


if __name__ == "__main__":
    main()
