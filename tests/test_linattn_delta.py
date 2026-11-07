"""Tests for loop_linattn_delta.py: the readers over whole-iteration updates (the paper's update interface).

The load-bearing test is `test_matches_reference_oracle`. It checks combine() against
`tests/reference_linattn.py` -- an INDEPENDENT materialised implementation written from the
specification, importing nothing from loop_linattn_delta.py and sharing no helper with it. That
independence is the point: a reference that reuses the module's private helpers (`_split`, `_beta`,
`_gate`, ...) moves both sides of every assertion together when one of them is wrong; the oracle
cannot do that.

`test_matches_materialised_state` remains, but as a FACTORISATION check only: it builds S explicitly
from the module's own helpers, so it proves the factored kernel equals a materialised one under the
SAME parameterisation. It is NOT an independent numerical oracle and must not be mistaken for one.

combine(states, inputs=inputs): states[r] is the output of iteration r before its boundary blend and
inputs[r] the state that entered it (inputs[0] = h_0), one input per state. The memory holds
Delta_r = states[r] - inputs[r] for EVERY r including r = 0. Keys and values come from the updates,
the QUERY is the eps-clamped L2 normalisation of the current state states[-1], never an update, and
h_0 leaves the memory and is added back AFTER the read:

    y = (1 - alpha) states[-1] + alpha (inputs[0] + u_r) .

Most tests build the pairs from a chain of states [h_0 .. h_r] with `chain_combine` (inputs =
chain[:-1], states = chain[1:], so Delta_j = h_{j+1} - h_j); the oracle differential uses independent
inputs and states, so that no difference of adjacent states could pass for the update.

    python tests/test_linattn_delta.py
    python tests/test_linattn_delta.py --compile  # also check torch.compile agrees with eager
"""

import argparse
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))      # loop_linattn_delta.py at the repository root
sys.path.insert(0, _HERE)                         # reference_linattn.py next to this file

from loop_linattn_delta import (  # noqa: E402
    LINATTN_KINDS, LoopLinAttn,
    _LACT_NORM_EPS as LACT_EPS,
    _LACT_ETA_MAX as LACT_ETA_MAX,
    _RIDGE_MIN as RIDGE_MIN,
)
import reference_linattn as ref  # noqa: E402

# A single head at the full width is part of this interface's definition, asserted in the
# constructor; H is 1 everywhere and dh == D. MULTIHEAD is a value the constructor must REFUSE.
B, T, D, H, LOOPS = 2, 3, 32, 1, 3
MULTIHEAD = 4


def make(kind, seed=0):
    torch.manual_seed(seed)
    return LoopLinAttn(D, LOOPS, kind=kind, num_heads=H)


def states(n, scale=1.0, seed=1, dtype=torch.float32):
    """n states (B, T, D), each an independent draw scaled by `scale`."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, T, D, generator=g, dtype=torch.float32).to(dtype) * scale
            for _ in range(n)]


def chain_combine(mod, st):
    """combine() on the pairs a chain of states [h_0 .. h_r] defines: inputs = st[:-1],
    states = st[1:], so the updates are the consecutive differences."""
    return mod.combine(st[1:], inputs=st[:-1])


def deltas_and_query(st):
    """The read's inputs as a chain defines them: consecutive differences as the sources, the
    current state as the query. Used to drive `_read` directly; what combine() actually
    writes is pinned by `test_combine_sources_are_block_deltas`."""
    return [b - a for a, b in zip(st[:-1], st[1:])], st[-1]


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
# Materialised references, built from the module's OWN helpers. These check the
# FACTORISATION (factored kernel == explicit dh x dh state under the same parameters).
# They are NOT independent numerical oracles -- that is reference_linattn.py's job.
# --------------------------------------------------------------------------------

def _materialised_dplr_read(mod, ins, outs):
    """u = S q with S built explicitly over the update sources, one DPLR transition per
    iteration. The erase direction is the current write key everywhere (rwkv7 included:
    khat_r = L2norm(delta_r) is exactly ks[r])."""
    dfs = [(o - i).float() for i, o in zip(ins, outs)]
    n = len(dfs)
    ks = [mod._l2(mod._split(d)) for d in dfs]
    vs = [mod._split(d) for d in dfs]
    q = mod._l2(mod._split(outs[-1].float()))
    b, t, nh, dh = q.shape
    S = torch.zeros(b, t, nh, dh, dh)
    eye = torch.eye(dh).view(1, 1, 1, dh, dh)

    for r in range(n):
        beta = mod._beta(r) if mod.uses_delta else None
        dt = mod._dt(r) if mod.kind == "mamba2" else None
        gate = mod._gate(r, dt=dt) if mod.decay is not None else None
        er = ks[r]

        if r > 0:
            if mod.additive_dplr:
                # A_r = Diag(w) - khat (rem * khat)^T : diagonal PLUS rank one.
                rem = mod._removal(r)
                A = (torch.diag_embed(gate.expand(*gate.shape[:-1], dh))
                     - er.unsqueeze(-1) * (rem * er).unsqueeze(-2))
            else:
                # A_r = Diag(a) (I - beta khat khat^T)
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

    return (S * q.unsqueeze(-2)).sum(-1).reshape(outs[-1].shape)


def _materialised_mesa_read(mod, ins, outs):
    """`mesa` by building the dh x dh solution, instead of the n x n Woodbury form."""
    dfs = [(o - i).float() for i, o in zip(ins, outs)]
    n = len(dfs)
    K = torch.stack([mod._l2(mod._split(d)) for d in dfs], dim=-2)      # (b,t,H,n,dh)
    V = torch.stack([mod._split(d) for d in dfs], dim=-2)
    q = mod._l2(mod._split(outs[-1].float()))
    dh = q.shape[-1]
    gam = torch.sigmoid(torch.stack(
        [mod._head_scalar(mod.b_gamma, j).squeeze(-1)
         for j in range(n)], dim=-1)).expand(*V.shape[:-1])
    lam = (RIDGE_MIN + torch.nn.functional.softplus(
        mod.ridge.float())).view(1, 1, -1, 1, 1)
    G = K.transpose(-1, -2) @ (gam.unsqueeze(-1) * K)                   # (b,t,H,dh,dh)
    G = G + lam * torch.eye(dh).view(1, 1, 1, dh, dh)
    S = V.transpose(-1, -2) @ (gam.unsqueeze(-1) * K) @ torch.linalg.inv(G)
    return (S @ q.unsqueeze(-1)).squeeze(-1).reshape(outs[-1].shape)


def _materialised_lact_read(mod, ins, outs):
    """`lact` by materialising the dh x dh fast weight and stepping it explicitly."""
    dfs = [(o - i).float() for i, o in zip(ins, outs)]
    K = [mod._l2(mod._split(d)) for d in dfs]
    V = [mod._split(d) for d in dfs]
    q = mod._l2(mod._split(outs[-1].float()))
    b, t, nh, dh = q.shape
    W = torch.zeros(b, t, nh, dh, dh)                 # [value, key]
    for r in range(len(dfs)):
        g = torch.zeros_like(W)
        for j in range(r + 1):
            eta = torch.nn.functional.softplus(
                mod._head_scalar(mod.b_eta, j)).clamp(max=LACT_ETA_MAX)   # (..,H,1)
            resid = (W @ K[j].unsqueeze(-1)).squeeze(-1) - V[j]        # (b,t,H,dh)
            g = g + eta.unsqueeze(-1) * (resid.unsqueeze(-1) * K[j].unsqueeze(-2))
        W = W - g
        nrm2 = W.flatten(-2).pow(2).sum(-1) + LACT_EPS ** 2
        W = W / nrm2.sqrt()[..., None, None]
    return (W @ q.unsqueeze(-1)).squeeze(-1).reshape(outs[-1].shape)


def materialised_read(mod, ins, outs):
    if mod.kind == "mesa":
        return _materialised_mesa_read(mod, ins, outs)
    if mod.kind == "lact":
        return _materialised_lact_read(mod, ins, outs)
    return _materialised_dplr_read(mod, ins, outs)


def materialised_combine(mod, ins, outs):
    """The junction output from the materialised state: inputs[0] added back AFTER the
    read, then the alpha blend."""
    u = materialised_read(mod, ins, outs)
    alpha = mod.alpha.detach().float()
    y = (1.0 - alpha) * outs[-1].float() + alpha * (ins[0].float() + u)
    return y.to(outs[-1].dtype)


# Chain-form convenience wrappers: a chain [h_0 .. h_r] gives inputs st[:-1], states st[1:].
def naive_read(mod, st):
    return materialised_read(mod, st[:-1], st[1:])


def naive_combine(mod, st):
    return materialised_combine(mod, st[:-1], st[1:])


def mesa_matrix_read(mod, st):
    return _materialised_mesa_read(mod, st[:-1], st[1:])


def lact_matrix_read(mod, st):
    return _materialised_lact_read(mod, st[:-1], st[1:])


# --------------------------------------------------------------------------------
# The INDEPENDENT oracle differential.
# --------------------------------------------------------------------------------

def test_matches_reference_oracle():
    """combine() == tests/reference_linattn.py, for every mode, depth and scale.

    The oracle imports nothing from loop_linattn_delta.py and shares no helper with it,
    so a bug in the implementation cannot hide by moving both sides of the assertion:
    a read scaled by a constant, key/value channels reversed, a write coefficient
    halved, or a combine that ignored `inputs` and sourced the memory from the states
    would all be caught here. The inputs and states are drawn independently, so
    delta_r = out_r - in_r is genuinely not any difference of adjacent states.

    `lact` is excluded from the numeric comparison: the oracle's `lact` is a single
    unnormalised gradient step over the prefix from an empty memory, while the
    implementation takes one Frobenius-normalised step per junction from the memory
    of the previous junction (the paper's eq. lact-prefix-map). The two differ by
    design, so `lact` is pinned by `test_lact_matches_the_materialised_fast_weight`
    and the chunk-size tests instead.
    """
    for kind in LINATTN_KINDS:
        if kind == "lact":
            continue
        mod = randomise_(make(kind))
        with torch.no_grad():
            mod.alpha.fill_(0.5)              # nonzero, so the read actually shows up
        params = mod.state_dict()
        for n in range(1, LOOPS + 2):        # n = number of update sources
            for scale in (1.0, 300.0):
                ins = states(n, scale=scale, seed=100 + n)
                outs = states(n, scale=scale, seed=200 + n)
                got = mod.combine(list(outs), inputs=list(ins))
                want = ref.reference_junction(params, ins, outs, kind)
                torch.testing.assert_close(
                    got, want.to(got.dtype), rtol=2e-4, atol=2e-4 * max(scale, 1.0),
                    msg=lambda m, k=kind, n=n, s=scale:
                        f"{k} n={n} scale={s}: combine() != oracle\n{m}")
    print(f"PASS combine() matches the independent oracle for every mode, across depths "
          f"and scales ({len(LINATTN_KINDS)} kinds)")


def test_matches_materialised_state():
    """The factored kernel == the explicit dh x dh state, for every mode and depth.

    A FACTORISATION check (module helpers on both sides), not an independent oracle.
    Both levels of the contract are checked: the READ (`_read` on the updates and the
    query) and the JUNCTION OUTPUT (combine() adds inputs[0] back after the read).
    """
    for kind in LINATTN_KINDS:
        if kind in ("mesa", "lact"):
            continue          # not DPLR recurrences; they have their own references
        mod = randomise_(make(kind))
        with torch.no_grad():
            mod.alpha.fill_(0.5)          # a nonzero alpha, so the read shows up
        for n in range(1, LOOPS + 2):     # n = number of update sources
            for scale in (1.0, 300.0):    # 300 ~ the real residual-stream norm
                st = states(n + 1, scale=scale)
                ds, qs = deltas_and_query(st)
                got = mod._read(ds, qs)
                ref_read = naive_read(mod, st)
                torch.testing.assert_close(
                    got, ref_read, rtol=2e-4, atol=2e-4 * max(scale, 1.0),
                    msg=lambda m, k=kind, n=n, s=scale:
                        f"{k} n={n} scale={s}: factored != materialised\n{m}")
                torch.testing.assert_close(
                    chain_combine(mod, st), naive_combine(mod, st),
                    rtol=2e-4, atol=2e-4 * max(scale, 1.0),
                    msg=lambda m, k=kind, n=n, s=scale:
                        f"{k} n={n} scale={s}: combine() != "
                        f"(1-a) h_r + a (h_0 + u_ref)\n{m}")
    print(f"PASS factored == materialised, and combine adds inputs[0] back after the "
          f"read ({len(LINATTN_KINDS)} kinds x depths x scales)")


def test_combine_sources_are_block_deltas():
    """What combine() writes into the memory is states[r] - inputs[r], for EVERY r.

    The sources are CAPTURED at the implementation's own read boundary -- a spy wraps
    the instance's `_read` -- rather than recomputed, so a combine that differenced
    wrongly, sourced the memory from the states, or slipped inputs[0] into the memory
    cannot pass by matching a reference that made the same mistake. Pins the r = 0
    rule and the query.
    """
    for kind in LINATTN_KINDS:
        mod = randomise_(make(kind))

        captured = {}
        orig_read = mod._read

        def spy(sources, query, _orig=orig_read, _cap=captured):
            _cap["sources"] = list(sources)
            _cap["query"] = query
            return _orig(sources, query)

        # independent ins/outs: delta_r = out_r - in_r is not any adjacent difference,
        # so sourcing the memory from the states is exposed
        ins = states(LOOPS, scale=300.0, seed=41)
        outs = states(LOOPS, scale=300.0, seed=42)
        mod._read = spy
        mod.combine(list(outs), inputs=list(ins))
        mod._read = orig_read

        src = captured["sources"]
        assert len(src) == len(outs), \
            f"{kind}: {len(src)} sources for {len(outs)} states"
        for r in range(len(src)):
            assert torch.equal(src[r], outs[r] - ins[r]), \
                f"{kind}: source {r} is not states[{r}] - inputs[{r}]"
        assert torch.equal(captured["query"], outs[-1]), \
            f"{kind}: query is not the current state states[-1]"

    # Malformed calls are hard errors, not silent empty reads.
    mod = make("gdn")
    for bad, kw in ((states(2), {"inputs": states(1)}),        # length mismatch
                    ([], {"inputs": []})):                     # empty pair list
        raised = False
        try:
            mod.combine(bad, **kw)
        except ValueError:
            raised = True
        assert raised, f"combine() accepted a malformed call: {bad!r} {kw!r}"
    print("PASS combine() writes exactly the updates states[r] - inputs[r] (captured from "
          "the implementation); the query is states[-1], and malformed calls are refused")


def test_lact_zero_state_stays_finite_in_backward():
    """An all-zero history must not NaN the backward through LaCT's normalisation.

    d/dx sqrt(x) is infinite at 0, so putting eps after the sqrt guards the division and
    not the gradient. With the update memory the all-zero case is training step 0: every
    residual writer is zero, so EVERY update is exactly zero. scale=0 makes the updates
    exactly zero; scale=1e-20 makes them tiny but nonzero (below the eps clamp).
    """
    for scale in (0.0, 1e-20):
        mod = make("lact")
        st = [(torch.zeros(B, T, D) + (i + 1) * scale).requires_grad_(True)
              for i in range(LOOPS + 1)]
        out = chain_combine(mod, st)
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
    for n in range(1, LOOPS + 2):          # n = number of update sources
        for scale in (1.0, 300.0):
            st = states(n + 1, scale=scale)
            ds, qs = deltas_and_query(st)
            torch.testing.assert_close(
                mod._read(ds, qs), lact_matrix_read(mod, st),
                rtol=2e-4, atol=2e-4,
                msg=lambda m, n=n, s=scale:
                    f"lact n={n} scale={s}: factored != materialised\n{m}")
    print("PASS lact: the factored chunk step == the explicit fast-weight step")


def _lact_chunk_of_one(mod, st):
    """The same update restricted to the newest update source -- LaCT's chunk size 1."""
    dfs = [(b - a).float() for a, b in zip(st[:-1], st[1:])]
    K = [mod._l2(mod._split(d)) for d in dfs]
    V = [mod._split(d) for d in dfs]
    q = mod._l2(mod._split(st[-1].float()))
    b, t, nh, dh = q.shape
    W = torch.zeros(b, t, nh, dh, dh)
    for r in range(len(dfs)):
        eta = torch.nn.functional.softplus(
            mod._head_scalar(mod.b_eta, r)).clamp(max=LACT_ETA_MAX)
        resid = (W @ K[r].unsqueeze(-1)).squeeze(-1) - V[r]
        W = W - eta.unsqueeze(-1) * (resid.unsqueeze(-1) * K[r].unsqueeze(-2))
        nrm2 = W.flatten(-2).pow(2).sum(-1) + LACT_EPS ** 2
        W = W / nrm2.sqrt()[..., None, None]
    return (W @ q.unsqueeze(-1)).squeeze(-1).reshape(st[-1].shape)


def test_lact_chunk_size_changes_the_answer():
    """chunk = whole history is genuinely different from chunk = newest source.

    The LaCT-inspired reader's distinguishing feature is the step over the whole stored
    prefix, so the arm is worthless if it silently behaves like a chunk of 1. Measured
    across collinearity levels OF THE SOURCES -- the sources are the updates, so the
    collinearity is built at the update level and the states are integrated back up
    from it.
    """
    mod = randomise_(make("lact"))
    seen = {}
    for pert, label in ((0.3, "0.95"), (0.7, "0.78"), (2.0, "0.35")):
        g = torch.Generator().manual_seed(11)
        base = torch.randn(B, T, D, generator=g) * 300.0
        h0 = torch.randn(B, T, D, generator=g) * 300.0
        ds = [base + pert * 300.0 * torch.randn(B, T, D, generator=g)
              for _ in range(LOOPS + 1)]
        st = [h0]
        for d in ds:
            st.append(st[-1] + d)
        f = torch.stack([d.reshape(-1) for d in ds])
        f = f / f.norm(dim=-1, keepdim=True)
        cos = float((f[0] * f[1]).sum())
        allc = mod._read(*deltas_and_query(st)).detach()
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

    Varying one common scale cannot distinguish "this parameter does nothing" from "this
    parameter's overall magnitude does nothing"; the insensitivity is an artefact of the
    stream's norm making the write term dominate the carried one. Tilting eta ACROSS
    sources moves the read a great deal, so the parameters are live.
    """
    seen = {}
    for scale in (1.0, 300.0):
        stx = states(LOOPS + 1, scale=scale)
        dsx, qsx = deltas_and_query(stx)
        outs = []
        for raw in (-3.0, 0.0, 1.0, 3.0):      # same value everywhere: pure scale
            mod = make("lact")
            with torch.no_grad():
                mod.b_eta.fill_(raw)
            outs.append(mod._read(dsx, qsx).detach())
        r0 = outs[0]
        seen[scale] = max(float((o - r0).abs().max())
                          for o in outs[1:]) / float(r0.abs().max())
    assert seen[300.0] < 0.05, (
        f"at the stream scale a common eta no longer washes out ({seen[300.0]:.3f}); "
        f"the docstring is stale")
    assert seen[1.0] > 0.2, (
        f"at unit scale a common eta only moves the read {seen[1.0]:.3f}. If it is now "
        f"small everywhere, the insensitivity may be structural after all")

    st = states(LOOPS + 1, scale=300.0)
    ds, qs = deltas_and_query(st)
    mod = make("lact")
    with torch.no_grad():
        mod.b_eta.zero_()
    flat = mod._read(ds, qs).detach()
    with torch.no_grad():
        mod.b_eta.copy_(torch.linspace(-2.0, 2.0, mod.b_eta.numel()).view_as(mod.b_eta))
    tilted = mod._read(ds, qs).detach()
    rel_spread = float((tilted - flat).abs().max()) / float(flat.abs().max())
    assert rel_spread > 0.05, (
        f"tilting eta across sources only moves the read by {rel_spread:.3e}; then "
        f"`lact` really would be a zero-parameter reader")
    body = ", ".join(f"input scale {s:g}: {v:.1%}" for s, v in seen.items())
    print(f"PASS lact: a common eta scale is weak ONLY at the stream scale ({body}); "
          f"tilting eta across sources moves the read {rel_spread:.0%}")


def test_mesa_matches_the_dense_solution():
    """`mesa`'s n x n Woodbury form == the dh x dh ridge-regression solution."""
    mod = randomise_(make("mesa"))
    for n in range(1, LOOPS + 2):              # n = number of update sources
        for scale in (1.0, 300.0):
            st = states(n + 1, scale=scale)
            ds, qs = deltas_and_query(st)
            torch.testing.assert_close(
                mod._read(ds, qs), mesa_matrix_read(mod, st),
                rtol=2e-4, atol=2e-4 * max(scale, 1.0),
                msg=lambda m, n=n, s=scale: f"mesa n={n} scale={s}: "
                                            f"n x n form != dense solution\n{m}")
    # the corner the randomisation never reaches: saturated gamma, floored ridge,
    # identical (perfectly collinear) sources. Identical STATES would make every update
    # exactly zero, so the states are a ramp: every update is the same nonzero vector.
    deg = make("mesa")
    with torch.no_grad():
        deg.ridge.fill_(-20.0)
        deg.b_gamma.fill_(-20.0)
    ramp = [torch.ones(B, T, D) * 300.0 * (j + 1.0) for j in range(LOOPS + 2)]
    dsd, qsd = deltas_and_query(ramp)
    got, ref_dense = deg._read(dsd, qsd), mesa_matrix_read(deg, ramp)
    assert torch.isfinite(got).all(), "mesa: degenerate corner is not finite"
    torch.testing.assert_close(got, ref_dense, rtol=1e-3, atol=1e-3,
                               msg="mesa: reference and implementation disagree at "
                                   "saturated gamma / floored ridge / collinear keys")
    # TRUE saturation, and the BACKWARD. sigmoid(-100) underflows to exactly 0 in fp32,
    # and building sqrt(gamma) as sqrt(sigmoid(x)) then has an infinite derivative there.
    for raw in (-100.0, -1e38):
        sat = make("mesa")
        with torch.no_grad():
            sat.alpha.fill_(0.5)
            sat.b_gamma.fill_(raw)
        out = chain_combine(sat, states(LOOPS + 1, scale=300.0))
        assert torch.isfinite(out).all(), f"mesa b_gamma={raw}: forward"
        out.square().sum().backward()
        for name, p in sat.named_parameters():
            assert torch.isfinite(p.grad).all(), \
                f"mesa b_gamma={raw}: {name} grad is not finite"
    print("PASS mesa: the n x n Woodbury read == the dh x dh ridge solution (floored "
          "ridge, identical sources), and a fully saturated gamma keeps the BACKWARD "
          "finite")


def test_mesa_ridge_floor_is_applied():
    """The RIDGE_MIN floor is OBSERVED in the implementation, not just in the test.

    The floor is made to BITE: identical (collinear) update sources make the Gram matrix
    rank-deficient, so a solve with the ridge driven far below the floor is singular. The
    module can only return the FLOORED oracle's finite answer if it actually applied the
    floor; drop it and the collinear-key solve blows up or disagrees.
    """
    m = make("mesa")
    with torch.no_grad():
        m.ridge.fill_(-30.0)                 # softplus(-30) ~ 9e-14, far below RIDGE_MIN
    ramp = [torch.ones(B, T, D) * 300.0 * (j + 1.0) for j in range(LOOPS + 2)]
    got = chain_combine(m, ramp)              # every update identical
    want = ref.reference_junction(m.state_dict(), ramp[:-1], ramp[1:], "mesa")
    assert torch.isfinite(got).all(), \
        "mesa: floored ridge is not finite on collinear keys"
    torch.testing.assert_close(
        got, want.to(got.dtype), rtol=1e-4, atol=1e-3,
        msg="mesa ridge floor not applied: the collinear-key solve disagrees with the "
            "floored oracle")
    print("PASS mesa applies the RIDGE_MIN floor (observed against the oracle on a "
          "singular-without-it collinear system)")


def test_hgrn2_gate_telescopes():
    """hgrn2's KEY-CHANNEL weights are non-negative and telescope to 1 - prod(a).

    The tied input/forget gate buys per-CHANNEL leaky integration: every w_j is
    elementwise non-negative and, along each key channel, the weights over the history
    telescope to 1 - prod_r a_r. It is NOT a convex combination of the sources -- the
    source coefficient c_j = w_j . q is signed, because q is a signed vector.
    """
    mod = randomise_(make("hgrn2"))
    st = states(LOOPS + 1, scale=300.0)
    dfs = [(b - a).float() for a, b in zip(st[:-1], st[1:])]
    n = len(dfs)
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
    V = torch.stack([mod._split(d) for d in dfs], 0)
    q = mod._l2(mod._split(st[-1].float()))
    u = ((stacked * q.unsqueeze(0)).sum(-1, keepdim=True) * V).sum(0)
    torch.testing.assert_close(u.reshape(st[-1].shape),
                               mod._read(dfs, st[-1]),
                               rtol=2e-4, atol=6e-2,
                               msg="hgrn2's read does not match its closed-form weights")

    c = (stacked * q.unsqueeze(0)).sum(-1).detach()
    assert float(c.min()) < 0.0, \
        "hgrn2 source coefficients came out non-negative on signed states"
    print(f"PASS hgrn2: key weights non-negative and telescope to 1-prod(a); source "
          f"coefficients are signed (min {float(c.min()):.3f})")


def test_rwkv7_transition_is_additive_dplr():
    """rwkv7's transition is Diag(w) - khat (a*khat)^T, not Diag(w)(I - beta khat khat^T).

    The 2x2 hand anchor pins the additive form and its gap from the multiplicative one.
    Then the MODULE is observed: rwkv7's read matches the independent oracle, whose
    `_recurrent_state` builds diag(g) - outer(kappa, rho*kappa) from scratch. If the
    implementation had shipped the multiplicative map instead, the module would disagree
    with the oracle.
    """
    g = torch.tensor([0.8, 0.6])
    e = torch.tensor([1.0, 1.0]) / math.sqrt(2.0)
    rem = torch.tensor([0.2, 0.9])
    A = torch.diag(g) - torch.outer(e, rem * e)
    expect = torch.tensor([[0.70, -0.45], [-0.10, 0.15]])
    torch.testing.assert_close(A, expect, rtol=1e-4, atol=1e-4,
                               msg="the hand-computed reference itself is wrong")
    u = torch.tensor([1.0, 0.0])
    got = g * u - (rem * e) * (e * u).sum()
    torch.testing.assert_close(got, A[0], rtol=1e-4, atol=1e-4,
                               msg="the additive-DPLR key-side update is not A^T")
    beta = 0.5
    mult = (torch.eye(2) - beta * torch.outer(e, e)) @ torch.diag(g)
    gap = float((mult - A).abs().max())
    assert gap > 0.1, f"additive and multiplicative DPLR differ by only {gap:.3e}"

    # Observe the implementation: the module's rwkv7 read is the ADDITIVE-DPLR oracle's.
    mod = randomise_(make("rwkv7"))
    with torch.no_grad():
        mod.alpha.fill_(0.5)
    params = mod.state_dict()
    for n in (2, 3):                            # >= 2 sources, so a transition happens
        st = states(n + 1, scale=300.0)
        want = ref.reference_junction(params, st[:-1], st[1:], "rwkv7")
        torch.testing.assert_close(
            chain_combine(mod, st), want.to(torch.float32), rtol=2e-4, atol=6e-2,
            msg=f"rwkv7 n={n}: module read != the additive-DPLR oracle")
    print(f"PASS rwkv7 is additive DPLR (Diag(w) - khat (a*khat)^T, gap {gap:.2f} from "
          f"the multiplicative form), and the module matches the additive-DPLR oracle")


def test_parameterisation_contract():
    """Hand-written expectations for every mode's gates: which ROW, which activation.

    Everything below is computed by hand from the parameter values, with no module
    helper on the right-hand side. The row conventions:
      * beta, eta, gamma      -> row = SOURCE j, num_loops rows
      * decay gates           -> row = TRANSITION j-1, max(num_loops - 1, 1) rows
      * mamba2 dt, hgrn2 gate -> row = SOURCE j (they also appear in the write)
    """
    sig = torch.sigmoid
    sp = torch.nn.functional.softplus
    h = states(1, scale=300.0)[0].float()
    k = make("gdn")._l2(make("gdn")._split(h))        # a unit key, (B, T, H, dh)
    rows, grows = LOOPS, max(LOOPS - 1, 1)

    def rowvals(shape, lo=-1.5, hi=1.5):
        return torch.linspace(lo, hi, int(torch.tensor(shape).prod())).view(shape)

    # ---- beta: source row, sigmoid, and it is also the write coefficient ----
    for kind in ("deltanet", "gdn", "kda"):
        m = make(kind)
        with torch.no_grad():
            m.b_beta.copy_(rowvals(m.b_beta.shape))
        assert m.b_beta.shape[0] == rows, \
            f"{kind}: beta has {m.b_beta.shape[0]} rows, want {rows}"
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
            f"{kind}: gate has {m.b_gate.shape[0]} rows, want {grows}"
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
        gg = sig(m.b_gate[j]).view(1, 1, *m.b_gate.shape[1:])
        torch.testing.assert_close(m._gate(j), gg, rtol=0, atol=0)
        torch.testing.assert_close(m._write_vec(j, torch.zeros_like(gg), gate=gg),
                                   1.0 - gg, rtol=0, atol=0,
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
    small = torch.tensor([[[1e-3] + [0.0] * (D - 1)]])
    n_small = float(m._l2(m._split(small)).norm(dim=-1).flatten()[0])
    assert abs(n_small - 1.0) < 1e-6, (
        f"||khat|| = {n_small:.6f} for a 1e-3 vector, not 1.0 -- the additive-eps form "
        f"is back and I - k k^T is no longer a projection")
    n_big = float(m._l2(m._split(states(1, scale=300.0)[0].float()))
                  .norm(dim=-1).flatten()[0])
    assert abs(n_big - 1.0) < 1e-6, f"||khat|| = {n_big:.6f} at production scale"
    tiny = torch.tensor([[[5e-7] + [0.0] * (D - 1)]])
    n_tiny = float(m._l2(m._split(tiny)).norm(dim=-1).flatten()[0])
    assert n_tiny < 1.0, f"the eps clamp is not engaging below eps (got {n_tiny})"

    # ---- mesa gamma: SOURCE row, and sqrt(gamma) built stably from the logit ----
    m = make("mesa")
    with torch.no_grad():
        m.b_gamma.copy_(rowvals(m.b_gamma.shape, -3.0, 3.0))
    for j in (0, rows - 1):
        want = sig(m.b_gamma[j]).reshape(-1)                       # (H,)
        got = sig(m._head_scalar(m.b_gamma, j)).reshape(-1)
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

    print("PASS parameterisation contract: row indices, activations and key "
          "normalisation match hand-written values")


def test_parameter_effectiveness():
    """How many of each mode's NOMINAL parameters actually move the output?

    A tensor counts as EFFECTIVE if perturbing it by +-1.0 in raw (pre-activation) space
    moves the read by more than 1% of the read's own scale, in the best of four
    directions. The expected-inert set is pinned: anything joining it is a finding.
    """
    INERT_EXPECTED = set()
    st = states(LOOPS + 1, scale=300.0)
    ds, qs = deltas_and_query(st)
    rows = []
    inert = set()
    for kind in LINATTN_KINDS:
        mod = randomise_(make(kind))
        base = mod._read(ds, qs).detach()
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
                    moved = mod._read(ds, qs).detach()
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
          f"inert = {sorted(inert)}")


def test_all_zero_history_stays_finite():
    """An all-zero states list -- EVERY update exactly zero, which is training step 0 --
    gives a finite output and finite gradients, at alpha = 0 and at a nonzero alpha."""
    for kind in LINATTN_KINDS:
        for alpha in (0.0, 0.5):
            mod = make(kind)
            with torch.no_grad():
                mod.alpha.fill_(alpha)
            st = [torch.zeros(B, T, D).requires_grad_(True)
                  for _ in range(LOOPS + 1)]
            out = chain_combine(mod, st)
            assert torch.isfinite(out).all(), \
                f"{kind} alpha={alpha}: forward is not finite on zero states"
            out.sum().backward()
            for name, p in mod.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), \
                    f"{kind} alpha={alpha}: {name} grad is not finite"
            for i, x in enumerate(st):
                assert x.grad is not None and torch.isfinite(x.grad).all(), \
                    f"{kind} alpha={alpha}: input {i} grad is not finite"
    print("PASS an all-zero history (every update exactly zero) stays finite, forward "
          "and backward, at alpha = 0 and alpha = 0.5, for every kind")


def test_num_heads_forced_to_one():
    """num_heads is part of the interface's definition, not a tunable: the default is 1
    at the FULL width, and any other value is refused by the constructor (asserted)."""
    st = states(LOOPS + 1, scale=300.0)
    for kind in LINATTN_KINDS:
        torch.manual_seed(0)
        mod = LoopLinAttn(D, LOOPS, kind=kind)          # default num_heads
        assert mod.num_heads == 1 and mod.head_dim == D, \
            f"{kind}: default is num_heads={mod.num_heads}, head_dim={mod.head_dim}"
        k = mod._l2(mod._split(states(1)[0].float()))
        assert k.shape[-1] == D, f"{kind}: single-head key is not full-width"
        with torch.no_grad():
            mod.alpha.fill_(0.5)
        out = chain_combine(mod, st)
        assert out.shape == st[-1].shape and torch.isfinite(out).all(), \
            f"{kind}: single-head combine failed"
        for bad in (2, MULTIHEAD, D):
            raised = False
            try:
                LoopLinAttn(D, LOOPS, kind=kind, num_heads=bad)
            except AssertionError:
                raised = True
            assert raised, f"{kind}: num_heads={bad} was accepted, must be refused"
    print(f"PASS single-head at the full width {D} is the default and the only legal "
          f"num_heads; every other value is refused, for every kind")


def test_init_equivalence():
    """alpha init 0 => combine() returns h_r BITWISE (torch.equal), in bf16 and fp32:
    every reader starts as Plain Loop."""
    for kind in LINATTN_KINDS:
        mod = make(kind)
        for dtype in (torch.float32, torch.bfloat16):
            for n in range(1, LOOPS + 2):        # n = number of update sources
                st = states(n + 1, scale=300.0, dtype=dtype)
                out = chain_combine(mod, st)
                assert out.dtype == dtype, f"{kind}: {out.dtype} != {dtype}"
                assert torch.equal(out, st[-1]), \
                    f"{kind} {dtype} n={n}: not bitwise h_r at init"

    # the one documented exception, pinned at the bit level. Equal input and state make
    # the single update exactly zero, which is also the smallest legal history.
    mod = make("kda")
    neg = torch.tensor([[[-0.0, 1.0, 2.0, 3.0] * (D // 4)]])
    out = mod.combine([neg], inputs=[neg])
    assert torch.equal(out, neg), "signed zero should still compare equal"
    zero = neg == 0
    assert torch.equal(out.view(torch.int32)[~zero], neg.view(torch.int32)[~zero]), \
        "every NONZERO value must be bit-identical at init"
    assert bool(torch.signbit(neg[zero]).all()) and not bool(torch.signbit(out[zero]).any()), \
        "the -0.0 -> +0.0 exception no longer happens; tighten the docstring to bitwise"
    print("PASS init-equivalence holds bitwise via torch.equal (fp32 + bf16); "
          "-0.0 -> +0.0 is the only bit difference, and the dtype contract holds")


def test_grid_collapses():
    """Each reader must reduce to its neighbour when its extra gate is removed."""
    st = states(LOOPS + 1, scale=300.0)
    ds, qs = deltas_and_query(st)
    BIG = 30.0        # sigmoid(30) == 1.0 in fp32; the gate is off

    def read(kind, mut):
        mod = make(kind)
        with torch.no_grad():
            mut(mod)
        return mod._read(ds, qs)

    def kda_flat(m):
        m.b_beta.copy_(torch.linspace(-1.0, 1.0, m.b_beta.numel()).view_as(m.b_beta))
        m.b_gate.copy_(torch.full_like(m.b_gate, 0.3))

    def gdn_flat(m):
        m.b_beta.copy_(torch.linspace(-1.0, 1.0, m.b_beta.numel()).view_as(m.b_beta))
        m.b_gate.copy_(torch.full_like(m.b_gate, 0.3))

    torch.testing.assert_close(read("kda", kda_flat), read("gdn", gdn_flat),
                               rtol=1e-5, atol=1e-3,
                               msg="kda with a constant channel gate != gdn")

    def gdn_open(m):
        m.b_beta.copy_(torch.linspace(-1.0, 1.0, m.b_beta.numel()).view_as(m.b_beta))
        m.b_gate.fill_(BIG)

    def dn(m):
        m.b_beta.copy_(torch.linspace(-1.0, 1.0, m.b_beta.numel()).view_as(m.b_beta))

    torch.testing.assert_close(read("gdn", gdn_open), read("deltanet", dn),
                               rtol=1e-5, atol=1e-3,
                               msg="gdn with the gate open != deltanet")

    torch.testing.assert_close(read("gla", lambda m: m.b_gate.fill_(BIG)),
                               read("linattn", lambda m: None),
                               rtol=1e-5, atol=1e-3,
                               msg="gla with the gate open != linattn")

    gam = make("retnet").gamma.flatten()
    assert torch.all(gam > 0.96) and torch.all(gam < 1.0), gam
    assert float(gam.pow(LOOPS).min()) > 0.90, \
        "retnet's total decay over the loop axis should be < 10%"
    print("PASS grid collapses: kda->gdn->deltanet, gla->linattn, retnet is ~undecayed")


def test_batch_invariance():
    """combine() on a batch == combine() per sample. No cross-position/sample state."""
    for kind in LINATTN_KINDS:
        mod = randomise_(make(kind))
        ins = states(LOOPS, scale=300.0, seed=41)
        outs = states(LOOPS, scale=300.0, seed=42)
        full = mod.combine(outs, inputs=ins)
        for i in range(B):
            one = mod.combine([h[i:i + 1] for h in outs], inputs=[h[i:i + 1] for h in ins])
            torch.testing.assert_close(full[i:i + 1], one, rtol=1e-5, atol=1e-3,
                                       msg=f"{kind}: batch element {i} differs")
    print("PASS batch invariance (per-sample == batched) for every kind")


def test_depth_beyond_construction():
    """A history longer than the constructed depth reuses the last parameter row
    (clamped, not wrapped), and the factored read still matches the reference."""
    for kind in LINATTN_KINDS:
        mod = randomise_(make(kind))
        st = states(LOOPS + 5, scale=300.0)      # LOOPS+4 sources > LOOPS rows
        ds, qs = deltas_and_query(st)
        got = mod._read(ds, qs)
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
        st = states(LOOPS + 1, scale=1.0)
        opt = torch.optim.SGD(mod.parameters(), lr=1e-3)
        for step in (0, 1):
            opt.zero_grad(set_to_none=True)
            chain_combine(mod, st).square().mean().backward()
            for name, p in mod.named_parameters():
                assert p.grad is not None, f"{kind}: {name} has no grad"
                assert torch.isfinite(p.grad).all(), f"{kind}: {name} grad non-finite"
                assert bool(p.grad.abs().max() > 0), \
                    f"{kind}: {name} has a zero gradient at step {step}"
            opt.step()
    print("PASS every parameter is gradient-reachable")


def test_no_dead_gate_rows():
    """Every ROW of every gate parameter must be reachable."""
    for kind in LINATTN_KINDS:
        mod = randomise_(make(kind))
        with torch.no_grad():
            mod.alpha.fill_(0.5)
        chain_combine(mod, states(LOOPS + 1, scale=300.0)).square().sum().backward()
        for name, p in mod.named_parameters():
            if p.dim() == 0 or name == "alpha":
                continue
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
        out = chain_combine(mod, st)
        assert torch.isfinite(out).all(), f"A_log={A_log} dt={dt_bias}: output not finite"
        gate = mod._gate(1).detach()
        assert float(gate.min()) > 0.0, \
            f"A_log={A_log} dt={dt_bias}: decay underflowed to exactly 0"
        out.square().sum().backward()
        for name, p in mod.named_parameters():
            assert torch.isfinite(p.grad).all(), \
                f"A_log={A_log} dt={dt_bias}: {name} grad is not finite"
    for kind, name, val in (("mamba2", "dt_bias", 1e38), ("lact", "b_eta", 1e38)):
        mod = make(kind)
        with torch.no_grad():
            mod.alpha.fill_(0.5)
            getattr(mod, name).fill_(val)
        out = chain_combine(mod, st)
        assert torch.isfinite(out).all(), f"{kind}: {name}={val} gives a non-finite read"
        out.square().sum().backward()
        for pn, p in mod.named_parameters():
            assert torch.isfinite(p.grad).all(), f"{kind}: {name}={val} -> {pn} grad"
    print("PASS extreme dt / A_log / eta stay finite in forward AND backward, and the "
          "decay never reaches exactly 0")


def test_gate_bias_init_solves_for_transitions():
    """_gate_bias_init pins retention of the oldest source over max(R-1, 1) TRANSITIONS,
    one fewer than the R stored updates, and agrees with the oracle's derivation."""
    from loop_linattn_delta import _gate_bias_init, _GATE_RETENTION_INIT
    for loops in (1, 2, 3, 4, 16):
        bias = _gate_bias_init(loops)
        steps = max(loops - 1, 1)
        retention = (1.0 / (1.0 + math.exp(-bias))) ** steps        # float64
        assert abs(retention - _GATE_RETENTION_INIT) < 1e-12, (
            f"R={loops}: sigmoid({bias:.6f})^{steps} = {retention:.9f}, "
            f"expected {_GATE_RETENTION_INIT}")
        assert abs(bias - ref.gate_bias_init(loops)) < 1e-9, \
            f"R={loops}: implementation bias {bias} != oracle {ref.gate_bias_init(loops)}"
    # and the built gate rows carry exactly that bias, at the transition count
    for kind in ("gla", "gdn", "kda", "rwkv7"):
        m = make(kind)
        assert torch.allclose(m.b_gate, torch.full_like(m.b_gate,
                              _gate_bias_init(LOOPS))), f"{kind}: b_gate init"
    print("PASS _gate_bias_init solves sigmoid(b)^max(R-1,1) == 0.95, matches the "
          "oracle, and is what the gate rows are filled with")


def test_param_counts():
    rows = LOOPS              # one write per SOURCE = per update
    grows = max(LOOPS - 1, 1)  # one decay per TRANSITION -- one fewer again
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


def kda_paper_read(mod, st):
    """KDA in the PAPER's orientation, written without this file's transition helpers.

    KDA, as defined in Kimi Linear (Moonshot AI, 2025), keeps a key x value state and applies
        S_paper <- E D S_paper + beta k v^T,     E = I - beta k k^T,  D = Diag(a)
    and reads o = S_paper^T q. Independent of the file's left/right convention, so it
    pins the ORDER of the decay and the erase.
    """
    assert mod.kind == "kda"
    dfs = [(b - a).float() for a, b in zip(st[:-1], st[1:])]
    ks = [mod._l2(mod._split(d)) for d in dfs]
    vs = [mod._split(d) for d in dfs]
    q = mod._l2(mod._split(st[-1].float()))
    b, t, nh, dh = q.shape
    S = torch.zeros(b, t, nh, dh, dh)                  # [key, value]
    eye = torch.eye(dh).view(1, 1, 1, dh, dh)

    for r in range(len(dfs)):
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
    for n in range(1, LOOPS + 2):              # n = number of update sources
        st = states(n + 1, scale=300.0)
        ds, qs = deltas_and_query(st)
        torch.testing.assert_close(
            mod._read(ds, qs), kda_paper_read(mod, st), rtol=2e-4, atol=6e-2,
            msg=lambda m, n=n: f"kda n={n}: does not match KDA's own recurrence\n{m}")

    st = states(LOOPS + 1, scale=300.0)
    dfs = [(b - a).float() for a, b in zip(st[:-1], st[1:])]
    ks = [mod._l2(mod._split(d)) for d in dfs]
    vs = [mod._split(d) for d in dfs]
    q = mod._l2(mod._split(st[-1].float()))
    W = None
    for r in range(len(dfs)):
        beta = mod._beta(r)
        if W is not None:                              # ERASE first, then gate (wrong)
            kr = ks[r].unsqueeze(0)
            W = W - beta.unsqueeze(0) * (W * kr).sum(-1, keepdim=True) * kr
            W = mod._gate(r).unsqueeze(0) * W
        new = (beta * ks[r]).unsqueeze(0)
        W = new if W is None else torch.cat([W, new], 0)
    V = torch.stack(vs, 0)
    other = ((W * q.unsqueeze(0)).sum(-1, keepdim=True) * V).sum(0).reshape(st[-1].shape)
    gap = float((other - mod._read(dfs, st[-1])).detach().abs().max())
    assert gap > 6e-1, (
        f"the two gate/erase orders differ by only {gap:.3e} on this input, within an "
        f"order of magnitude of the correctness check's own atol (6e-2)")
    print(f"PASS kda == KDA's own recurrence; the wrong order differs by {gap:.3g}, "
          f"so the check has teeth")


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
            x = chain_combine(mod, hist)
        x.square().sum()
    return sum(total)


def test_scaling_differs_by_path():
    """The three code paths have DIFFERENT complexity in the iteration count: the DPLR
    kernel is O(R^3) saved activations over a whole forward, `lact` O(R^4), and `mesa`
    carries an n x n solve per junction."""
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
        f"lact ({l2:.1f}x) should grow FASTER than kda ({k2:.1f}x) per doubling")
    assert m2 > 4.0, f"mesa growth {ratios['mesa']} is not super-quadratic"
    assert m2 < k2, (
        f"mesa ({m2:.1f}x) should grow more slowly than kda ({k2:.1f}x)")
    body = ", ".join(f"{k}: x{a:.1f} then x{b:.1f}" for k, (a, b) in ratios.items())
    print(f"PASS saved activations scale differently per path ({body}) -- DPLR cubic, "
          f"lact steeper")


def test_compile_agrees():
    """torch.compile agrees with eager for every kind."""
    import torch._dynamo
    ins = states(LOOPS, scale=300.0, seed=41)
    outs = states(LOOPS, scale=300.0, seed=42)
    for kind in LINATTN_KINDS:
        torch._dynamo.reset()
        mod = randomise_(make(kind))
        eager = mod.combine(outs, inputs=ins)
        comp = torch.compile(mod.combine, fullgraph=True)(outs, inputs=ins)
        torch.testing.assert_close(comp, eager, rtol=1e-5, atol=1e-3,
                                   msg=f"{kind}: compiled != eager")
    print("PASS torch.compile agrees with eager for every kind")


# The ordered roster the runner iterates over.
ALL_TESTS = [
    test_matches_reference_oracle,
    test_combine_sources_are_block_deltas,
    test_matches_materialised_state,
    test_parameterisation_contract,
    test_parameter_effectiveness,
    test_all_zero_history_stays_finite,
    test_num_heads_forced_to_one,
    test_lact_zero_state_stays_finite_in_backward,
    test_lact_matches_the_materialised_fast_weight,
    test_lact_chunk_size_changes_the_answer,
    test_lact_global_eta_scale_is_weak_only_at_the_stream_scale,
    test_mesa_matches_the_dense_solution,
    test_mesa_ridge_floor_is_applied,
    test_kda_matches_the_paper_orientation,
    test_hgrn2_gate_telescopes,
    test_rwkv7_transition_is_additive_dplr,
    test_init_equivalence,
    test_grid_collapses,
    test_batch_invariance,
    test_depth_beyond_construction,
    test_gradients_reach_every_parameter,
    test_no_dead_gate_rows,
    test_extreme_gates_stay_finite,
    test_gate_bias_init_solves_for_transitions,
    test_scaling_differs_by_path,
    test_param_counts,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true",
                    help="also run the compiled-vs-eager check (slow)")
    a = ap.parse_args()

    torch.manual_seed(0)
    for t in ALL_TESTS:
        t()
    if a.compile:
        test_compile_agrees()
    print("\nALL LINATTN TESTS PASSED")


if __name__ == "__main__":
    main()
