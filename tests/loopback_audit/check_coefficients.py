"""The Loopback read at the model level (the training model and the plain-PyTorch port), fp64: the shared blocks are
replaced by controllable maps x -> x + D_s (D_s fixed random, independent of x), so every state is an exact linear
combination h0 + sum_i c_i D_i whose coefficients are recovered per token by least squares and compared with
  - zero queries: c_i = lam^(floor((s+1)/K) - floor(i/K)) at every read s (decay after every iteration, including
    the last before the coda), h0 coefficient exactly 1, the prior the cell carries == the paper's schedule (C1, C2)
  - learned queries: the true coefficients at every read == the coefficients the cell tracks (across iteration
    boundaries too), h0 coefficient 1, sum_i c_i == B_s, and the queries move the coefficients (C4, C5)
  - beta = 0 and zero queries: bitwise Plain Loop (mode null) (C6)
Topologies [1 -> 4x4 -> 1] (the 191M shape) and [1 -> 3x2 -> 1]; lam in {0.25, 0.5}.
"""

import torch

from common import Report, build_flame, build_port, paper_prior, patch_kernels, tiny_raw

patch_kernels()
rep = Report("check_coefficients")
B, T, D_ = 2, 5, 64


def inner(model, impl):
    return model.model


def install(model, impl, Pp, K, R, Dl, rec):
    """Replace the KR middle blocks by x -> x + D_s and the first coda block by an identity that records its input."""
    layers = inner(model, impl).layers
    for j in range(K * R):
        layer = layers[Pp + j]

        def fwd(x, _j=j, **kw):
            rec["inputs"].append(x.detach().clone())
            return x + Dl[_j]
        layer.forward = fwd
    coda = layers[Pp + K * R]

    def coda_fwd(x, **kw):
        rec["loop_out"] = x.detach().clone()
        return x
    coda.forward = coda_fwd


def wrap_combine(model, impl, rec):
    cell = inner(model, impl).loop_dattn
    orig = cell.combine

    def combine(*a, **kw):
        out, coeffs, prior = orig(*a, **kw)
        rec["coeffs"].append(coeffs.detach().clone())
        rec["prior"].append(prior.detach().clone())
        return out, coeffs, prior
    cell.combine = combine


def states(rec, K, R):
    xs = rec["inputs"] + [rec["loop_out"]]
    assert len(xs) == K * R + 1, len(xs)
    return xs   # xs[0] = h0, xs[s+1] = state after read s


def lstsq(x, h0, Ds):
    A = torch.stack([h0] + Ds, dim=-1)
    b, t, d, m = A.shape
    sol = torch.linalg.lstsq(A.reshape(b * t, d, m), x.reshape(b * t, d, 1)).solution.reshape(b, t, m)
    resid = (torch.einsum("btdm,btm->btd", A, sol) - x).abs().max().item()
    return sol[..., 0], sol[..., 1:], resid


def run(impl, raw, mode, beta, Dl, ids, query=None):
    over = dict(loop_residual_mode=mode, skip0_beta=beta)
    model = (build_flame if impl == "flame" else build_port)(raw, seed=0, dtype=torch.float64, **over)
    Pp, K = raw["num_prelude_layers"], raw["num_middle_param_layers"]
    R = (raw["num_hidden_layers"] - Pp - raw["num_coda_layers"]) // K
    rec = dict(inputs=[], coeffs=[], prior=[])
    install(model, impl, Pp, K, R, Dl, rec)
    if mode == "delta_attnres":
        with torch.no_grad():
            if query is None:
                inner(model, impl).loop_dattn.query.zero_()
            else:
                inner(model, impl).loop_dattn.query.copy_(query)
        wrap_combine(model, impl, rec)
    with torch.no_grad():
        if impl == "flame":
            model(input_ids=ids)
        else:
            model(ids)
    return states(rec, K, R), rec


for (K, R) in [(4, 4), (3, 2)]:
    raw = tiny_raw(num_hidden_layers=1 + K * R + 1, num_prelude_layers=1, num_coda_layers=1, num_middle_param_layers=K)
    gen = torch.Generator().manual_seed(5 + K)
    Dl = [torch.randn(B, T, D_, generator=gen, dtype=torch.float64) * (0.5 + 0.1 * s) for s in range(K * R)]
    ids = torch.randint(0, raw["vocab_size"], (B, T), generator=gen)
    query = torch.randn(K * R - 1, D_, generator=gen, dtype=torch.float64) * 0.6
    for impl in ("flame", "port"):
        for lam in (0.25, 0.5):
            beta = 1.0 - lam
            tag = f"[{impl}] [1->{K}x{R}->1] lam={lam}"
            # ---- C1 / C2: zero queries = the decay-only loop: coefficients lam^(iteration ends since the write),
            #      h0 coefficient 1, at every read including the last before the coda ----
            xs_d0, rec0 = run(impl, raw, "delta_attnres", beta, Dl, ids, query=None)
            h0 = xs_d0[0]
            ok, worst, worst_a = True, 0.0, 0.0
            for s in range(K * R):
                a, c, resid = lstsq(xs_d0[s + 1], h0, Dl[: s + 1])
                want = paper_prior(K, R, lam, s)
                e = (c - want).abs().max().item()
                worst, worst_a = max(worst, e), max(worst_a, (a - 1).abs().max().item())
                ok &= e < 1e-5 and (a - 1).abs().max().item() < 1e-5 and resid < 1e-6
            last = (xs_d0[-1] - (h0 + sum(lam ** (R - i // K) * Dl[i] for i in range(K * R)))).abs().max().item()
            rep.check(f"C1 zero queries: state after every read == h0 + sum lam^(#iteration ends since write) d_i, h0 coeff 1 {tag}",
                      ok and last < 1e-9, f"max coeff err {worst:.1e}, max|a-1| {worst_a:.1e}; last state closed-form err {last:.1e}")
            okp = all((p.double() - paper_prior(K, R, lam, s)).abs().max().item() < 1e-15 for s, p in enumerate(rec0["prior"]))
            rep.check(f"C2 prior inside the model == eq:boundary-schedule at every read {tag}", okp)
            # ---- C4: nonzero query, true coefficients by least squares == tracked coefficients ----
            xs_dq, recq = run(impl, raw, "delta_attnres", beta, Dl, ids, query=query)
            ok, worst, worst_a, worst_res = True, 0.0, 0.0, 0.0
            for s in range(K * R):
                a, c, resid = lstsq(xs_dq[s + 1], h0, Dl[: s + 1])
                e = (c - recq["coeffs"][s]).abs().max().item()
                worst, worst_a, worst_res = max(worst, e), max(worst_a, (a - 1).abs().max().item()), max(worst_res, resid)
                ok &= e < 1e-5 and (a - 1).abs().max().item() < 1e-5 and resid < 1e-6
            rep.check(f"C4 delta_attnres q!=0: true per-delta coefficients == tracked coeffs at every read, h0 coeff 1 {tag}", ok,
                      f"max|c_true - c_tracked| {worst:.1e}, max|a-1| {worst_a:.1e}, lstsq resid {worst_res:.1e}")
            # the tracked coefficients sum to B_s at every read (mass preserved) and differ from the prior (query active)
            ok_mass = all((c.sum(-1) - p.sum()).abs().max().item() < 1e-5 for c, p in zip(recq["coeffs"], recq["prior"]))
            moved = any((c - p).abs().max().item() > 1e-3 for c, p in zip(recq["coeffs"][1:], recq["prior"][1:]))
            rep.check(f"C4 sum_i c_i == B_s at every read, and the query actually moves the coefficients {tag}", ok_mass and moved)
            # ---- C5: across a boundary the newest delta's true coefficient is lam * (its target) ... i.e. after the
            # boundary read the coefficient of the newest delta equals the tracked value (not 1, not lam * 1 twice) ----
            ok5 = True
            for s in range(K - 1, K * R, K):
                a, c, _ = lstsq(xs_dq[s + 1], h0, Dl[: s + 1])
                ok5 &= (c[..., -1] - recq["coeffs"][s][..., -1]).abs().max().item() < 1e-9
                # and for the zero-query model the newest one is exactly lam at the boundary read
                a0, c0_, _ = lstsq(xs_d0[s + 1], h0, Dl[: s + 1])
                ok5 &= (c0_[..., -1] - lam).abs().max().item() < 1e-10
            rep.check(f"C5 boundary reads: newest delta enters at lam (q=0) / at its tracked target (q!=0); no second decay {tag}", ok5)
        # ---- C6: beta = 0 and zero queries == Plain Loop (mode None) bitwise ----
        xs_plain, _ = run(impl, raw, None, 0.0, Dl, ids)
        xs_d00, _ = run(impl, raw, "delta_attnres", 0.0, Dl, ids, query=None)
        ok = all(torch.equal(x, y) for x, y in zip(xs_d00, xs_plain))
        rep.check(f"C6 Loopback beta=0 q=0 == Plain Loop (mode null): all states bitwise [{impl}] [1->{K}x{R}->1]", ok)

raise SystemExit(rep.close())
