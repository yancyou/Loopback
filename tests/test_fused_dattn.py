"""The fused Triton read of `delta_attnres` (fused_dattn.py) against the reference read (LoopDeltaAttnRes.combine
in looped_transformer.py), on this repo's own cell.

  1. fp32: out, the installed coefficients and every gradient (query, each delta through both the value and the
     RMSNorm-key path, proposal / h0 through base, the previous coefficients) agree with the reference to ~1e-5
     relative, over ragged token counts, 2..20 sources, boundary and in-iteration reads, d in {768, 1024, 1536, 3072}.
  2. bf16: fused and reference are both compared with an fp64 evaluation of the same bf16-rounded inputs; the fused
     error must not exceed the reference's by more than a small factor (the reference accumulates in bf16).
  3. init: zero query and cbar == prior give out == base and tc == c0 bitwise, as the reference guarantees.
  4. the whole model: LoopedTransformerLM with LOOP_DATTN_FUSED=1 against the reference read, fp32: bitwise at init
     (zero queries), ~1e-6 relative after the queries are perturbed, forward and every gradient.
  5. --bench: one unrolled pass (KR reads over 1..KR sources, forward + backward) at the 191M / 374M / 778M shapes,
     fused vs reference: wall time and peak memory.

Needs a GPU.  Run from the repo root:  python tests/test_fused_dattn.py [--bench]
"""
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fused_dattn as FD  # noqa: E402
import looped_transformer as M  # noqa: E402


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        raise SystemExit(1)


def make(b, t, d, n, dtype, seed, zero_query=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    deltas = [(torch.randn(b, t, d, device="cuda", generator=g) * (0.5 + 0.1 * i)).to(dtype).requires_grad_() for i in range(n)]
    proposal = torch.randn(b, t, d, device="cuda", generator=g).to(dtype).requires_grad_()
    h0 = torch.randn(b, t, d, device="cuda", generator=g).to(dtype).requires_grad_()
    q = torch.zeros(d, device="cuda") if zero_query else torch.randn(d, device="cuda", generator=g) * 0.3
    prior = torch.rand(n - 1, device="cuda", generator=g) * 0.9 + 0.1
    coeffs = prior.view(1, 1, -1) + 0.2 * torch.randn(b, t, n - 1, device="cuda", generator=g)
    up = torch.randn(b, t, d, device="cuda", generator=g)
    up2 = torch.randn(b, t, n, device="cuda", generator=g)
    return deltas, proposal, h0, q, prior, coeffs, up, up2


def run(d, n, deltas, proposal, h0, q, prior, coeffs, up, up2, fused, boundary, beta=0.75):
    for x in deltas + [proposal, h0]:
        x.grad = None
    # n sources = step n - 1; block n - 1 of the first iteration is a boundary read with n blocks per iteration, in-iteration with n + 1
    mod = M.LoopDeltaAttnRes(d, num_loops=2, num_blocks=n if boundary else n + 1, beta=beta, eps=1e-6).cuda()
    with torch.no_grad():
        mod.query[n - 2].copy_(q)
    mod.query.grad = None
    cf = coeffs.detach().clone().requires_grad_()
    step = n - 1
    if fused:
        base = proposal + beta * (h0 - proposal) if boundary else proposal
        base_coeffs = torch.cat((cf, torch.ones(*proposal.shape[:-1], 1, device="cuda")), dim=-1)
        next_prior = torch.cat((prior, torch.ones(1, device="cuda")), dim=-1)
        if boundary:
            base_coeffs, next_prior = (1.0 - beta) * base_coeffs, (1.0 - beta) * next_prior
        rstds = [FD.rstd(x, 1e-6) for x in deltas]
        assert FD.fused_read_supported(base, deltas, rstds, base_coeffs)
        out, tc = FD.FusedDattnRead.apply(mod.query[step - 1], base, base_coeffs, next_prior, *deltas, *rstds)
    else:
        out, tc, _ = mod.combine(proposal, h0, deltas, cf, prior, 0, n - 1)
    ((out.float() * up).sum() + (tc * up2).sum()).backward()
    return (out.detach().double(), tc.detach().double(), mod.query.grad[step - 1].detach().double().clone(),
            [x.grad.detach().double().clone() for x in deltas], proposal.grad.detach().double().clone(),
            h0.grad.detach().double().clone() if h0.grad is not None else torch.zeros_like(proposal, dtype=torch.float64),
            cf.grad.detach().double().clone())


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def cmp(ra, rb):
    o, tc, gq, gd, gp, gh, gc = ra
    o2, tc2, gq2, gd2, gp2, gh2, gc2 = rb
    return dict(out=rel(o, o2), tc=rel(tc, tc2), dq=rel(gq, gq2), dv=max(rel(a, b) for a, b in zip(gd, gd2)),
                dprop=rel(gp, gp2), dh0=rel(gh, gh2) if gh2.abs().max() > 0 else 0.0, dcoef=rel(gc, gc2))


def correctness():
    print("1. fp32: fused vs reference", flush=True)
    for (b, t, d) in ((2, 64, 768), (2, 37, 1024), (1, 50, 1536), (1, 9, 3072)):
        for n in (2, 3, 8, 16, 20):
            for boundary in (False, True):
                args = make(b, t, d, n, torch.float32, seed=1000 * n + d + int(boundary))
                e = cmp(run(d, n, *args, fused=True, boundary=boundary), run(d, n, *args, fused=False, boundary=boundary))
                check(f"b{b} t{t} d{d} n{n} {'boundary' if boundary else 'in-iteration'}", max(e.values()) < 3e-5,
                      " ".join(f"{k} {v:.1e}" for k, v in e.items()))
    print("2. bf16: both against fp64 on the same bf16-rounded inputs", flush=True)
    for (b, t, d) in ((2, 256, 1024), (2, 128, 1536), (4, 128, 768)):
        for n in (4, 16, 20):
            args = make(b, t, d, n, torch.bfloat16, seed=7 * n + d)
            r_ref = run(d, n, *args, fused=False, boundary=True)
            r_fus = run(d, n, *args, fused=True, boundary=True)
            deltas, proposal, h0, q, prior, coeffs, up, up2 = args
            args64 = ([x.detach().double().requires_grad_() for x in deltas], proposal.detach().double().requires_grad_(),
                      h0.detach().double().requires_grad_(), q.double(), prior.double(), coeffs.double(), up.double(), up2.double())
            r_64 = run(d, n, *args64, fused=False, boundary=True)
            er, ef = cmp(r_ref, r_64), cmp(r_fus, r_64)
            check(f"b{b} t{t} d{d} n{n}", all(ef[k] <= 2.0 * er[k] + 2e-3 for k in er),
                  "err vs fp64: fused " + " ".join(f"{k} {v:.1e}" for k, v in ef.items()) + " | reference " + " ".join(f"{v:.1e}" for v in er.values()))
    print("3. init: zero query, cbar == prior -> out == base, tc == c0 bitwise", flush=True)
    for dtype in (torch.float32, torch.bfloat16):
        for n in (2, 9, 16):
            b, t, d = 2, 40, 768
            deltas, proposal, h0, q, prior, coeffs, up, up2 = make(b, t, d, n, dtype, seed=n, zero_query=True)
            coeffs = prior.view(1, 1, -1).expand(b, t, -1).contiguous()
            for boundary in (False, True):
                base = proposal + 0.75 * (h0 - proposal) if boundary else proposal
                base_coeffs = torch.cat((coeffs, torch.ones(b, t, 1, device="cuda")), -1)
                next_prior = torch.cat((prior, torch.ones(1, device="cuda")), -1)
                if boundary:
                    base_coeffs, next_prior = 0.25 * base_coeffs, 0.25 * next_prior
                rstds = [FD.rstd(x, 1e-6) for x in deltas]
                with torch.no_grad():
                    out, tc = FD.FusedDattnRead.apply(q, base, base_coeffs, next_prior, *deltas, *rstds)
                check(f"{dtype} n{n} {'boundary' if boundary else 'in-iteration'}",
                      torch.equal(out, base) and torch.equal(tc, next_prior.view(1, 1, -1).expand(b, t, -1)))


def whole_model(role, path):
    """Run in a subprocess (the switch is read once per process): --save writes the reference outputs, --compare
    loads them and runs the fused path."""
    cfg = M.LoopedTransformerConfig(vocab_size=128, hidden_size=64, num_hidden_layers=12, num_heads=4,
                                    num_prelude_layers=2, num_coda_layers=2, num_middle_param_layers=2,
                                    max_position_embeddings=64, loop_residual_mode="delta_attnres", skip0_beta=0.75)
    torch.manual_seed(0); ids = torch.randint(0, 128, (2, 24), device="cuda")

    def one(perturb):
        torch.manual_seed(1); m = M.LoopedTransformerLM(cfg).to("cuda", torch.float32).train()
        if perturb:
            g = torch.Generator(device="cpu").manual_seed(2)
            with torch.no_grad():
                m.model.loop_dattn.query.copy_(torch.randn(m.model.loop_dattn.query.shape, generator=g) * 0.2)
        out = m(ids, labels=ids); out["loss"].backward()
        return dict(logits=out["logits"].detach().cpu(), loss=out["loss"].detach().cpu(),
                    grads={n: p.grad.detach().cpu() for n, p in m.named_parameters() if p.grad is not None})
    res = {"init": one(False), "trained": one(True)}
    if role == "save":
        torch.save(res, path); return
    ref = torch.load(path); assert M._fused_dattn(), "LOOP_DATTN_FUSED=1 did not activate the fused read"
    for tag in ("init", "trained"):
        a, b = res[tag], ref[tag]
        fwd_same = torch.equal(a["logits"], b["logits"]) and torch.equal(a["loss"], b["loss"])
        e_l = rel(a["logits"], b["logits"]); e_g = max(rel(a["grads"][k], b["grads"][k]) for k in b["grads"])
        check(f"whole model, {tag}: forward bitwise={fwd_same}, logits rel {e_l:.1e}, max grad rel {e_g:.1e}",
              (fwd_same and e_g < 1e-5) if tag == "init" else (e_l < 1e-5 and e_g < 1e-4))


def bench(d, kr, b, t, reps=3):
    print(f"5. bench d{d} KR{kr} b{b} t{t} bf16: {kr} reads over 1..{kr} sources, fwd+bwd", flush=True)
    res = {}
    for fused in (True, False):
        mod = M.LoopDeltaAttnRes(d, num_loops=2, num_blocks=kr // 2, beta=0.75, eps=1e-6).cuda()
        with torch.no_grad():
            mod.query.normal_(0, 0.05)
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        times = []
        for rep in range(reps + 1):
            h0 = torch.randn(b, t, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            x, deltas, keys, coeffs, prior = h0, [], [], None, None
            torch.cuda.synchronize(); t0 = time.time(); loss = 0.0
            for s in range(kr):
                y = x * 1.001 + 0.01 * torch.tanh(x)
                deltas.append(y - x)
                l, blk = divmod(s, kr // 2)
                if fused:
                    at_boundary = blk == kr // 2 - 1
                    base = y + 0.75 * (h0 - y) if at_boundary else y
                    one_tok = torch.ones(b, t, 1, device="cuda")
                    base_coeffs = one_tok if coeffs is None else torch.cat((coeffs, one_tok), -1)
                    next_prior = torch.ones(1, device="cuda") if prior is None else torch.cat((prior, torch.ones(1, device="cuda")), -1)
                    if at_boundary:
                        base_coeffs, next_prior = 0.25 * base_coeffs, 0.25 * next_prior
                    keys.append(FD.rstd(deltas[-1], 1e-6))
                    if s == 0:
                        x, coeffs, prior = base, base_coeffs, next_prior
                    else:
                        x, coeffs = FD.FusedDattnRead.apply(mod.query[s - 1], base, base_coeffs, next_prior, *deltas, *keys); prior = next_prior
                else:
                    x, coeffs, prior = mod.combine(y, h0, deltas, coeffs, prior, l, blk, keys=keys)
                loss = loss + x.float().mean()
            loss.backward(); torch.cuda.synchronize()
            if rep:
                times.append(time.time() - t0)
            del deltas, keys, coeffs, prior, loss, x, y, h0
        res[fused] = (sum(times) / len(times), torch.cuda.max_memory_allocated() / 2**30)
        print(f"   {'fused    ' if fused else 'reference'}: {res[fused][0] * 1e3:8.1f} ms   peak {res[fused][1]:.1f} GiB", flush=True)
    print(f"   speed-up {res[False][0] / res[True][0]:.2f}x", flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    if "--save" in sys.argv:
        whole_model("save", sys.argv[sys.argv.index("--save") + 1]); sys.exit()
    if "--compare" in sys.argv:
        whole_model("compare", sys.argv[sys.argv.index("--compare") + 1]); sys.exit()
    if "--bench" in sys.argv:
        b = int(os.environ.get("BENCH_B", "8"))
        bench(768, 16, b, 4096); bench(1024, 20, b, 4096); bench(1536, 16, b, 4096); sys.exit()
    print("fused module:", FD.__file__, "| model module:", M.__file__, flush=True)
    correctness()
    print("4. whole model: LOOP_DATTN_FUSED=1 against the reference read", flush=True)
    here = os.path.abspath(__file__); ref = "/tmp/test_fused_dattn_ref.pt"
    env = dict(os.environ); env["LOOP_DATTN_FUSED"] = "0"
    subprocess.run([sys.executable, here, "--save", ref], check=True, env=env)
    env["LOOP_DATTN_FUSED"] = "1"
    subprocess.run([sys.executable, here, "--compare", ref], check=True, env=env)
    print("ALL OK", flush=True)
