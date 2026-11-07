"""The stacked layout of Loopback's fused read (fused_dattn.DattnStack, the trainer's default with LOOP_DATTN_FUSED=1)
against the list layout and the reference read, on a GPU with reference attention (deterministic, so "bitwise" means
bitwise):
  1. the read cell (LoopDeltaAttnRes.combine) in a K x R chain with a learned write per block application:
     stack vs list -- outputs bitwise; gradients bitwise in bf16, within 1e-6 in fp32 (training and eval);
     stack vs the reference read -- fp32 round-off;
  2. a small whole trainer model, fp32 and bf16: reference read / list / stack -- logits, loss and every parameter
     gradient: stack == list bitwise in bf16 (the training dtype); in fp32 the logits bitwise and the gradients within
     1e-6 (the split backward contracts a multiply-add differently); both within fp32 round-off of the reference read.
Run:  LOOP_DATTN_FUSED=1 python tests/test_dattn_stack.py
"""
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "trainer"))
os.environ["LOOP_DATTN_FUSED"] = "1"
import flame.custom_models.looped_transformer.modeling_looped_transformer as MLT  # noqa: E402
from flame.custom_models.looped_transformer import LoopedTransformerConfig, LoopedTransformerForCausalLM  # noqa: E402
from flame.custom_models.looped_transformer import fused_dattn as FD  # noqa: E402

if not torch.cuda.is_available():
    print("needs a GPU"); sys.exit(0)


def ref_attn(q, k, v, causal=False, **kw):
    import torch.nn.functional as F
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if k.shape[1] != q.shape[1]:
        g = q.shape[1] // k.shape[1]; k = k.repeat_interleave(g, 1); v = v.repeat_interleave(g, 1)
    m = None
    if causal:
        ql, kl = q.size(-2), k.size(-2)
        m = torch.arange(kl, device=q.device).view(1, kl) <= (torch.arange(ql, device=q.device).view(ql, 1) + max(kl - ql, 0))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=False).transpose(1, 2).contiguous()


MLT.flash_attn_func = ref_attn
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {name} {extra}", flush=True)


def set_path(fused, stack):
    os.environ["LOOP_DATTN_FUSED"], os.environ["LOOP_DATTN_STACK"] = ("1" if fused else "0"), ("1" if stack else "0")
    MLT._FUSED_DATTN = None


def rel(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-30)).item()


def chain(path, K, R, N=2048, D=256, dtype=torch.bfloat16, grad=True, seed=0):
    set_path(path != "reference", path == "stack")
    torch.manual_seed(seed)
    cell = MLT.LoopDeltaAttnRes(SimpleNamespace(hidden_size=D), R, K, beta=0.75, eps=1e-6).cuda()
    with torch.no_grad():
        cell.query.normal_(0, 0.5)
    Ws = [(torch.randn(D, D, device="cuda") * 0.03).requires_grad_(grad) for _ in range(K * R)]
    h0 = (torch.randn(1, N, D, device="cuda") * 0.5).to(dtype).requires_grad_(grad)
    stack = FD.DattnStack(K * R, (1, N, D), dtype, "cuda") if path == "stack" else None
    h, deltas, keys, coeffs, prior = h0, [], [], None, None
    with torch.set_grad_enabled(grad):
        for s in range(K * R):
            l, blk = divmod(s, K)
            x = h
            hs = (h.float() @ Ws[s]).to(dtype) + h
            deltas.append(stack.write(s, hs, x) if stack is not None else hs - x)
            h, coeffs, prior = cell.combine(hs, h0, deltas, coeffs, prior, l, blk, keys=keys, stack=stack)
        loss = (h.float() ** 2).mean()
        if grad:
            loss.backward()
    grads = [w.grad.clone() for w in Ws] + [h0.grad.clone(), cell.query.grad.clone()] if grad else []
    return loss.item(), h.float(), grads


print("1. the read cell, K x R chains", flush=True)
for K, R, dtype in ((2, 4, torch.bfloat16), (3, 3, torch.bfloat16), (2, 4, torch.float32)):
    ll, hl, gl = chain("list", K, R, dtype=dtype)
    ls, hs, gs = chain("stack", K, R, dtype=dtype)
    lr, hr, gr = chain("reference", K, R, dtype=dtype)
    gworst = max(rel(a, b) for a, b in zip(gs, gl))
    if dtype == torch.bfloat16:   # the training dtype: bitwise, output and gradients
        check(f"K{K} R{R} bfloat16: stack == list, output and all {len(gl)} gradients bitwise",
              torch.equal(hl, hs) and all(torch.equal(a, b) for a, b in zip(gl, gs)))
    else:   # fp32: the output bitwise; the gradients differ by the FMA contraction of the split backward (~1e-7)
        check(f"K{K} R{R} float32: stack == list, output bitwise {torch.equal(hl, hs)}, worst grad rel {gworst:.1e}",
              torch.equal(hl, hs) and gworst < 1e-6)
    tol = 1e-5 if dtype == torch.float32 else 3e-2
    check(f"K{K} R{R} {str(dtype)[6:]}: stack vs reference read: output rel {rel(hs, hr):.1e}, worst grad rel "
          f"{max(rel(a, b) for a, b in zip(gs, gr)):.1e}", rel(hs, hr) < tol and max(rel(a, b) for a, b in zip(gs, gr)) < tol)
_, hl, _ = chain("list", 3, 3, grad=False); _, hs, _ = chain("stack", 3, 3, grad=False)
check("K3 R3 eval (no grad): stack == list bitwise", torch.equal(hl, hs))

print("2. a whole trainer model", flush=True)
CFG = dict(hidden_size=64, num_hidden_layers=10, num_heads=4, num_kv_heads=2, max_position_embeddings=64, vocab_size=128,
           rope_theta=10000.0, use_cache=False, fuse_norm=False, fuse_swiglu=False, fuse_cross_entropy=False,
           fuse_linear_cross_entropy=False, num_prelude_layers=2, num_coda_layers=2, num_middle_param_layers=2,
           loop_residual_mode="delta_attnres", skip0_beta=0.75)   # P2 K2 R3 C2
torch.manual_seed(1)
ids = torch.randint(0, CFG["vocab_size"], (2, 48), device="cuda")


def model_run(path, dtype):
    set_path(path != "reference", path == "stack")
    torch.manual_seed(0)
    m = LoopedTransformerForCausalLM(LoopedTransformerConfig(**CFG)).cuda().to(dtype).train()
    with torch.no_grad():
        m.model.loop_dattn.query.normal_(0, 0.3)
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    return out.loss.float().item(), out.logits.float(), {n: p.grad.float().clone() for n, p in m.named_parameters() if p.grad is not None}


for dtype in (torch.float32, torch.bfloat16):
    ll, Ll, gl = model_run("list", dtype)
    ls, Ls, gs = model_run("stack", dtype)
    lr, Lr, gr = model_run("reference", dtype)
    gworst = max(rel(gs[n], gl[n]) for n in gl)
    if dtype == torch.bfloat16:
        check(f"bfloat16: stack == list: loss {ls:.6f} vs {ll:.6f}, logits and all {len(gl)} gradients bitwise",
              ls == ll and torch.equal(Ls, Ll) and all(torch.equal(gs[n], gl[n]) for n in gl))
    else:
        check(f"float32: stack == list: logits bitwise {torch.equal(Ls, Ll)}, worst grad rel {gworst:.1e}",
              torch.equal(Ls, Ll) and gworst < 1e-6)
    worst = max(rel(gs[n], gr[n]) for n in gr)
    tol = 1e-5 if dtype == torch.float32 else 3e-2
    check(f"{str(dtype)[6:]}: stack vs reference read: logits rel {rel(Ls, Lr):.1e}, worst grad rel {worst:.1e}",
          rel(Ls, Lr) < tol and worst < tol)
print("ALL OK" if ok else "SOME CHECKS FAILED", flush=True)
sys.exit(0 if ok else 1)
