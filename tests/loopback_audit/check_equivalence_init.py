"""Equivalences on the real model (fp32, torch reference kernels) and the query's parameters and init:
  C6   skip0_beta = 0 and zero queries == Plain Loop (mode null): logits, loss and every shared gradient bitwise, on
       a small random model (both implementations, weights randomised away from init) and on the real 191M configs
       lt124m_ours_p2k4l4c2 (beta 0) vs lt124m_plain_p2k4l4c2 built from torch.manual_seed(0)
  C7   the query tensor has (KR-1) d entries (11,520 at 191M; 19 x 1,024 at 374M), is zero at init in both
       implementations and gets a live gradient there, the first read uses no query (row 0 is first used at s = 1),
       and the trainer's meta-device path (meta -> _is_hf_initialized=False -> to_empty -> post_init) zeroes it and
       initialises every parameter.
"""
import sys

import torch

from common import (M, TRAINER, Report, build_flame, build_port, load_cfg, patch_kernels, randomize_, tiny_raw)

patch_kernels()
rep = Report("check_equivalence_init")


def fwd_bwd(model, impl, ids):
    model.train()
    model.zero_grad(set_to_none=True)
    if impl == "flame":
        o = model(input_ids=ids, labels=ids)
        logits, loss = o.logits, o.loss
    else:
        o = model(ids, labels=ids)
        logits, loss = o["logits"], o["loss"]
    loss.backward()
    grads = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in model.named_parameters()}
    return logits.detach().clone(), loss.detach().clone(), grads


def compare(tag, impl, m_a, m_b, ids, exclude=("loop_dattn.query",)):
    """m_a: the Loopback model, m_b: the reference (Plain Loop). Shared params by name."""
    la, lossa, ga = fwd_bwd(m_a, impl, ids)
    lb, lossb, gb = fwd_bwd(m_b, impl, ids)
    shared = [n for n in gb if n in ga]
    extra_a = [n for n in ga if n not in gb]
    ok_logits, ok_loss = torch.equal(la, lb), torch.equal(lossa, lossb)
    ok_grads, worst = True, (None, 0.0)
    n_cmp = 0
    for n in shared:
        if ga[n] is None and gb[n] is None:
            continue
        n_cmp += 1
        if ga[n] is None or gb[n] is None or not torch.equal(ga[n], gb[n]):
            ok_grads = False
            d = float("inf") if (ga[n] is None or gb[n] is None) else (ga[n] - gb[n]).abs().max().item()
            if d > worst[1]:
                worst = (n, d)
    qg = [ga[n] for n in extra_a if ga[n] is not None]
    qinfo = ", ".join(f"{n}: grad |.|max={ga[n].abs().max().item():.2e} finite={torch.isfinite(ga[n]).all().item()}" for n in extra_a)
    rep.check(f"{tag} [{impl}] logits bitwise", ok_logits, f"max|diff| {(la - lb).abs().max().item():.1e}")
    rep.check(f"{tag} [{impl}] loss bitwise", ok_loss, f"{lossa.item():.9f} vs {lossb.item():.9f}")
    rep.check(f"{tag} [{impl}] all {n_cmp} shared-parameter gradients bitwise", ok_grads, f"worst {worst}")
    rep.note(f"    extra params of the delta_attnres model: {qinfo or 'none'}")
    return qg


# ----------------------------------------------------------------------------------------------------------------
# small random models
# ----------------------------------------------------------------------------------------------------------------
raw = tiny_raw(num_hidden_layers=1 + 2 * 3 + 1, num_prelude_layers=1, num_coda_layers=1, num_middle_param_layers=2)   # [1->2x3->1]
ids = torch.randint(0, raw["vocab_size"], (2, 16), generator=torch.Generator().manual_seed(3))
for impl, build in (("flame", build_flame), ("port", build_port)):
    for beta, ref_mode, label in ((0.0, None, "C6 small: Loopback(beta=0, q=0) == Plain Loop (mode null)"),):
        m_d = build(raw, seed=0, loop_residual_mode="delta_attnres", skip0_beta=beta)
        m_r = build(raw, seed=0, loop_residual_mode=ref_mode, skip0_beta=beta)
        # same weights, exercised away from init (the port test's randomisation), query re-zeroed
        randomize_(m_d, torch.Generator().manual_seed(1234))
        with torch.no_grad():
            m_d.model.loop_dattn.query.zero_()
        m_r.load_state_dict({k: v for k, v in m_d.state_dict().items() if k in m_r.state_dict()}, strict=True)
        sd_d, sd_r = m_d.state_dict(), m_r.state_dict()
        same = all(torch.equal(sd_d[k], sd_r[k]) for k in sd_r)
        rep.check(f"{label} [{impl}] shared weights identical after randomisation", same)
        compare(label, impl, m_d, m_r, ids)
    # the query gets a finite, nonzero gradient once it is at zero but the model is away from init (live routing)
    m_d = build(raw, seed=0, loop_residual_mode="delta_attnres", skip0_beta=0.75)
    randomize_(m_d, torch.Generator().manual_seed(1234))
    with torch.no_grad():
        m_d.model.loop_dattn.query.zero_()
    _, _, g = fwd_bwd(m_d, impl, ids)
    qg = g["model.loop_dattn.query"]
    rep.check(f"C7 [{impl}] query gradient at q=0 is finite and every row nonzero (live)", qg is not None and torch.isfinite(qg).all().item() and bool((qg.norm(dim=-1) > 0).all()),
              f"row norms {[f'{v:.2e}' for v in qg.norm(dim=-1).tolist()]}")
    # the first read (s = 0, one source) uses no query: row 0 is first consumed at s = 1
    m_d.eval()
    Pp = raw["num_prelude_layers"]
    with torch.no_grad():
        q_rand = torch.randn_like(m_d.model.loop_dattn.query) * 0.5
        m_d.model.loop_dattn.query.zero_()
        hs0 = (m_d(input_ids=ids, output_hidden_states=True).hidden_states if impl == "flame" else m_d(ids, output_hidden_states=True)["hidden_states"])
        m_d.model.loop_dattn.query[0].copy_(q_rand[0])
        hs1 = (m_d(input_ids=ids, output_hidden_states=True).hidden_states if impl == "flame" else m_d(ids, output_hidden_states=True)["hidden_states"])
    # hidden_states[j] = the input of layer j; [Pp+1] = state after read s=0, [Pp+2] = state after read s=1
    rep.check(f"C7 [{impl}] first read (s=0) uses no query: state after read 0 unchanged by query row 0, state after read 1 changed",
              torch.equal(hs0[Pp + 1], hs1[Pp + 1]) and not torch.equal(hs0[Pp + 2], hs1[Pp + 2]))

# ----------------------------------------------------------------------------------------------------------------
# the real 191M configs (flame), fp32: construction under the same seed, then forward/backward on 1 x 32 tokens
# ----------------------------------------------------------------------------------------------------------------
ours = load_cfg("191m_d768x20/lt124m_ours_p2k4l4c2_4klen.json")
plain = load_cfg("191m_d768x20/lt124m_plain_p2k4l4c2_4klen.json")
m374 = load_cfg("374m_d1024x24/lt1024x24_ours_p2k10l2c2_4klen.json")
ids = torch.randint(0, ours["vocab_size"], (1, 32), generator=torch.Generator().manual_seed(7))

for impl, build in (("flame", build_flame), ("port", build_port)):
    m_o = build(ours, seed=0)
    q = m_o.model.loop_dattn.query
    rep.check(f"C7 191M [{impl}] query shape {tuple(q.shape)} == (KR-1, d) = (15, 768), 11,520 entries, all zero at init",
              tuple(q.shape) == (15, 768) and q.numel() == 11520 and q.abs().max().item() == 0.0)
    del m_o
    # C6 at 191M: Loopback with skip0_beta = 0 vs Plain Loop
    m_o0 = build(ours, seed=0, skip0_beta=0.0)
    m_p = build(plain, seed=0)
    so, sp = m_o0.state_dict(), m_p.state_dict()
    rep.check(f"C6 191M [{impl}] Loopback(beta=0) vs Plain Loop from seed 0: shared params bitwise", all(torch.equal(so[k], sp[k]) for k in sp))
    compare("C6 191M Loopback(beta=0, q=0) vs Plain Loop", impl, m_o0, m_p, ids)
    del m_o0, m_p
    m3 = build(m374, seed=0)
    q = m3.model.loop_dattn.query
    rep.check(f"C7 374M [{impl}] query shape {tuple(q.shape)} == (KR-1, d) = (19, 1024), {q.numel()} entries, zero at init",
              tuple(q.shape) == (19, 1024) and q.abs().max().item() == 0.0)
    del m3

# ----------------------------------------------------------------------------------------------------------------
# C7: the trainer's meta-device init path (flame/train.py:226-295): meta -> _is_hf_initialized=False -> to_empty -> post_init
# ----------------------------------------------------------------------------------------------------------------
sys.path.insert(0, TRAINER)
from flame.models.parallelize_fla import validate_fsdp_meta_init  # noqa: E402
from common import flame_cfg  # noqa: E402

cfg = flame_cfg(ours)
with torch.device("meta"):
    mm = M.LoopedTransformerForCausalLM(cfg)
    mm.apply(lambda m: setattr(m, "_is_hf_initialized", False))
try:
    validate_fsdp_meta_init(mm)
    rep.check("C7 validate_fsdp_meta_init (every module with direct params has reset_parameters) passes on the 191M ours model", True)
except RuntimeError as e:
    rep.check("C7 validate_fsdp_meta_init passes on the 191M ours model", False, str(e)[:300])
mm.to_empty(device="cpu")
with torch.no_grad():
    for p in mm.parameters():
        p.fill_(float("nan"))           # prove that post_init writes EVERY parameter
    torch.manual_seed(0)
    mm.post_init()
nan_left = [n for n, p in mm.named_parameters() if not torch.isfinite(p).all()]
q = mm.model.loop_dattn.query
rep.check("C7 meta -> to_empty -> post_init: query is all zeros", q.abs().max().item() == 0.0 and torch.isfinite(q).all().item())
rep.check("C7 meta -> to_empty -> post_init: no parameter left uninitialised (NaN-filled before post_init)", not nan_left, f"left: {nan_left[:10]}")
emb_std = mm.model.embeddings.weight.std().item()
norms_one = all(torch.equal(p, torch.ones_like(p)) for n, p in mm.named_parameters() if n.endswith("norm.weight"))
rep.check("C7 meta path: embedding std ~ 0.02, every RMSNorm gain == 1", abs(emb_std - 0.02) < 0.002 and norms_one, f"emb std {emb_std:.4f}")

raise SystemExit(rep.close())
