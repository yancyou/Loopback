"""PORT EQUIVALENCE: `looped_transformer.py` (this repository, plain PyTorch) against the flame training model it
reproduces (`trainer/flame/custom_models/looped_transformer`), on the SAME weights and inputs.

Three levels, on every config in CONFIGS (the paper's configurations: Standard, Plain Loop, Loopback and readers of
both interfaces, at 191M / 374M / 778M / 1.4B / 3B and in the depth sweep) plus one synthetic variant of the 191M
[2->4x4->2] topology per released residual rule (MODE_VARIANTS):

  1. fp32, kernels swapped for their torch references on BOTH sides (attention, RoPE, SwiGLU;
     fla's fused norm / cross-entropy switched off by config): logits, loss and the gradient of
     every parameter must be BITWISE equal. Weights are randomised first (loop-cell parameters
     with a larger std so every gate / query / reader is exercised away from its init), so the
     data-dependent paths are compared, not just the init.
  2. fp32, the port on its own kernels (F.scaled_dot_product_attention, its own RoPE) against
     the reference-kernel flame model: agreement to fp32 round-off (reported, asserted < 1e-3).
  3. bf16, the flame model on its real training kernels (fla fused RMSNorm / SwiGLU / CE,
     flash-attention, Triton RoPE) against the port in bf16: agreement to bf16 round-off
     (reported, asserted loosely).

Plus: state-dict keys identical (strict load), parameter counts identical, `from_json` accepts every released
config and rejects unknown keys, and the init contracts: Loopback at beta 0 and every reader at init are bitwise
Plain Loop, and the Loopback query is zero at init.

Run from the repository root inside the flame environment (the trainer is the reference; set
TRAINER_DIR to point at another flame tree):

    python tests/test_looped_transformer_port.py [--quick] [--only SUBSTR] [--seq N]
    python tests/test_looped_transformer_port.py --configs "configs/374m_d1024x24/*.json"

Level 3 needs a GPU; on CPU the script runs levels 1-2 only (--quick is implied).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
TRAINER = os.environ.get("TRAINER_DIR", os.path.join(ROOT, "trainer"))
sys.path.insert(0, TRAINER)

import looped_transformer as P  # noqa: E402

CONFIGS = [  # relative to the repository root
    "configs/191m_d768x20/lt124m_dense20_4klen.json",
    "configs/191m_d768x20/lt124m_plain_p2k4l4c2_4klen.json",
    "configs/191m_d768x20/lt124m_ours_p2k4l4c2_4klen.json",
    "configs/191m_d768x20/lt124m_la_gdn_p2k4l4c2_4klen.json",
    "configs/191m_d768x20/lt124m_bd_kda_p2k4l4c2_4klen.json",
    "configs/374m_d1024x24/lt1024x24_dense24_4klen.json",
    "configs/374m_d1024x24/lt1024x24_ours_p2k10l2c2_4klen.json",
    "configs/374m_d1024x24/lt1024x24_bd_mesa_p2k10l2c2_4klen.json",
    "configs/depth_sweep/d896x32/lt896x32_ours_p2k14l2c2_4klen.json",
    "configs/778m_d1536x24/lt760m_dense24_4klen.json",
    "configs/778m_d1536x24/lt760m_ours_p2k8l2c6_4klen.json",
    "configs/depth_sweep/d1088x48/lt1088x48_ours_p3k21l2c3_4klen.json",
    "configs/1.4b_d2048x25/lt2048_ours_p3k8l2c6_4klen.json",
    "configs/3b_d3072x24/lt3072x24_ours_p6k6l2c6_4klen.json",
]

# one variant per released residual rule of the 191M [2->4x4->2] topology
MODE_BASE = "configs/191m_d768x20/lt124m_ours_p2k4l4c2_4klen.json"
MODE_VARIANTS = {
    "plain":         {"loop_residual_mode": None},
    "delta_attnres": {"loop_residual_mode": "delta_attnres", "skip0_beta": 0.75},
}
# the loop-axis linear-attention family: eleven readers x {absolute states (la_, 8 head-subspaces), per-iteration
# updates (bd_, single head)} = the paper's family table
LINATTN_KINDS = ("linattn", "retnet", "mamba2", "gla", "hgrn2", "deltanet", "gdn", "kda", "rwkv7", "lact", "mesa")
for _k in LINATTN_KINDS:
    MODE_VARIANTS[f"la_{_k}"] = {"loop_residual_mode": _k, "linattn_source": "state", "attnres_num_heads": 8}
    MODE_VARIANTS[f"bd_{_k}"] = {"loop_residual_mode": _k, "linattn_source": "delta"}

# modes whose bf16 forward is chaotic by construction (none among the released modes): level 3 reports instead of
# asserting on their logits
BF16_SENSITIVE = {}


# ---------------------------------------------------------------------------------------
# torch references shared by both models in the bitwise run
# ---------------------------------------------------------------------------------------

def ref_attn(q, k, v, causal=True, **kwargs):
    """(B, T, H, D) causal attention in the input dtype; the flame file's own fallback."""
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if k.shape[1] != q.shape[1]:
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        T = q.size(-2)
        mask = torch.ones(T, T, device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v).transpose(1, 2).contiguous()


def ref_rotary(x, cos, sin, interleaved=False, inplace=False, seqlen_offsets=0, cu_seqlens=None):
    """The Triton rotary kernel's arithmetic in torch: fp32 rotation of GPT-NeoX halves, rounded to
    the activation dtype. cos/sin: (TR, D/2) in the activation dtype, as fla caches them."""
    assert not interleaved and cu_seqlens is None and seqlen_offsets == 0
    T = x.shape[1]
    c = cos[:T].float()
    s = sin[:T].float()
    c = torch.cat((c, c), dim=-1)[None, :, None, :]
    s = torch.cat((s, s), dim=-1)[None, :, None, :]
    xf = x.float()
    return (xf * c + P.rotate_half(xf) * s).to(x.dtype)


def torch_swiglu(gate, up):
    return F.silu(gate) * up


# ---------------------------------------------------------------------------------------

def load_flame():
    import flame.custom_models.looped_transformer.modeling_looped_transformer as M
    import flame.custom_models.looped_transformer.rotary as MR
    import fla.modules.mlp as fla_mlp
    from flame.custom_models.looped_transformer.configuration_looped_transformer import LoopedTransformerConfig
    return M, MR, fla_mlp, LoopedTransformerConfig


def build_flame(M, FlameConfig, raw: dict, reference_kernels: bool, dtype, device):
    cfg = dict(raw)
    if reference_kernels:
        cfg.update(fuse_norm=False, fuse_swiglu=False, fuse_cross_entropy=False, fuse_linear_cross_entropy=False)
    elif cfg.get("fuse_linear_cross_entropy"):
        # the linear-fused loss never materialises logits (nothing to compare); the fused CE on
        # materialised logits is the same training-kernel family
        cfg.update(fuse_linear_cross_entropy=False, fuse_cross_entropy=True)
    model = M.LoopedTransformerForCausalLM(FlameConfig(**cfg))
    return model.to(device=device, dtype=dtype)


def randomize_(model: torch.nn.Module, gen: torch.Generator, std: float = 0.02, cell_std: float = 0.1):
    """Overwrite every parameter with N(0, std); norm gains around 1; the loop cells' small
    parameters (queries, gate biases / scales, mixing deltas, decays -- anything under `model.loop_*`
    with at most 4 * hidden elements) N(0, cell_std) so every gate, router and mixer acts away from
    its calibrated init (0.1: large enough that every path is live in the bitwise fp32 comparison,
    small enough that the stream mixing does not turn bf16 rounding into O(10%) logit chaos at
    level 3). Shared parameters are iterationed once (named_parameters dedups)."""
    hidden = model.config.hidden_size
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1.0 + torch.randn(p.shape, generator=gen) * 0.1)
            elif ".loop_" in name and p.numel() <= 4 * hidden:
                p.copy_(torch.randn(p.shape, generator=gen) * cell_std)
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * std)


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def unique_numel(model):
    seen, n = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p)); n += p.numel()
    return n


def run_pair(flame_model, port_model, ids, labels):
    out_f = flame_model(input_ids=ids, labels=labels)
    out_p = port_model(ids, labels=labels)
    return out_f.logits, out_f.loss, out_p["logits"], out_p["loss"]


def grads_by_name(model):
    return {n: p.grad for n, p in model.named_parameters()}


def check_config(name: str, raw: dict, M, MR, fla_mlp, FlameConfig, device, quick: bool, seq: int):
    big = raw.get("hidden_size", 768) >= 1024
    T = min(seq, 32) if big else seq
    B = 2
    gen = torch.Generator().manual_seed(1234)
    ids = torch.randint(0, raw["vocab_size"], (B, T), generator=gen).to(device)
    labels = ids.clone()
    report = {}

    # ---- 1. fp32, reference kernels on both sides: bitwise ----
    saved = (M.flash_attn_func, MR.rotary_embedding, fla_mlp.swiglu, P.sdpa_causal)
    M.flash_attn_func = ref_attn
    MR.rotary_embedding = ref_rotary
    fla_mlp.swiglu = torch_swiglu
    P.sdpa_causal = lambda q, k, v: ref_attn(q, k, v, causal=True)
    try:
        flame_fp32 = build_flame(M, FlameConfig, raw, reference_kernels=True, dtype=torch.float32, device=device)
        randomize_(flame_fp32, gen)
        port = P.LoopedTransformerLM(P.LoopedTransformerConfig.from_dict(raw, source=name)).to(device=device, dtype=torch.float32)
        missing, unexpected = port.load_state_dict(flame_fp32.state_dict(), strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        assert unique_numel(port) == unique_numel(flame_fp32), (unique_numel(port), unique_numel(flame_fp32))
        report["params"] = unique_numel(port)
        report["non_embed_params"] = port.num_params(exclude_embeddings=True)

        flame_fp32.train(); port.train()
        lf, lossf, lp, lossp = run_pair(flame_fp32, port, ids, labels)
        assert torch.isfinite(lf).all() and torch.isfinite(lossf), f"{name}: non-finite flame output (randomisation too wide)"
        report["bitwise_logits"] = torch.equal(lf, lp)
        report["bitwise_loss"] = torch.equal(lossf, lossp)
        report["fp32_ref_maxdiff"] = maxdiff(lf, lp)
        lossf.backward(); lossp.backward()
        gf, gp = grads_by_name(flame_fp32), grads_by_name(port)
        assert set(gf) == set(gp), (set(gf) ^ set(gp))
        worst_name, worst = None, 0.0
        bitwise_grads = True
        n_grads = 0
        for n in gf:
            assert (gf[n] is None) == (gp[n] is None), n
            if gf[n] is None:
                continue
            n_grads += 1
            assert torch.isfinite(gf[n]).all(), f"{name}: non-finite flame gradient at {n}"
            if not torch.equal(gf[n], gp[n]):
                bitwise_grads = False
            d = maxdiff(gf[n], gp[n])
            if d > worst:
                worst_name, worst = n, d
        report["bitwise_grads"] = bitwise_grads
        report["grad_maxdiff"] = (worst_name, worst)
        report["n_grads"] = n_grads
        report["loss"] = lossp.item()
        assert report["bitwise_logits"], f"{name}: fp32 reference-kernel logits differ by {report['fp32_ref_maxdiff']}"
        assert report["bitwise_loss"], f"{name}: fp32 reference-kernel loss differs"
        assert bitwise_grads, f"{name}: gradient of {worst_name} differs by {worst}"
        flame_fp32.zero_grad(set_to_none=True); port.zero_grad(set_to_none=True)

        # ---- 2. fp32, the port on its own kernels vs the reference-kernel flame model ----
        P.sdpa_causal = saved[3]
        with torch.no_grad():
            lp_native = port(ids, labels=labels)["logits"]
        report["fp32_native_maxdiff"] = maxdiff(lf, lp_native)
        assert report["fp32_native_maxdiff"] < 1e-3, f"{name}: native fp32 port differs by {report['fp32_native_maxdiff']}"
        state = {k: v.detach().clone() for k, v in flame_fp32.state_dict().items()}
        del flame_fp32
    finally:
        M.flash_attn_func, MR.rotary_embedding, fla_mlp.swiglu, P.sdpa_causal = saved

    # ---- 3. bf16, flame on its real kernels vs the port in bf16 ----
    if not quick:
        flame_bf16 = build_flame(M, FlameConfig, raw, reference_kernels=False, dtype=torch.bfloat16, device=device)
        flame_bf16.load_state_dict(state, strict=True)
        port_bf16 = port.to(torch.bfloat16)
        flame_bf16.eval(); port_bf16.eval()
        with torch.no_grad():
            lf16, lossf16, lp16, lossp16 = run_pair(flame_bf16, port_bf16, ids, labels)
        report["bf16_logits_maxdiff"] = maxdiff(lf16, lp16)
        report["bf16_logits_meandiff"] = (lf16.float() - lp16.float()).abs().mean().item()
        report["bf16_loss"] = (lossf16.item(), lossp16.item())
        scale = lf16.float().abs().max().item()
        sensitive = raw.get("loop_residual_mode") in BF16_SENSITIVE
        if sensitive:
            report["bf16_note"] = f"reported only: {BF16_SENSITIVE[raw['loop_residual_mode']]}"
            assert abs(lossf16.item() - lossp16.item()) < 5e-2, f"{name}: bf16 losses {lossf16.item()} vs {lossp16.item()}"
        else:
            # bf16 round-off through 20 blocks and (for the stream-mixing cells) 2 loop junctions:
            # a few percent of the logit scale; the loss agrees to 2e-2 in every configuration
            assert report["bf16_logits_maxdiff"] < 0.1 * max(scale, 1.0), f"{name}: bf16 real-kernel logits differ by {report['bf16_logits_maxdiff']} (scale {scale:.2f})"
            assert abs(lossf16.item() - lossp16.item()) < 2e-2, f"{name}: bf16 losses {lossf16.item()} vs {lossp16.item()}"
        del flame_bf16, port_bf16
    del port
    torch.cuda.empty_cache()
    return report


# ---------------------------------------------------------------------------------------
# init contracts, on a tiny model, port only (the flame side of each is its docstring's claim,
# and level 1 above shows the two sides compute the same function)
# ---------------------------------------------------------------------------------------

def _tiny(**over):
    base = dict(hidden_size=64, num_hidden_layers=6, num_heads=4, num_prelude_layers=1, num_coda_layers=1,
                num_middle_param_layers=2, vocab_size=97, loop_residual_mode=None, skip0_beta=0.75, rope_theta=1e4)
    base.update(over)
    return P.LoopedTransformerConfig(**base)


def _same_trunk(model_a, model_b):
    """Copy every parameter of model_a that model_b also has (the shared trunk), leaving model_b's
    own loop-cell parameters at their init."""
    sd = model_a.state_dict()
    own = model_b.state_dict()
    sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing, unexpected = model_b.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return missing


def check_init_and_json():
    torch.manual_seed(0)
    plain = P.LoopedTransformerLM(_tiny())
    ids = torch.randint(0, 97, (2, 11))
    with torch.no_grad():
        ref = plain(ids)["logits"]

    dattn = P.LoopedTransformerLM(_tiny(loop_residual_mode="delta_attnres"))
    assert dattn.model.loop_dattn.query.shape == (2 * 2 - 1, 64)

    # the released cells' init contracts against Plain Loop on the same trunk
    contracts = {  # key -> (config overrides, max |logit diff| allowed, what the cell promises)
        "delta_attnres":  ({"skip0_beta": 0.0}, 0.0, "beta 0, zero queries: bitwise Plain Loop"),
    }
    for k in LINATTN_KINDS:   # alpha = 0 => combine() returns the current state, for every kind and both sources
        contracts[f"la_{k}"] = ({"linattn_source": "state"}, 0.0, "alpha = 0: bitwise the current state")
        contracts[f"bd_{k}"] = ({"linattn_source": "delta"}, 0.0, "alpha = 0: bitwise the current state")
    rows = []
    for key, (over, tol, promise) in contracts.items():
        mode = key[3:] if key.startswith(("la_", "bd_")) else key
        m = P.LoopedTransformerLM(_tiny(loop_residual_mode=mode, **over))
        _same_trunk(plain, m)
        with torch.no_grad():
            d = maxdiff(m(ids)["logits"], ref)
        rows.append(f"    {key:15s} init |logits - plain| max = {d:.2e}  (allowed {tol:.1e}: {promise})")
        assert d <= tol, f"{key}: init differs from the plain loop by {d} > {tol}"
    print("init contracts:\n" + "\n".join(rows))

    # the Loopback query is zero at init (the cell's own reset_parameters wins over the Linear draw)
    assert dattn.model.loop_dattn.query.abs().sum() == 0

    # tied head shares the embedding tensor
    t = P.LoopedTransformerLM(_tiny(tie_word_embeddings=True))
    assert t.lm_head.weight is t.model.embeddings.weight
    # from_json is strict about unknown keys
    tmp = os.path.join(HERE, "_tmp_port_cfg.json")
    try:
        json.dump({"model_type": "looped_transformer", "hidden_size": 64, "num_heads": 4, "num_hidden_layers": 4,
                   "num_prelude_layers": 1, "num_coda_layers": 1, "num_middle_param_layers": 2, "vocab_size": 97,
                   "rope_theta": 1e4, "chunk_size": 512}, open(tmp, "w"))
        try:
            P.LoopedTransformerConfig.from_json(tmp)
            raise AssertionError("unknown key chunk_size was accepted")
        except ValueError as e:
            assert "chunk_size" in str(e)
    finally:
        os.remove(tmp)
    print("init/json checks: ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the bf16 real-kernel level")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--only", type=str, default=None, help="substring filter on config names")
    ap.add_argument("--configs", type=str, nargs="+", default=None,
                    help="one or more globs of flame config JSONs to check INSTEAD of the built-in list and mode "
                         "variants (relative to this repository, e.g. 'configs/looped_760m/lt760m_*_4klen.json'); "
                         "every glob must match")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU; the bf16 real-kernel level needs one", file=sys.stderr)
        args.quick = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    check_init_and_json()
    M, MR, fla_mlp, FlameConfig = load_flame()
    specs = []
    for pat in (args.configs or []):
        hits = sorted(glob.glob(pat if os.path.isabs(pat) else os.path.join(ROOT, pat)))
        if not hits:
            raise SystemExit(f"--configs {pat!r} matched no file")
        for path in hits:
            specs.append((os.path.basename(path).replace(".json", ""), json.load(open(path))))
    for pat in ([] if args.configs else CONFIGS):
        hits = sorted(glob.glob(os.path.join(ROOT, pat)))
        if not hits:
            raise SystemExit(f"no config matches {pat}")
        specs.append((os.path.basename(hits[0]).replace(".json", ""), json.load(open(hits[0]))))
    base_hits = [] if args.configs else sorted(glob.glob(os.path.join(ROOT, MODE_BASE)))
    if args.configs:
        pass
    elif base_hits:
        base = json.load(open(base_hits[0]))
        base = {k: v for k, v in base.items() if k not in ("loop_residual_mode", "skip0_beta")}
        for key, over in MODE_VARIANTS.items():
            specs.append((f"p2k4l4c2:{key}", {**base, **over}))
    else:
        print(f"(no config matches {MODE_BASE}; mode variants skipped)")
    if args.only:
        specs = [s for s in specs if args.only in s[0]]
    print(f"device {device}; {len(specs)} configs; seq {args.seq}")
    for name, raw in specs:
        t0 = time.time()
        cfg = P.LoopedTransformerConfig.from_dict(raw, source=name)
        rep = check_config(name, raw, M, MR, fla_mlp, FlameConfig, device, args.quick, args.seq)
        gname, gd = rep["grad_maxdiff"]
        line = (f"{name:44s} {cfg.topology():16s} mode={cfg.loop_residual_mode} params={rep['params']/1e6:.2f}M "
                f"(non-embed {rep['non_embed_params']/1e6:.2f}M) | fp32 ref: logits bitwise={rep['bitwise_logits']} "
                f"grads bitwise={rep['bitwise_grads']} ({rep['n_grads']} tensors, worst {gd:.1e} at {gname}) | fp32 native maxdiff={rep['fp32_native_maxdiff']:.2e}")
        if not args.quick:
            line += (f" | bf16 real kernels: max={rep['bf16_logits_maxdiff']:.3f} mean={rep['bf16_logits_meandiff']:.4f} "
                     f"loss {rep['bf16_loss'][0]:.4f}/{rep['bf16_loss'][1]:.4f}")
        print(line + f"  [{time.time()-t0:.0f}s]", flush=True)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
