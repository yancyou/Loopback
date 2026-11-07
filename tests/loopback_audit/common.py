"""Shared helpers of the Loopback mechanism checks: both implementations on CPU, the training model's GPU kernels
swapped for torch references (as tests/test_looped_transformer_port.py does)."""
import hashlib
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

AUDIT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(AUDIT))
TRAINER = os.environ.get("TRAINER_DIR", os.path.join(ROOT, "trainer"))   # the training model (flame)
PORT = ROOT                                                           # looped_transformer.py
CFG_DIR = os.path.join(ROOT, "configs")

sys.path.insert(0, TRAINER)
sys.path.insert(0, PORT)
torch.set_num_threads(int(os.environ.get("AUDIT_THREADS", "8")))

import flame.custom_models.looped_transformer.modeling_looped_transformer as M  # noqa: E402
import flame.custom_models.looped_transformer.rotary as MR  # noqa: E402
import fla.modules.mlp as fla_mlp  # noqa: E402
from flame.custom_models.looped_transformer.configuration_looped_transformer import LoopedTransformerConfig as FlameConfig  # noqa: E402
import looped_transformer as P  # noqa: E402

assert os.path.realpath(M.__file__).startswith(os.path.realpath(TRAINER)), M.__file__
assert os.path.realpath(P.__file__).startswith(os.path.realpath(PORT)), P.__file__


def ref_attn(q, k, v, causal=True, **kw):
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    T = q.size(-2)
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    return torch.matmul(torch.softmax(s, -1), v).transpose(1, 2).contiguous()


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def ref_rotary(x, cos, sin, interleaved=False, inplace=False, seqlen_offsets=0, cu_seqlens=None):
    T = x.shape[1]
    c = torch.cat((cos[:T].float(),) * 2, -1)[None, :, None, :]
    s = torch.cat((sin[:T].float(),) * 2, -1)[None, :, None, :]
    xf = x.float()
    return (xf * c + rotate_half(xf) * s).to(x.dtype)


def patch_kernels():
    """flash attention, Triton RoPE, fla SwiGLU -> torch; the port's sdpa -> the same ref_attn."""
    M.flash_attn_func = ref_attn
    MR.rotary_embedding = ref_rotary
    fla_mlp.swiglu = lambda g, u: F.silu(g) * u
    P.sdpa_causal = lambda q, k, v: ref_attn(q, k, v, causal=True)


def flame_cfg(raw: dict, **over) -> FlameConfig:
    cfg = dict(raw, fuse_norm=False, fuse_swiglu=False, fuse_cross_entropy=False, fuse_linear_cross_entropy=False)
    cfg.update(over)
    return FlameConfig(**cfg)


def build_flame(raw: dict, seed: int | None = 0, dtype=torch.float32, **over):
    if seed is not None:
        torch.manual_seed(seed)
    model = M.LoopedTransformerForCausalLM(flame_cfg(raw, **over))
    return model.to(dtype)


def build_port(raw: dict, seed: int | None = 0, dtype=torch.float32, **over):
    if seed is not None:
        torch.manual_seed(seed)
    cfg = P.LoopedTransformerConfig.from_dict(dict(raw, **over), source="audit")
    return P.LoopedTransformerLM(cfg).to(dtype)


def load_cfg(name: str) -> dict:
    path = name if os.path.isabs(name) else os.path.join(CFG_DIR, name)
    return json.load(open(path))


def tiny_raw(**over) -> dict:
    """A small flame-style config dict (the JSON keys the paper configs use)."""
    raw = dict(model_type="looped_transformer", hidden_size=64, num_heads=4, num_hidden_layers=6,
               num_prelude_layers=1, num_coda_layers=1, num_middle_param_layers=2, vocab_size=97,
               rope_theta=1e4, max_position_embeddings=256, norm_eps=1e-6, initializer_range=0.02,
               loop_residual_mode="delta_attnres", skip0_beta=0.75, tie_word_embeddings=False,
               hidden_act="swish", attention_bias=False, elementwise_attn_output_gate=False,
               headwise_attn_output_gate=False, fuse_cross_entropy=True, fuse_norm=True,
               fuse_linear_cross_entropy=False, use_cache=True)
    raw.update(over)
    return raw


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def randomize_(model, gen, std=0.02, cell_std=0.1):
    """The port test's randomisation: every weight away from init, loop-cell small params N(0, cell_std)."""
    hidden = model.config.hidden_size
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.copy_(1.0 + torch.randn(p.shape, generator=gen) * 0.1)
            elif ".loop_" in n and p.numel() <= 4 * hidden:
                p.copy_(torch.randn(p.shape, generator=gen) * cell_std)
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * std)


def paper_prior(K: int, R: int, lam: float, s: int) -> torch.Tensor:
    """eq:boundary-schedule: c0_{s,i} = lam^(floor((s+1)/K) - floor(i/K)), i = 0..s."""
    return torch.tensor([lam ** ((s + 1) // K - i // K) for i in range(s + 1)], dtype=torch.float64)


class Report:
    """PASS / FAIL lines on stdout; close() returns the number of failures."""

    def __init__(self, name):
        self.name = name
        self.fails = 0

    def check(self, name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        if not ok:
            self.fails += 1
        print(f"[{tag}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)

    def note(self, text):
        print(text, flush=True)

    def close(self):
        print(f"[{'ALL PASS' if self.fails == 0 else str(self.fails) + ' FAIL'}] {self.name}", flush=True)
        return self.fails
