"""The Loopback Transformer in plain PyTorch: the model that trainer/ (flame) trains, without fla, flash-attn,
transformers or torchtitan, so the trained checkpoints can be read and evaluated anywhere. It reproduces
trainer/flame/custom_models/looped_transformer/ bitwise in fp32 (tests/test_looped_transformer_port.py).

Architecture (`LoopedTransformerConfig`):
  * token embedding N(0, `initializer_range`); untied lm_head unless `tie_word_embeddings`;
  * pre-RMSNorm blocks (learnable gain, `norm_eps`) -> causal multi-head attention with RoPE (`rope_theta`) ->
    pre-RMSNorm -> SwiGLU MLP (intermediate = the multiple of 256 at or above 2/3 * hidden * `hidden_ratio`);
  * topology [P -> K x R -> C]: `num_prelude_layers` unique blocks, then `num_hidden_layers` - P - C applications
    that cycle over `num_middle_param_layers` shared blocks (R iterations of K blocks), then `num_coda_layers`
    unique blocks; `num_middle_param_layers=None` is the standard Transformer (every block unique);
  * the residual rule between block applications (`loop_residual_mode`):
      None              addition: Plain Loop (and, without sharing, the standard Transformer);
      "delta_attnres"   Loopback (`LoopDeltaAttnRes`): the block updates are stored, geometric decay sets their
                        total weight (lambda = 1 - `skip0_beta` at every iteration end, the loop input keeps
                        coefficient 1) and one learned query per read reallocates it; zero queries at init, so the
                        model starts as its decay-only reference; the fused Triton read of fused_dattn.py when
                        LOOP_DATTN_FUSED=1;
      a reader kind     one of LINATTN_KINDS: a linear-attention reader on the loop axis, read at iteration ends,
                        over the stored states (`linattn_source` "state", loop_linattn_state.py, `attnres_num_heads`
                        heads) or the whole-iteration updates ("delta", loop_linattn_delta.py, one head).
  * init: every Linear and the embedding N(0, `initializer_range`), RMSNorm gains 1, the Loopback query zero and
    every reader at its Plain-Loop values (a cell's own `reset_parameters` runs after its Linear children).

State-dict keys are the trainer model's (model.embeddings.weight, model.layers.{i}.attn.q_proj.weight,
model.layers.{i}.mlp.gate_proj.weight, model.norm.weight, model.loop_dattn.query, model.loop_linattn.*,
lm_head.weight), so a trainer checkpoint loads with `load_state_dict(..., strict=True)`; a shared block appears
once per layer position.

Numerics: fla's fused RMSNorm / SwiGLU / cross-entropy, flash-attention and the Triton RoPE are replaced by
`torch.nn.RMSNorm`, `F.silu(g) * u`, `F.cross_entropy`, `F.scaled_dot_product_attention` and an fp32 RoPE that rounds
to the activation dtype as the Triton kernel does. With the kernels swapped for their torch references the two
models agree bitwise in fp32; with the real bf16 kernels they agree to bf16 round-off.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from loop_linattn_state import LINATTN_KINDS, LoopLinAttn as LoopLinAttnState   # noqa: E402  (this repo's own family cells)
from loop_linattn_delta import LINATTN_KINDS as _BD_KINDS, LoopLinAttn as LoopLinAttnDelta  # noqa: E402
assert tuple(LINATTN_KINDS) == tuple(_BD_KINDS), (LINATTN_KINDS, _BD_KINDS)

LOOP_RESIDUAL_MODES = (None, "delta_attnres", *LINATTN_KINDS)

# trainer config keys that describe kernels or HF plumbing; the second and third groups are accepted only at the
# value the released configs use.
_IGNORED_JSON_KEYS = {
    "model_type", "fuse_norm", "fuse_swiglu", "fuse_cross_entropy", "fuse_linear_cross_entropy",
    "use_l2warp", "use_cache", "bos_token_id", "eos_token_id", "pad_token_id", "attention_bias",
    "transformers_version", "architectures", "torch_dtype", "dtype",
}
_MUST_BE_FALSE_JSON_KEYS = ("headwise_attn_output_gate", "elementwise_attn_output_gate", "use_short_conv",
                            "zero_init_residual_proj", "qk_norm", "qkv_bias")
_MUST_BE_NONE_JSON_KEYS = ("window_size",)
_MUST_EQUAL_JSON_KEYS = {"linattn_gate_source": "param", "init_scheme": "flame"}   # options the release has no other value of


@dataclass
class LoopedTransformerConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 20
    num_heads: int = 12
    rope_theta: float = 1e6
    max_position_embeddings: int = 4096
    hidden_ratio: int = 4
    intermediate_size: int | None = None
    hidden_act: str = "swish"
    initializer_range: float = 0.02
    elementwise_affine: bool = True
    norm_eps: float = 1e-6
    vocab_size: int = 32000
    tie_word_embeddings: bool = False
    # topology
    num_prelude_layers: int = 2
    num_coda_layers: int = 4
    num_middle_param_layers: int | None = 7
    # the residual rule between block applications (the module docstring); skip0_beta = 1 - lambda of Loopback's decay
    loop_residual_mode: str | None = "delta_attnres"
    skip0_beta: float = 0.75
    # the readers (field names are the trainer's): heads of the state interface; "state" | "delta" interface
    attnres_num_heads: int = 8
    linattn_source: str = "state"

    def __post_init__(self):
        if self.hidden_act != "swish":
            raise ValueError(f"only hidden_act='swish' (SwiGLU) is implemented, got {self.hidden_act!r}")
        if not self.elementwise_affine:
            raise ValueError("the trained model has learnable RMSNorm gains (elementwise_affine=True)")
        if self.loop_residual_mode not in LOOP_RESIDUAL_MODES:
            raise ValueError(f"loop_residual_mode must be one of {LOOP_RESIDUAL_MODES}, got {self.loop_residual_mode!r}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}")
        if self.rope_theta is None or self.rope_theta <= 0:
            raise ValueError("rope_theta must be a positive float")
        if self.num_prelude_layers < 0 or self.num_coda_layers < 0:
            raise ValueError("num_prelude_layers and num_coda_layers must be >= 0")
        middle = self.num_hidden_layers - self.num_prelude_layers - self.num_coda_layers
        if middle < 1:
            raise ValueError(f"prelude {self.num_prelude_layers} + coda {self.num_coda_layers} leaves no middle "
                             f"layer out of {self.num_hidden_layers}")
        if self.num_middle_param_layers is not None and not (1 <= self.num_middle_param_layers <= middle):
            raise ValueError(f"num_middle_param_layers must be in [1, {middle}], got {self.num_middle_param_layers}")
        if not 0.0 <= float(self.skip0_beta) < 1.0:
            raise ValueError(f"skip0_beta must be in [0, 1), got {self.skip0_beta}")
        if self.linattn_source not in ("state", "delta"):
            raise ValueError(f"linattn_source must be 'state' or 'delta', got {self.linattn_source!r}")
        if self.loop_residual_mode in LINATTN_KINDS and self.linattn_source == "state" and self.hidden_size % self.attnres_num_heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} not divisible by attnres_num_heads {self.attnres_num_heads}")
        self.skip0_beta = float(self.skip0_beta)

    # ---- topology helpers ----
    @property
    def num_middle_layers(self) -> int:
        return self.num_hidden_layers - self.num_prelude_layers - self.num_coda_layers

    @property
    def loop_block_size(self) -> int:
        return self.num_middle_param_layers or self.num_middle_layers

    @property
    def num_loops(self) -> int:
        return self.num_middle_layers // self.loop_block_size

    @property
    def mlp_intermediate_size(self) -> int:
        if self.intermediate_size is not None:
            return self.intermediate_size
        # fla GatedMLP: the multiple of 256 at or above 2/3 * hidden * ratio
        size = int(self.hidden_size * self.hidden_ratio * 2 / 3)
        return 256 * ((size + 256 - 1) // 256)

    def topology(self) -> str:
        if self.num_middle_param_layers is None:
            return f"dense{self.num_hidden_layers}"
        return f"[{self.num_prelude_layers}->{self.loop_block_size}x{self.num_loops}->{self.num_coda_layers}]"

    # ---- flame JSON ----
    @classmethod
    def from_json(cls, path: str) -> "LoopedTransformerConfig":
        """Build from a trainer config JSON (configs/). Every key is a field of this config, an accepted kernel /
        HF key, or an error: a key this implementation does not have is never ignored silently."""
        with open(path) as f:
            raw = json.load(f)
        return cls.from_dict(raw, source=path)

    @classmethod
    def from_dict(cls, raw: dict, source: str = "<dict>") -> "LoopedTransformerConfig":
        """`from_json` on an already-loaded config dict; `source` names it in errors."""
        if raw.get("model_type", "looped_transformer") != "looped_transformer":
            raise ValueError(f"{source}: model_type {raw['model_type']!r} is not looped_transformer")
        for k in _MUST_BE_FALSE_JSON_KEYS:
            if raw.get(k, False):
                raise ValueError(f"{source}: {k}=true is not implemented in this port")
        for k in _MUST_BE_NONE_JSON_KEYS:
            if raw.get(k) is not None:
                raise ValueError(f"{source}: {k}={raw[k]!r} is not implemented in this port")
        for k, v in _MUST_EQUAL_JSON_KEYS.items():
            if raw.get(k, v) != v:
                raise ValueError(f"{source}: {k}={raw[k]!r} is not implemented in this port (only {v!r})")
        names = {f.name for f in fields(cls)}
        kwargs, unknown = {}, []
        for k, v in raw.items():
            if k in names:
                kwargs[k] = v
            elif k in _IGNORED_JSON_KEYS or k in _MUST_BE_FALSE_JSON_KEYS or k in _MUST_BE_NONE_JSON_KEYS or k in _MUST_EQUAL_JSON_KEYS:
                continue
            else:
                unknown.append(k)
        if unknown:
            raise ValueError(f"{source}: keys this port does not implement: {sorted(unknown)}")
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE as the flame model applies it (fla `RotaryEmbedding`, non-interleaved): every pair
    of head dims rotates, cos/sin come from fp32 positions and inv_freq and are cached in the
    activation dtype; the rotation itself is computed in fp32 and rounded back, which is what
    the Triton kernel does."""

    def __init__(self, dim: int, base: float):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.register_buffer("inv_freq", torch.empty(-(dim // -2), dtype=torch.float32), persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.inv_freq.copy_(1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32,
                                                                       device=self.inv_freq.device) / self.dim)))

    def _update_cos_sin_cache(self, seqlen: int, device, dtype):
        if (seqlen > self._seq_len_cached or self._cos_cached is None
                or self._cos_cached.device != device or self._cos_cached.dtype != dtype):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            inv_freq = self.inv_freq.to(device)
            if inv_freq.dtype != torch.float32:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
            freqs = torch.outer(t, inv_freq)                       # (T, dim/2)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """q, k: (B, T, H, D). Returns the rotated q, k."""
        T = q.shape[1]
        self._update_cos_sin_cache(T, q.device, q.dtype)
        cos = self._cos_cached[:T].float()
        sin = self._sin_cached[:T].float()
        cos = torch.cat((cos, cos), dim=-1)[None, :, None, :]     # (1, T, 1, D)
        sin = torch.cat((sin, sin), dim=-1)[None, :, None, :]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * cos + rotate_half(xf) * sin).to(x.dtype)


def sdpa_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal attention on (B, T, H, D) tensors, returning (B, T, H, D)."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
    return out.transpose(1, 2)


class LoopedAttention(nn.Module):
    def __init__(self, config: LoopedTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=config.rope_theta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        q, k = self.rotary(q, k)
        attn = sdpa_causal(q, k, v)
        attn = attn.reshape(B, T, self.hidden_size).to(hidden_states.dtype)
        return self.o_proj(attn)


class SwiGLUMLP(nn.Module):
    """fla `GatedMLP` without the fused kernel: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: LoopedTransformerConfig):
        super().__init__()
        self.intermediate_size = config.mlp_intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LoopedTransformerBlock(nn.Module):
    def __init__(self, config: LoopedTransformerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = LoopedAttention(config, layer_idx)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states))
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


# --------------------------------------------------------------------------------------
# Loopback's read
# --------------------------------------------------------------------------------------

def _rms_no_affine(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm without a gain: the keys of the stored updates."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


_FUSED_DATTN = None


def _fused_dattn():
    """The fused Triton read (fused_dattn.py), imported on first use. Off unless
    LOOP_DATTN_FUSED=1: the paper's runs used the reference read below, which the kernel reproduces
    (fp32 to ~1e-7, bitwise at zero queries); with it the read is one kernel per destination instead
    of about 3n, and training runs at ~1.5-1.8x the reference read's throughput at 191M-778M."""
    global _FUSED_DATTN
    if _FUSED_DATTN is None:
        if os.environ.get("LOOP_DATTN_FUSED", "0") != "1":
            _FUSED_DATTN = False
        else:
            try:
                import fused_dattn
                _FUSED_DATTN = fused_dattn
            except Exception as e:  # noqa: BLE001 -- any import failure means: use the reference read
                warnings.warn(f"delta_attnres: fused read unavailable ({e!r}); using the reference read")
                _FUSED_DATTN = False
    return _FUSED_DATTN


class LoopDeltaAttnRes(nn.Module):
    """Loopback's read (`delta_attnres`; section 4 of the paper).

    After application s of shared block k (s = rK + k in iteration r) the block update d_s = f_k(h_s) - h_s joins
    the stored history and the next state is

        h_{s+1} = h_0 + sum_{i <= s} c_{s,i} d_i,
        c_{s,i} = B_s softmax_i(z_{s,i} + log c0_{s,i}),    z_{s,i} = q_s^T RMSNorm(d_i) / sqrt(d),
        c0_{s,i} = lambda^(floor((s+1)/K) - floor(i/K)),   B_s = sum_i c0_{s,i},   lambda = 1 - beta.

    The decay prior c0 gives the new update weight 1 and multiplies every stored weight by lambda at the end of an
    iteration (beta = `skip0_beta`); attention redistributes the total B_s with one learned query per destination
    (KR - 1 queries of width d; the first read has one source and needs none). Zero queries give c = c0, the
    decay-only loop h <- h_0 + lambda (F(h) - h_0); lambda = 1 gives Plain Loop. Queries start at zero.

    The state is carried as h_0 + sum_i cbar_i d_i with the installed coefficients cbar tracked per token, and a
    read adds sum_i (c_{s,i} - cbar_i) d_i, so every destination reads the bank once; the prior is per read, not
    per token. `keys`, if given, is a caller-owned list caching each update's RMSNorm scale (fused read) or
    normalized key (reference read)."""

    def __init__(self, hidden_size: int, num_loops: int, num_blocks: int, beta: float, eps: float = 1e-6):
        super().__init__()
        if num_loops < 1 or num_blocks < 1:
            raise ValueError(f"delta_attnres needs positive num_loops/num_blocks, got {num_loops}/{num_blocks}")
        self.dim = int(hidden_size)
        self.num_loops = int(num_loops)
        self.num_blocks = int(num_blocks)
        self.num_steps = self.num_loops * self.num_blocks
        self.beta = float(beta)
        self.eps = float(eps)
        self.query = nn.Parameter(torch.zeros(self.num_steps - 1, self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.query.zero_()

    def combine(self, proposal, h0, deltas, coeffs, prior_coeffs, loop_idx, block_idx, keys=None):
        if not 0 <= block_idx < self.num_blocks:
            raise ValueError(f"delta_attnres block_idx={block_idx}, need [0,{self.num_blocks})")
        step = loop_idx * self.num_blocks + block_idx
        if step >= self.num_steps:
            raise ValueError(f"delta_attnres step {step} exceeds constructed depth {self.num_steps}")
        if len(deltas) != step + 1:
            raise ValueError(f"delta_attnres step {step} received {len(deltas)} deltas, expected {step + 1}")

        at_boundary = block_idx == self.num_blocks - 1
        base = proposal + self.beta * (h0 - proposal) if at_boundary else proposal

        b, t, _ = proposal.shape
        one_tok = torch.ones(b, t, 1, device=proposal.device, dtype=torch.float32)
        one_prior = torch.ones(1, device=proposal.device, dtype=torch.float32)
        base_coeffs = one_tok if coeffs is None else torch.cat((coeffs, one_tok), dim=-1)
        next_prior = one_prior if prior_coeffs is None else torch.cat((prior_coeffs, one_prior), dim=-1)
        if at_boundary:
            r = 1.0 - self.beta
            base_coeffs = r * base_coeffs
            next_prior = r * next_prior
        fused = _fused_dattn()
        if fused:
            # fused read (fused_dattn.py): the cache holds each write's RMSNorm scale, (b, t, 1) fp32, not its key
            if keys is not None and len(keys) < len(deltas):
                keys.append(fused.rstd(deltas[-1], self.eps))
            if len(deltas) == 1:
                return base, base_coeffs, next_prior
            R_list = keys if keys is not None else [fused.rstd(d, self.eps) for d in deltas]
            if fused.fused_read_supported(base, deltas, R_list, base_coeffs):
                out, target = fused.FusedDattnRead.apply(self.query[step - 1], base, base_coeffs, next_prior,
                                                         *deltas, *R_list)
                return out, target, next_prior
            keys = None                                    # unsupported layout: the reference read, keys recomputed
        if keys is not None and len(keys) < len(deltas):
            keys.append(_rms_no_affine(deltas[-1].float(), self.eps))   # cache once per write
        if len(deltas) == 1:
            return base, base_coeffs, next_prior

        K_list = keys if keys is not None else [_rms_no_affine(d.float(), self.eps) for d in deltas]
        q = self.query[step - 1].float()
        inv = 1.0 / math.sqrt(self.dim)
        scores = torch.stack([torch.einsum("b t d, d -> b t", k, q) for k in K_list], dim=-1) * inv
        log_prior = torch.log(next_prior.clamp_min(1e-8))       # (n,)
        p0 = torch.softmax(log_prior, dim=-1)
        p = torch.softmax(scores + log_prior.view(1, 1, -1), dim=-1)
        mass = next_prior.sum()
        target_coeffs = next_prior.view(1, 1, -1) + mass * (p - p0.view(1, 1, -1))
        coeff_delta = (target_coeffs - base_coeffs).to(proposal.dtype)   # (b, t, n)
        correction = None
        for i, d in enumerate(deltas):
            term = coeff_delta[..., i].unsqueeze(-1) * d
            correction = term if correction is None else correction + term
        return base + correction, target_coeffs, next_prior


# --------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------

class LoopedTransformerModel(nn.Module):
    def __init__(self, config: LoopedTransformerConfig):
        super().__init__()
        self.config = config
        self.num_prelude_layers = config.num_prelude_layers
        self.num_coda_layers = config.num_coda_layers
        self.num_middle_layers = config.num_middle_layers
        self.num_middle_param_layers = config.loop_block_size
        self.share_idxs: list[int | None] = []

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        middle_blocks: list[LoopedTransformerBlock | None] = [None] * self.num_middle_param_layers
        layers = []
        for layer_idx in range(config.num_hidden_layers):
            block = LoopedTransformerBlock(config, layer_idx)
            if self.num_prelude_layers <= layer_idx < config.num_hidden_layers - self.num_coda_layers:
                share_idx = (layer_idx - self.num_prelude_layers) % self.num_middle_param_layers
                self.share_idxs.append(share_idx)
                if middle_blocks[share_idx] is None:
                    middle_blocks[share_idx] = block
                else:
                    self._tie_block_weights(block, middle_blocks[share_idx])
            else:
                self.share_idxs.append(None)
            layers.append(block)
        self.layers = nn.ModuleList(layers)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        # ---- the residual rule between block applications (attribute names are the state-dict keys) ----
        self.loop_residual_mode = config.loop_residual_mode
        self.middle_start = self.num_prelude_layers                  # P
        self.K = self.num_middle_param_layers                        # loop block size
        self.num_recursions = self.num_middle_layers // self.K       # L
        self.middle_end = self.num_prelude_layers + self.num_middle_layers
        self.loop_starts = {self.middle_start + l * self.K: l for l in range(self.num_recursions)}
        self.loop_ends = {self.middle_start + (l + 1) * self.K - 1: l for l in range(self.num_recursions)}
        mode = self.loop_residual_mode
        if mode == "delta_attnres":
            self.loop_dattn = LoopDeltaAttnRes(config.hidden_size, self.num_recursions, self.K,
                                               beta=config.skip0_beta, eps=config.norm_eps)
        elif mode in LINATTN_KINDS:
            # a reader on the loop axis, read at every iteration end
            self.linattn_source = config.linattn_source
            if self.linattn_source == "state":
                self.loop_linattn = LoopLinAttnState(config.hidden_size, self.num_recursions, kind=mode,
                                                     num_heads=config.attnres_num_heads, eps=config.norm_eps)
            else:
                self.loop_linattn = LoopLinAttnDelta(config.hidden_size, self.num_recursions, kind=mode, num_heads=1,
                                                     eps=config.norm_eps)
        self.gradient_checkpointing = False

    @staticmethod
    def _tie_block_weights(block: LoopedTransformerBlock, shared: LoopedTransformerBlock) -> None:
        block.attn_norm = shared.attn_norm
        block.mlp_norm = shared.mlp_norm
        block.mlp = shared.mlp
        block.attn.q_proj = shared.attn.q_proj
        block.attn.k_proj = shared.attn.k_proj
        block.attn.v_proj = shared.attn.v_proj
        block.attn.o_proj = shared.attn.o_proj
        block.attn.rotary = shared.attn.rotary

    def extra_repr(self) -> str:
        lines = [f"topology {self.config.topology()}, loop_residual_mode={self.loop_residual_mode}, "
                 f"beta={self.config.skip0_beta}"]
        for i, share_idx in enumerate(self.share_idxs):
            lines.append(f"layer {i}: " + ("unique" if share_idx is None else f"share<-middle[{share_idx}]"))
        return "\n".join(lines)

    def _run_layer(self, layer, hidden_states):
        if self.gradient_checkpointing and self.training:
            return checkpoint(layer, hidden_states, use_reentrant=False)
        return layer(hidden_states)

    def forward(self, input_ids: torch.LongTensor, output_hidden_states: bool = False):
        # the trainer model's forward, branch for branch; only the HF plumbing is gone
        hidden_states = self.embeddings(input_ids)
        mode = self.loop_residual_mode
        all_hidden_states = [] if output_hidden_states else None
        attn_states = None    # readers: the stored states (b, seq, dim)
        attn_inputs = None    # update interface only: the state entering each iteration, paired with attn_states
        loop_h0 = None        # Loopback: the loop input (prelude output)
        dattn_deltas, dattn_keys, dattn_coeffs, dattn_prior, x_before = [], [], None, None, None

        for layer_idx, layer in enumerate(self.layers):
            # ---- before a block application ----
            if mode in LINATTN_KINDS:
                if layer_idx == self.middle_start:
                    attn_states = [hidden_states] if self.linattn_source == "state" else []
                    attn_inputs = [] if self.linattn_source == "delta" else None
                if attn_inputs is not None and layer_idx in self.loop_starts:
                    attn_inputs.append(hidden_states)   # h_0 for iteration 0, then the previous read's output
            elif mode == "delta_attnres":
                if layer_idx == self.middle_start:
                    loop_h0 = hidden_states
                if self.middle_start <= layer_idx < self.middle_end:
                    x_before = hidden_states        # this block's input, for its update

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states = self._run_layer(layer, hidden_states)

            # ---- after it: the read ----
            if mode in LINATTN_KINDS and layer_idx in self.loop_ends:
                attn_states.append(hidden_states)          # the current state: a source and the query
                if self.linattn_source == "state":
                    hidden_states = self.loop_linattn.combine(attn_states)
                else:
                    hidden_states = self.loop_linattn.combine(attn_states, inputs=attn_inputs)
            elif mode == "delta_attnres" and self.middle_start <= layer_idx < self.middle_end:
                k = layer_idx - self.middle_start
                l, blk = divmod(k, self.K)
                dattn_deltas.append(hidden_states - x_before)   # the block update d_s
                hidden_states, dattn_coeffs, dattn_prior = self.loop_dattn.combine(
                    hidden_states, loop_h0, dattn_deltas, dattn_coeffs, dattn_prior, l, blk, keys=dattn_keys)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states, all_hidden_states


class LoopedTransformerLM(nn.Module):
    """The causal LM: `LoopedTransformerModel` + lm_head + next-token loss. `forward(input_ids, labels)` follows
    the trainer's convention: `labels` is aligned with `input_ids` (pass `labels=input_ids`) and shifted
    internally, the last position ignored."""

    def __init__(self, config: LoopedTransformerConfig):
        super().__init__()
        self.config = config
        self.model = LoopedTransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reset_parameters()

    # ---- init: the trainer model's _init_weights, applied the way HF's post_init applies it (children first,
    #      then the module itself, so a cell's reset_parameters runs after its Linear children were drawn) ----
    @torch.no_grad()
    def reset_parameters(self):
        cfg = self.config

        def init_weights(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=cfg.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=cfg.initializer_range)
            elif hasattr(module, "reset_parameters") and module is not self:
                module.reset_parameters()

        self.apply(init_weights)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.weight

    def forward(self, input_ids: torch.LongTensor, labels: torch.LongTensor | None = None,
                output_hidden_states: bool = False):
        hidden_states, all_hidden_states = self.model(input_ids, output_hidden_states=output_hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], -100)), dim=1)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits, "hidden_states": all_hidden_states}

    # ---- bookkeeping ----
    def num_params(self, exclude_embeddings: bool = True) -> int:
        seen, total = set(), 0
        skip = {id(self.model.embeddings.weight), id(self.lm_head.weight)} if exclude_embeddings else set()
        for p in self.parameters():
            if id(p) in seen or id(p) in skip:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    @classmethod
    def from_json(cls, path: str) -> "LoopedTransformerLM":
        return cls(LoopedTransformerConfig.from_json(path))

    def load_flame_state_dict(self, state_dict: dict, strict: bool = True):
        """Load a state dict of the trainer model (same keys); a shared block appears once per layer position
        there and the shared modules here accept every copy."""
        return self.load_state_dict(state_dict, strict=strict)


__all__ = ["LoopedTransformerConfig", "LoopedTransformerLM", "LoopedTransformerModel",
           "LoopedTransformerBlock", "LoopedAttention", "SwiGLUMLP", "RotaryEmbedding",
           "LoopDeltaAttnRes",
           "sdpa_causal", "LOOP_RESIDUAL_MODES"]
