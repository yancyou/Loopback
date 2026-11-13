"""The looped transformer the trainer builds (a HuggingFace-style causal LM on fla's kernels): a pre-norm
transformer block, the [prelude -> K x R shared blocks -> coda] stack, full-sequence causal flash attention, and
the residual rule between block applications (`loop_residual_mode`):

  * None             addition: Plain Loop; with `num_middle_param_layers=None` every block is unique, the
                     standard Transformer;
  * "delta_attnres"  Loopback (LoopDeltaAttnRes): the block updates are stored, geometric decay sets their total
                     weight (lambda = 1 - `skip0_beta` at every iteration end; the loop input keeps coefficient 1)
                     and one learned query per read reallocates it; zero queries at init; the reference read, or
                     the fused Triton read of fused_dattn.py with LOOP_DATTN_FUSED=1;
  * a reader kind    one of LINATTN_KINDS (linattn, retnet, mamba2, gla, hgrn2, deltanet, gdn, kda, rwkv7, lact,
                     mesa): a linear-attention reader on the loop axis, read at iteration ends, over the stored
                     states (`linattn_source` "state", loop_linattn_state.py, `attnres_num_heads` heads) or the
                     whole-iteration updates ("delta", loop_linattn_delta.py, one head).

The block: pre-RMSNorm (fla's fused norm, learnable gain, eps from the config) -> attention (q/k/v/o projections,
RoPE from rotary.py with theta from the config) -> pre-RMSNorm -> fla GatedMLP (SwiGLU, intermediate = the multiple
of 256 at or above 2/3 * hidden * ratio). Init N(0, initializer_range) on every Linear and the embedding; the
loop cells' own reset_parameters (query zero, readers at Plain Loop). The plain-PyTorch looped_transformer.py at
the repository root is this model without fla / flash-attn / transformers, bitwise in fp32
(tests/test_looped_transformer_port.py); state-dict keys are shared (model.embeddings, model.layers.*,
model.norm, model.loop_dattn.query, model.loop_linattn.*, lm_head).

Activation checkpointing is not applied inside the model; flame's `--activation_checkpoint.mode` wraps the
blocks in `model.layers`.
"""
from __future__ import annotations

import math
import os
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.deprecation import deprecate_kwarg

try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    try:
        from flash_attn import flash_attn_func
    except ImportError:

        def flash_attn_func(q, k, v, causal=False, **kwargs):
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if k.shape[1] != q.shape[1]:
                groups = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            if causal:
                q_len = q.size(-2)
                k_len = k.size(-2)
                offset = max(k_len - q_len, 0)
                q_pos = torch.arange(q_len, device=q.device).view(q_len, 1) + offset
                k_pos = torch.arange(k_len, device=q.device).view(1, k_len)
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v).transpose(1, 2).contiguous()

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as TransformerMLP

from .configuration_looped_transformer import LINATTN_KINDS, LoopedTransformerConfig
from .loop_linattn_delta import LINATTN_KINDS as _BD_LINATTN_KINDS, LoopLinAttn as LoopLinAttnDelta
from .loop_linattn_state import LINATTN_KINDS as _STATE_LINATTN_KINDS, LoopLinAttn as LoopLinAttnState
from .rotary import RotaryEmbedding

# the config's plain tuple must be exactly the vendored cells' own registry (both files define the eleven)
assert tuple(LINATTN_KINDS) == tuple(_STATE_LINATTN_KINDS) == tuple(_BD_LINATTN_KINDS), (LINATTN_KINDS, _STATE_LINATTN_KINDS, _BD_LINATTN_KINDS)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer


def _rms_no_affine(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm without a gain: the keys of the stored updates."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


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
    per token. The reference read (below) and the fused Triton read (fused_dattn.py, LOOP_DATTN_FUSED=1) compute
    the same function (tests/test_fused_dattn.py); the plain-PyTorch cell is checked bitwise in
    tests/test_looped_transformer_port.py."""

    def __init__(self, config, num_loops: int, num_blocks: int, beta: float = 0.75, eps: float = 1e-6):
        super().__init__()
        if num_loops < 1 or num_blocks < 1:
            raise ValueError(f"delta_attnres needs positive num_loops/num_blocks, got {num_loops}/{num_blocks}")
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError(f"delta_attnres needs 0 <= beta < 1, got {beta}")
        self.dim = int(config.hidden_size)
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

    def combine(self, proposal, h0, deltas, coeffs, prior_coeffs, loop_idx, block_idx,
                keys=None, stack=None):
        """`keys`, if given, is a caller-owned list (one entry per stored update, extended here by one) caching
        each update's RMSNorm scale (fused read) or normalized key (reference read), so a forward pass normalizes
        every update once. `stack` (fused read only) is the pass's fused_dattn.DattnStack: the updates are its
        slices and the kernels index it directly (a faster backward; the list layout's results bitwise, or to fp32
        round-off when the stack also holds the scores, LOOP_DATTN_DOTS=1, the default)."""
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
        if fused and stack is not None:   # the stacked layout: the deltas are the stack's slices
            if keys is not None and len(keys) < len(deltas):
                keys.append(stack.rstd(step, deltas[-1], self.eps))
            if len(deltas) == 1:
                return base, base_coeffs, next_prior
            R_list = keys if keys is not None else [fused._alias(stack.R, i) for i in range(len(deltas))]
            out, target = fused.FusedDattnReadStk.apply(stack, step, self.query[step - 1], base, base_coeffs,
                                                        next_prior, *deltas, *R_list)
            return out, target, next_prior
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
        correction = None                                        # sum_i c_i * delta_i, no stacking
        for i, d in enumerate(deltas):
            term = coeff_delta[..., i].unsqueeze(-1) * d
            correction = term if correction is None else correction + term
        return base + correction, target_coeffs, next_prior


_FUSED_DATTN = None


def _fused_dattn():
    """The fused Loopback read (fused_dattn.py), imported on first use; False unless LOOP_DATTN_FUSED=1, or when
    Triton is unavailable."""
    global _FUSED_DATTN
    if _FUSED_DATTN is None:
        if os.environ.get("LOOP_DATTN_FUSED", "0") != "1":
            _FUSED_DATTN = False
        else:
            try:
                from . import fused_dattn
                _FUSED_DATTN = fused_dattn
            except Exception as e:  # noqa: BLE001 -- any import failure means: use the reference read
                warnings.warn(f"delta_attnres: fused read unavailable ({e!r}); using the reference read")
                _FUSED_DATTN = False
    return _FUSED_DATTN


def _validate_no_padding_attention_mask(attention_mask: torch.Tensor | None) -> None:
    if attention_mask is None:
        return
    if not bool(torch.all(attention_mask != 0).item()):
        raise ValueError("looped_transformer does not support padded attention_mask.")



class LoopedAttention(nn.Module):
    """Multi-head causal attention over the whole sequence: q/k/v/o projections, RoPE, one flash-attention call."""

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
        batch_size, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim)
        # RotaryEmbedding.forward reads its cos/sin cache without refreshing it; grow it here (a no-op after the
        # first call at this length)
        self.rotary._update_cos_sin_cache(q_len, device=q.device, dtype=q.dtype)
        q, k = self.rotary(q, k, seqlen_offset=0)
        attn_output = flash_attn_func(q, k, v, causal=True)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        attn_output = attn_output.reshape(batch_size, q_len, self.hidden_size).to(hidden_states.dtype)
        return self.o_proj(attn_output)


class LoopedTransformerBlock(GradientCheckpointingLayer):
    def __init__(self, config: LoopedTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = LoopedAttention(config, layer_idx)
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[Any]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn(self.attn_norm(hidden_states))
        if self.config.fuse_norm:      # the fused norm adds the residual and returns the new residual stream
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        return residual + self.mlp(hidden_states, **kwargs)


class LoopedTransformerPreTrainedModel(PreTrainedModel):
    config_class = LoopedTransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LoopedTransformerBlock"]
    _supports_cache_class = False

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()


class LoopedTransformerModel(LoopedTransformerPreTrainedModel):
    def __init__(self, config: LoopedTransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # ---- topology: unique prelude and coda blocks, shared middle blocks ----
        self.num_prelude_layers = config.num_prelude_layers
        self.num_coda_layers = config.num_coda_layers
        num_middle_layers = config.num_hidden_layers - self.num_prelude_layers - self.num_coda_layers
        self.num_middle_param_layers = config.num_middle_param_layers or num_middle_layers
        self.share_idxs: list[int | None] = []

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        middle_blocks: list[LoopedTransformerBlock | None] = [None] * self.num_middle_param_layers
        layers = []
        for layer_idx in range(config.num_hidden_layers):
            block = LoopedTransformerBlock(config, layer_idx)
            if self.num_prelude_layers <= layer_idx < config.num_hidden_layers - self.num_coda_layers:
                middle_pos = layer_idx - self.num_prelude_layers
                share_idx = middle_pos % self.num_middle_param_layers
                self.share_idxs.append(share_idx)
                if middle_blocks[share_idx] is None:
                    middle_blocks[share_idx] = block
                else:
                    self._tie_block_weights(block, middle_blocks[share_idx])
            else:
                self.share_idxs.append(None)
            layers.append(block)
        self.layers = nn.ModuleList(layers)
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        # ---- the residual rule between block applications (attribute names are the state-dict keys) ----
        self.loop_residual_mode = getattr(config, "loop_residual_mode", None)
        self.num_middle_layers = num_middle_layers
        self.middle_start = self.num_prelude_layers                  # P
        self.K = self.num_middle_param_layers                        # loop block size
        self.num_recursions = num_middle_layers // self.K            # L
        self.middle_end = self.num_prelude_layers + num_middle_layers
        self.loop_starts = {self.middle_start + l * self.K: l for l in range(self.num_recursions)}
        self.loop_ends = {self.middle_start + (l + 1) * self.K - 1: l for l in range(self.num_recursions)}
        if self.loop_residual_mode == "delta_attnres":
            self.loop_dattn = LoopDeltaAttnRes(config, self.num_recursions, self.K, beta=config.skip0_beta,
                                               eps=config.norm_eps)
        elif self.loop_residual_mode in LINATTN_KINDS:
            # a reader on the loop axis, read at every iteration end
            self.linattn_source = config.linattn_source
            if self.linattn_source == "state":
                self.loop_linattn = LoopLinAttnState(config.hidden_size, self.num_recursions, kind=self.loop_residual_mode,
                                                     num_heads=config.attnres_num_heads, eps=config.norm_eps)
            else:
                self.loop_linattn = LoopLinAttnDelta(config.hidden_size, self.num_recursions, kind=self.loop_residual_mode,
                                                     num_heads=1, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    @staticmethod
    def _tie_block_weights(block: "LoopedTransformerBlock", shared: "LoopedTransformerBlock") -> None:
        """Tie every learnable weight of `block` to `shared` (norms, attention projections, rotary, MLP)."""
        block.attn_norm = shared.attn_norm
        block.mlp_norm = shared.mlp_norm
        block.mlp = shared.mlp
        block.attn.q_proj = shared.attn.q_proj
        block.attn.k_proj = shared.attn.k_proj
        block.attn.v_proj = shared.attn.v_proj
        block.attn.o_proj = shared.attn.o_proj
        block.attn.rotary = shared.attn.rotary

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def extra_repr(self) -> str:
        lines = [
            f"prelude: {self.num_prelude_layers}, coda: {self.num_coda_layers}, "
            f"middle_param_layers: {self.num_middle_param_layers}, full-sequence attention",
        ]
        for i, share_idx in enumerate(self.share_idxs):
            tag = "unique" if share_idx is None else f"share<-middle[{share_idx}]"
            lines.append(f"layer {i}: {tag}")
        return "\n".join(lines)

    def _run_layer(self, layer, hidden_states):
        if self.gradient_checkpointing and self.training:
            return checkpoint(layer, hidden_states, use_reentrant=False)
        return layer(hidden_states)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            warnings.warn(
                "`LoopedTransformerModel` does not support output attention weights, "
                "so `output_attentions` is set to `False`.",
            )
            output_attentions = False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        _validate_no_padding_attention_mask(attention_mask)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        all_hidden_states = () if output_hidden_states else None
        mode = self.loop_residual_mode

        hidden_states = inputs_embeds
        attn_states = None    # readers: the stored states (b, seq, dim)
        attn_inputs = None    # update interface only: the state entering each iteration, paired with attn_states
        loop_h0 = None        # Loopback: the loop input (prelude output)
        dattn_deltas, dattn_keys, dattn_coeffs, dattn_prior, x_before = [], [], None, None, None
        dattn_stack = None    # fused read: the updates in one stack, their gradients accumulated in the kernels
        if (mode == "delta_attnres" and not self.gradient_checkpointing and _fused_dattn()
                and os.environ.get("LOOP_DATTN_STACK", "1") == "1"):
            scores = os.environ.get("LOOP_DATTN_DOTS", "1") == "1"    # each write stores its scores for the reads
            dattn_stack = _fused_dattn().DattnStack(self.loop_dattn.num_steps, tuple(inputs_embeds.shape),
                                                    inputs_embeds.dtype, inputs_embeds.device,
                                                    query=self.loop_dattn.query if scores else None,
                                                    eps=self.loop_dattn.eps)

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
                all_hidden_states = all_hidden_states + (hidden_states,)

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
                if dattn_stack is not None:
                    dattn_deltas.append(dattn_stack.write(k, hidden_states, x_before))   # d_s, into its slice
                else:
                    dattn_deltas.append(hidden_states - x_before)   # the block update d_s
                hidden_states, dattn_coeffs, dattn_prior = self.loop_dattn.combine(
                    hidden_states, loop_h0, dattn_deltas, dattn_coeffs, dattn_prior, l, blk,
                    keys=dattn_keys, stack=dattn_stack)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, None] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class LoopedTransformerForCausalLM(LoopedTransformerPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = LoopedTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[Any],
    ) -> dict[str, Any]:
        model_inputs: dict[str, Any]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs["past_key_values"] = None
        model_inputs["use_cache"] = False
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        for key, value in kwargs.items():
            if key not in {"cache_position", "position_ids"}:
                model_inputs[key] = value
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        fuse_lce_training = self.config.fuse_linear_cross_entropy and labels is not None
        logits = None if fuse_lce_training else self.lm_head(
            hidden_states[:, -logits_to_keep:] if logits_to_keep else hidden_states
        )
        loss = None
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                if logits_to_keep:
                    labels = labels[:, -logits_to_keep:]
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



__all__ = [
    "LoopedAttention",
    "LoopedTransformerBlock",
    "LoopedTransformerForCausalLM",
    "LoopedTransformerModel",
]
