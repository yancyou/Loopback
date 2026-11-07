"""Independent reference implementation of the loop-axis linear-attention family.

This file exists to be a NUMERICAL ORACLE for the test suite. It is written from the
written specification of the family, not from `loop_linattn_delta.py`. Nothing here is
imported from the implementation, no helper is shared with it, and the factorisation is
deliberately different: everything is a plain Python loop over visits with fully
materialised `dim x dim` matrices, one token at a time. That is slow and it does not
matter -- the tests run on tiny CPU tensors, and an oracle that shares code with the
thing it checks is not an oracle (a reference that reaches into the module's own private
helpers moves both sides of an assertion together when one of them is wrong).

The only thing taken from `loop_linattn_delta.py` is the *interface*: parameter names, tensor
shapes, and the initialiser constants, so that `reference_junction` can be handed the
module's own `state_dict()` unchanged.


THE SPECIFICATION
-----------------
For each token independently, the junction sees one visit per loop iteration (the
model's `num_loops` is the iteration count R). Visit `r` has an input state `in_r` and a
raw block output `out_r`. The memory source is the whole-visit update

    delta_r = out_r - in_r                               for every r, under one rule

The pre-loop state `h_0 = in_0` is NOT in the memory; it is added back after the read.
The memory operates on the whole hidden vector: a single head of width `dim`, with no
head splitting.

    k_r = delta_r / max(||delta_r||, eps)                 key   -- normalised delta
    v_r = delta_r                                         value -- raw delta
    q   = out_last / max(||out_last||, eps)               query -- normalised state

The state is materialised in VALUE-BY-KEY orientation, `S[dim, dim]`, so that

    S_j = S_{j-1} A_j + c_j v_j k_j^T
    u   = S q
    y   = (1 - alpha) * out_last + alpha * (h_0 + u)

Right-multiplication by `A_j` means every decay acts on the KEY axis, which is what
makes `diag(g)` a per-channel key gate and `(I - beta k k^T)` an erase along the current
key.

There are `max(n - 1, 1)` transitions for `n` sources -- one fewer than the number of
sources, because the first source is written into an empty memory and there is nothing
there to decay. Consequently:

  * write-side parameters (`b_beta`, `dt_bias`, `b_eta`, `b_gamma`) have one row per
    SOURCE and are indexed by `r`;
  * transition-side parameters (`b_gate`, `b_removal`) have one row per TRANSITION and
    are indexed by `r - 1`, and are never read at `r == 0`;
  * except for `mamba2` and `hgrn2`, whose gate also sets the write at `r == 0`, so
    their gate row is indexed by SOURCE.

Rows are clamped at use (`min(index, rows - 1)`), so a junction run deeper than the
depth it was built at reuses the last row rather than failing.

The eleven transitions, by their textbook definitions:

    linattn   S_j = S_{j-1} + v k^T                      additive, no decay
    retnet    S_j = g S_{j-1} + v k^T                    frozen per-head gamma
    mamba2    S_j = a_j S_{j-1} + dt_j v k^T             a_j = exp(-dt_j * exp(A_log)),
                                                         dt_j = softplus(.) > 0 drives
                                                         BOTH decay and write scale
    gla       S_j = S_{j-1} diag(g_j) + v k^T            diagonal data-independent gate
    hgrn2     S_j = S_{j-1} diag(g_j) + v (1 - g_j)^T    input gate tied to 1 - forget
    deltanet  S_j = S_{j-1} (I - b_j k k^T) + b_j v k^T  erase along key, then write
    gdn       S_j = S_{j-1} a_j (I - b_j k k^T)
                                        + b_j v k^T      deltanet + scalar decay
    kda       S_j = S_{j-1} diag(g_j) (I - b_j k k^T)
                                        + b_j v k^T      diagonal decay BEFORE the erase
    rwkv7     S_j = S_{j-1} (diag(g_j)
                             - kap (rho_j * kap)^T)
                                        + v k^T          additive DPLR: diagonal PLUS
                                                         rank one, per-channel removal
                                                         strength rho, removal direction
                                                         kap = the current normalised
                                                         delta, decoupled from the write
    lact      one gradient step per junction over the whole visible prefix
    mesa      ridge regression over the visible sources, solved in closed form

`lact` and `mesa` are not recurrences at all; they are computed directly from the whole
visible prefix, which is why they get their own paths below.


WHAT THIS ORACLE DELIBERATELY DOES NOT COVER
--------------------------------------------
* `num_heads > 1`. The specification says a single head of width `dim`. Parameter
  tensors carry a head axis; this file asserts it has extent 1 rather than inventing a
  splitting rule.
* The implementation's saturation guards (log-decay / dt / eta ceilings). Those are
  numerical armour, not semantics. This oracle is exact in the unsaturated regime, which
  is the only regime tiny test tensors reach.

Everything is computed in float64 by default -- an oracle should be more precise than
the thing it checks, not equally imprecise.


WHAT `self_check` DOES AND DOES NOT PIN
---------------------------------------
It asserts the four contracted properties -- the alpha=0 identity is bit-exact, an
all-zero delta sequence is finite forward and backward, every mode returns the input
shape, and the gate bias retains 0.95 over `max(L - 1, 1)` transitions -- and then a
layer of value-level checks that the four alone do not imply: each mode collapses onto
its stated parent when its extra mechanism is neutralised, no mode is silently identical
to `linattn` or to its parent at init, `retnet` at gamma=0 and `hgrn2` at forget=0 hit
their closed forms, `mamba2`'s dt scales the write and not only the decay, and `mesa` on
a single source matches the hand-derived Sherman-Morrison solution.

It does NOT prove the transitions correct. Notably it does not pin the ORDER of `kda`'s
diagonal decay and rank-one erase: the two orders agree at every setting that has a
closed form, and separating them needs a general two-source case, which could only be
checked by re-deriving the same formula. That, and every other residual degree of
freedom, is what comparing this oracle against `loop_linattn_delta.py` in the test
suite is for. A self check is a guard against this file rotting, not a substitute for the
differential test.
"""

from __future__ import annotations

import math
import sys

import torch
from torch import Tensor

__all__ = [
    "MODES",
    "reference_junction",
    "default_params",
    "gate_bias_init",
    "inv_softplus",
    "self_check",
]

MODES = ("linattn", "retnet", "mamba2", "gla", "hgrn2",
         "deltanet", "gdn", "kda", "rwkv7", "lact", "mesa")

# Which modes carry an erase strength `beta`.
_HAS_BETA = frozenset({"deltanet", "gdn", "kda"})
# Shape of the decay each mode applies: None | "fixed" | "scalar" | "diag".
_DECAY = {
    "linattn": None, "retnet": "fixed", "mamba2": "scalar", "gla": "diag",
    "hgrn2": "diag", "deltanet": None, "gdn": "scalar", "kda": "diag",
    "rwkv7": "diag", "lact": None, "mesa": None,
}
# Modes whose gate row is indexed by source rather than by transition, because the same
# gate also sets the write at r == 0.
_GATE_BY_SOURCE = frozenset({"mamba2", "hgrn2"})

# Initialiser constants, matching the implementation so a fresh module and
# `default_params` start from the same numbers.
GATE_RETENTION_INIT = 0.95          # retention of the oldest source after ALL transitions
SSM_DECAY = math.sqrt(1.0 / 5.0)    # mamba2's step-one decay at init
RIDGE_INIT = 1.0                    # mesa's ridge at init
RIDGE_MIN = 1e-4                    # floor added to softplus(ridge)
LACT_ETA_INIT = 1.0                 # lact's step size at init

# --------------------------------------------------------------------------------------
# Small numerical helpers. Written out rather than imported so this file stands alone.
# --------------------------------------------------------------------------------------

def inv_softplus(y: float) -> float:
    """The x with softplus(x) == y."""
    return math.log(math.expm1(y))


def gate_bias_init(num_loops: int) -> float:
    """The per-transition gate bias whose retention over the WHOLE loop axis is 0.95.

    A gated mode should begin as its ungated sibling, so what must start near one is the
    retention of the oldest source after every transition has been applied to it -- and
    the specification says there are `max(L - 1, 1)` transitions for `L` visits, not `L`.
    So the exponent is the transition count:

        sigmoid(b) ** max(L - 1, 1) == 0.95

    which is exactly what `self_check` asserts.
    """
    steps = max(int(num_loops) - 1, 1)
    per_step = GATE_RETENTION_INIT ** (1.0 / steps)
    return math.log(per_step / (1.0 - per_step))


def _unit(v: Tensor, eps: float) -> Tensor:
    """eps-clamped L2 normalisation: v / max(||v||, eps).

    Written on the SQUARED norm so the zero vector is differentiable. The obvious
    `v / clamp(sqrt((v*v).sum()), min=eps)` is NaN-producing at the origin: d||v||/dv is
    v / ||v|| = 0 / 0 there, and the clamp's zero gradient multiplies that NaN rather
    than killing it. Flooring the ARGUMENT of the sqrt instead keeps the whole chain
    finite. Forward-identical either way, since sqrt(max(||v||^2, eps^2)) is exactly
    max(||v||, eps).

    (`torch.linalg.norm` happens to define its own zero subgradient and so is also safe;
    the hand-rolled sqrt is not. `self_check` pins the safe behaviour either way.)
    """
    sq = (v * v).sum(-1, keepdim=True)
    return v / torch.sqrt(torch.clamp(sq, min=eps * eps))


def _lookup(params, name: str) -> Tensor | None:
    """Fetch `name` from a parameter dict, tolerating a module prefix.

    Accepts both a bare `state_dict()` of the junction (`"b_gate"`) and a full model
    state dict where the junction sits under some path (`"layers.3.junction.b_gate"`).
    """
    if name in params:
        return params[name]
    suffix = "." + name
    hits = [k for k in params if k.endswith(suffix)]
    if not hits:
        return None
    if len(hits) > 1:
        raise KeyError(
            f"{name!r} is ambiguous in this parameter dict: {sorted(hits)}. Pass the "
            f"junction's own state_dict, or strip the prefix first.")
    return params[hits[0]]


def _require(params, name: str, mode: str) -> Tensor:
    t = _lookup(params, name)
    if t is None:
        raise KeyError(f"mode {mode!r} needs parameter {name!r}, which is not in the "
                       f"parameter dict (have: {sorted(params)})")
    return t


def _row(t: Tensor, index: int) -> Tensor:
    """Row `index` of a per-junction parameter, clamped to the last available row."""
    return t[min(index, t.shape[0] - 1)]


def _scalar_row(t: Tensor, index: int, name: str) -> Tensor:
    """A `[rows, heads]` parameter at one row, as a 0-d tensor. Single head only."""
    row = _row(t, index)
    if row.numel() != 1:
        raise NotImplementedError(
            f"{name} has {row.numel()} heads; this oracle implements the specified "
            f"single-head memory of width dim only.")
    return row.reshape(())


def _vector_row(t: Tensor, index: int, dim: int, name: str) -> Tensor:
    """A `[rows, heads, head_dim]` parameter at one row, as a `[dim]` tensor."""
    row = _row(t, index)
    if row.numel() != dim:
        raise NotImplementedError(
            f"{name} row has {row.numel()} elements but the hidden width is {dim}; this "
            f"oracle implements the specified single-head memory only.")
    return row.reshape(dim)


# --------------------------------------------------------------------------------------
# The memory state, one token at a time.
# --------------------------------------------------------------------------------------

def _recurrent_state(mode: str, deltas: list[Tensor], params, eps: float) -> Tensor:
    """Run S_j = S_{j-1} A_j + c_j v_j k_j^T over the visits and return S.

    `deltas` is the list of `[dim]` sources for ONE token. Everything is materialised:
    `S` is a real `dim x dim` matrix, `A_j` is a real `dim x dim` matrix, and the
    products are ordinary matmuls. No factorisation, no chunking, no fused kernel.
    """
    dim = deltas[0].shape[0]
    ref = deltas[0]
    eye = torch.eye(dim, dtype=ref.dtype, device=ref.device)
    state = torch.zeros(dim, dim, dtype=ref.dtype, device=ref.device)

    for r, delta in enumerate(deltas):
        key = _unit(delta, eps)          # k_r
        value = delta                    # v_r
        first = (r == 0)                 # first source lands in an empty memory

        if mode == "linattn":
            # No decay at all. The transition is the identity.
            write = torch.outer(value, key)

        elif mode == "retnet":
            # A frozen per-head scalar, gamma_h = 1 - 2^-(5+h), supplied as a buffer.
            gamma = _require(params, "gamma", mode).reshape(-1)
            if gamma.numel() != 1:
                raise NotImplementedError(
                    "retnet gamma carries more than one head; single-head only.")
            if not first:
                state = gamma[0] * state
            write = torch.outer(value, key)

        elif mode == "mamba2":
            # SSD discretisation. One positive dt per source drives both the decay and
            # the write scale, which is why the row is indexed by source and is live at
            # r == 0 even though no decay is applied there.
            dt_bias = _require(params, "dt_bias", mode)
            a_log = _require(params, "A_log", mode).reshape(-1)
            if a_log.numel() != 1:
                raise NotImplementedError("mamba2 A_log carries more than one head.")
            dt = torch.nn.functional.softplus(_scalar_row(dt_bias, r, "dt_bias"))
            if not first:
                state = torch.exp(-dt * torch.exp(a_log[0])) * state
            write = dt * torch.outer(value, key)

        elif mode == "gla":
            # A diagonal, data-independent key gate, one row per TRANSITION.
            if not first:
                gate = torch.sigmoid(
                    _vector_row(_require(params, "b_gate", mode), r - 1, dim, "b_gate"))
                state = state @ torch.diag(gate)
            write = torch.outer(value, key)

        elif mode == "hgrn2":
            # One gate does both jobs: the write vector is the complement of the forget
            # gate rather than beta * khat, so the gate row is indexed by SOURCE and is
            # live at r == 0. The value is still the raw delta.
            gate = torch.sigmoid(
                _vector_row(_require(params, "b_gate", mode), r, dim, "b_gate"))
            if not first:
                state = state @ torch.diag(gate)
            write = torch.outer(value, 1.0 - gate)

        elif mode == "deltanet":
            # Erase along the current key with strength beta, then write with the same
            # strength. At r == 0 the erase hits an empty memory, but the write scale is
            # still beta, so the row is indexed by source.
            beta = torch.sigmoid(
                _scalar_row(_require(params, "b_beta", mode), r, "b_beta"))
            if not first:
                state = state @ (eye - beta * torch.outer(key, key))
            write = beta * torch.outer(value, key)

        elif mode == "gdn":
            # DeltaNet plus a scalar decay on the transition.
            beta = torch.sigmoid(
                _scalar_row(_require(params, "b_beta", mode), r, "b_beta"))
            if not first:
                decay = torch.sigmoid(
                    _scalar_row(_require(params, "b_gate", mode), r - 1, "b_gate"))
                state = state @ (decay * (eye - beta * torch.outer(key, key)))
            write = beta * torch.outer(value, key)

        elif mode == "kda":
            # A diagonal decay applied BEFORE the rank-one erase: diag(g) then
            # (I - beta k k^T), in that order.
            beta = torch.sigmoid(
                _scalar_row(_require(params, "b_beta", mode), r, "b_beta"))
            if not first:
                gate = torch.sigmoid(
                    _vector_row(_require(params, "b_gate", mode), r - 1, dim, "b_gate"))
                state = state @ torch.diag(gate) @ (eye - beta * torch.outer(key, key))
            write = beta * torch.outer(value, key)

        elif mode == "rwkv7":
            # Additive DPLR: diagonal PLUS a rank-one removal, not diagonal TIMES
            # (I - rank one). The removal strength is per channel and decoupled from the
            # write, which here carries no beta at all; the removal DIRECTION is the
            # current normalised delta.
            if not first:
                gate = torch.sigmoid(
                    _vector_row(_require(params, "b_gate", mode), r - 1, dim, "b_gate"))
                removal = torch.sigmoid(
                    _vector_row(_require(params, "b_removal", mode), r - 1, dim,
                                "b_removal"))
                kappa = key
                state = state @ (torch.diag(gate) - torch.outer(kappa, removal * kappa))
            write = torch.outer(value, key)

        else:
            raise AssertionError(f"unreachable: {mode!r} is not a recurrent mode")

        state = state + write

    return state


def _lact_state(deltas: list[Tensor], params, eps: float) -> Tensor:
    """One gradient step, taken at this junction, over the whole visible prefix.

    The memory starts empty and the objective is the usual per-source regression loss
    weighted by LaCT's per-source step size:

        L(W) = sum_j eta_j * 0.5 * || W k_j - v_j ||^2
        W    = 0 - grad L(0)

    The gradient is accumulated term by term below rather than simplified, so the
    "one step over the prefix" reading is visible in the code.
    """
    dim = deltas[0].shape[0]
    ref = deltas[0]
    weight = torch.zeros(dim, dim, dtype=ref.dtype, device=ref.device)
    grad = torch.zeros(dim, dim, dtype=ref.dtype, device=ref.device)

    b_eta = _require(params, "b_eta", "lact")
    for r, delta in enumerate(deltas):
        key = _unit(delta, eps)
        value = delta
        eta = torch.nn.functional.softplus(_scalar_row(b_eta, r, "b_eta"))
        residual = weight @ key - value
        grad = grad + eta * torch.outer(residual, key)

    return weight - grad


def _mesa_state(deltas: list[Tensor], params, eps: float) -> Tensor:
    """Ridge regression over the visible sources, in closed form.

        W = argmin  sum_j gamma_j || W k_j - v_j ||^2  +  lambda || W ||_F^2
          = (sum_j gamma_j v_j k_j^T) (sum_j gamma_j k_j k_j^T + lambda I)^-1

    gamma_j weights how much source j is fitted; lambda keeps the system conditioned and
    at large lambda collapses the solution onto a single small gradient step.
    """
    dim = deltas[0].shape[0]
    ref = deltas[0]
    cross = torch.zeros(dim, dim, dtype=ref.dtype, device=ref.device)   # sum g v k^T
    gram = torch.zeros(dim, dim, dtype=ref.dtype, device=ref.device)    # sum g k k^T

    b_gamma = _require(params, "b_gamma", "mesa")
    for r, delta in enumerate(deltas):
        key = _unit(delta, eps)
        value = delta
        gamma = torch.sigmoid(_scalar_row(b_gamma, r, "b_gamma"))
        cross = cross + gamma * torch.outer(value, key)
        gram = gram + gamma * torch.outer(key, key)

    ridge_raw = _require(params, "ridge", "mesa").reshape(-1)
    if ridge_raw.numel() != 1:
        raise NotImplementedError("mesa ridge carries more than one head.")
    lam = torch.nn.functional.softplus(ridge_raw[0]) + RIDGE_MIN

    eye = torch.eye(dim, dtype=ref.dtype, device=ref.device)
    system = gram + lam * eye
    # W = cross @ system^-1. `system` is symmetric, so W^T = system^-1 @ cross^T.
    return torch.linalg.solve(system, cross.transpose(0, 1)).transpose(0, 1)


# --------------------------------------------------------------------------------------
# The junction.
# --------------------------------------------------------------------------------------

def reference_junction(params, ins, outs, mode: str, *, eps: float = 1e-6,
                       dtype: torch.dtype = torch.float64) -> Tensor:
    """Reference junction output for one named mode.

    Args:
      params: mapping of parameter name to tensor. A junction's `state_dict()` works
        directly, as does a full model state dict provided exactly one entry matches
        each needed name.
      ins: list of per-visit input states, `ins[r]` is `in_r`, shape `[..., dim]`.
        `ins[0]` is the pre-loop state `h_0`.
      outs: list of per-visit raw block outputs, same length and shapes. `outs[-1]` is
        the current absolute state, which is both the query source and the identity
        branch.
      mode: one of `MODES`.
      eps: normalisation floor for the eps-clamped L2 normalisation.
      dtype: compute dtype; float64 by default because this is an oracle.

    Returns:
      `y = (1 - alpha) * out_last + alpha * (h_0 + S q)`, shape `[..., dim]`, in `dtype`.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not isinstance(ins, (list, tuple)) or not isinstance(outs, (list, tuple)):
        raise TypeError("ins and outs must be lists of per-visit tensors")
    if len(ins) != len(outs):
        raise ValueError(f"ins and outs must have equal length, got {len(ins)} and "
                         f"{len(outs)}")
    if len(ins) < 1:
        raise ValueError("need at least one visit, i.e. one block delta")
    if not (eps > 0 and math.isfinite(eps)):
        raise ValueError(f"eps must be positive and finite, got {eps!r}")

    shape = tuple(ins[0].shape)
    if len(shape) < 1:
        raise ValueError("states must have a trailing hidden dimension")
    for name, seq in (("ins", ins), ("outs", outs)):
        for r, t in enumerate(seq):
            if tuple(t.shape) != shape:
                raise ValueError(f"{name}[{r}] has shape {tuple(t.shape)}, expected "
                                 f"{shape}; every visit state must match")

    device = ins[0].device
    # Cast the parameters into the compute dtype up front. A module's state_dict is
    # float32 while this oracle works in float64, and mixing them raises deep inside a
    # matmul rather than anywhere informative.
    params = {k: (v.to(dtype=dtype, device=device) if isinstance(v, Tensor) else v)
              for k, v in params.items()}
    dim = shape[-1]
    n_tokens = 1
    for s in shape[:-1]:
        n_tokens *= s

    # Flatten every leading axis: the junction treats each token independently.
    flat_ins = [t.to(dtype=dtype, device=device).reshape(n_tokens, dim) for t in ins]
    flat_outs = [t.to(dtype=dtype, device=device).reshape(n_tokens, dim) for t in outs]
    deltas = [o - i for i, o in zip(flat_ins, flat_outs)]

    h0 = flat_ins[0]
    out_last = flat_outs[-1]

    reads = []
    for token in range(n_tokens):
        token_deltas = [d[token] for d in deltas]
        if mode == "lact":
            state = _lact_state(token_deltas, params, eps)
        elif mode == "mesa":
            state = _mesa_state(token_deltas, params, eps)
        else:
            state = _recurrent_state(mode, token_deltas, params, eps)
        query = _unit(out_last[token], eps)
        reads.append(state @ query)
    read = torch.stack(reads, dim=0)

    alpha = _require(params, "alpha", mode).reshape(-1)
    if alpha.numel() != 1:
        raise ValueError(f"alpha must be a single scalar, got {alpha.numel()} elements")
    a = alpha[0]

    # Written as two terms, not as `out_last + a * (...)`: at alpha == 0 the second term
    # is exactly zero and the first is exactly `out_last`, so the identity is bit-exact.
    y = (1.0 - a) * out_last + a * (h0 + read)
    return y.reshape(shape)


def default_params(dim: int, num_loops: int, mode: str, *,
                   dtype: torch.dtype = torch.float64, device=None) -> dict:
    """The parameter dict a freshly initialised junction should have, per the spec.

    Built here rather than imported so `self_check` never touches the implementation.
    Note that `b_gate` uses this file's `gate_bias_init`, whose exponent is the
    TRANSITION count `max(L - 1, 1)`, as the specification requires.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if num_loops < 1:
        raise ValueError(f"num_loops must be at least 1, got {num_loops!r}")

    kw = {"dtype": dtype, "device": device}
    rows = num_loops                                    # one row per SOURCE
    grows = rows if mode in _GATE_BY_SOURCE else max(num_loops - 1, 1)  # per TRANSITION
    heads = 1
    params: dict = {"alpha": torch.zeros(1, **kw)}

    if mode in _HAS_BETA:
        params["b_beta"] = torch.zeros(rows, heads, **kw)

    decay = _DECAY[mode]
    if decay == "fixed":
        # gamma_h = 1 - 2^-(5+h); a single head means h = 0.
        params["gamma"] = torch.full((1, 1, heads, 1), 1.0 - 2.0 ** -5.0, **kw)
    elif mode == "mamba2":
        params["A_log"] = torch.zeros(heads, **kw)
        params["dt_bias"] = torch.full(
            (rows, heads), inv_softplus(-math.log(SSM_DECAY)), **kw)
    elif decay == "scalar":
        params["b_gate"] = torch.full((grows, heads), gate_bias_init(num_loops), **kw)
    elif decay == "diag":
        params["b_gate"] = torch.full(
            (grows, heads, dim), gate_bias_init(num_loops), **kw)

    if mode == "rwkv7":
        params["b_removal"] = torch.zeros(grows, heads, dim, **kw)
    if mode == "lact":
        params["b_eta"] = torch.full((rows, heads), inv_softplus(LACT_ETA_INIT), **kw)
    if mode == "mesa":
        params["b_gamma"] = torch.zeros(rows, heads, **kw)
        params["ridge"] = torch.full((heads,), inv_softplus(RIDGE_INIT - RIDGE_MIN), **kw)

    return params


# --------------------------------------------------------------------------------------
# Self check.
# --------------------------------------------------------------------------------------

def self_check() -> bool:
    """Assert the oracle's own invariants. Exits nonzero on failure."""
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    torch.manual_seed(0)
    batch, seq, dim, num_loops = 2, 3, 6, 3

    def make_states(requires_grad: bool = False):
        ins, outs = [], []
        for _ in range(num_loops):
            ins.append(torch.randn(batch, seq, dim, dtype=torch.float64,
                                   requires_grad=requires_grad))
            outs.append(torch.randn(batch, seq, dim, dtype=torch.float64,
                                    requires_grad=requires_grad))
        return ins, outs

    # 1. Shape: every mode returns the shape it was given.
    ins, outs = make_states()
    for mode in MODES:
        params = default_params(dim, num_loops, mode)
        y = reference_junction(params, ins, outs, mode)
        check(tuple(y.shape) == (batch, seq, dim),
              f"{mode}: shape {tuple(y.shape)}, expected {(batch, seq, dim)}")
        check(bool(torch.isfinite(y).all()), f"{mode}: non-finite output on random input")

    # 2. alpha == 0 hands back the current absolute state EXACTLY, for every mode.
    for mode in MODES:
        params = default_params(dim, num_loops, mode)
        params["alpha"] = torch.zeros(1, dtype=torch.float64)
        y = reference_junction(params, ins, outs, mode)
        check(torch.equal(y, outs[-1].to(torch.float64)),
              f"{mode}: alpha=0 did not return out_last exactly "
              f"(max |diff| = {float((y - outs[-1]).abs().max()):.3e})")

    # 3. An all-zero delta sequence is finite forward AND backward, for every mode.
    #    Zero deltas drive the eps-clamped normalisation to the origin, which is where a
    #    naive `v / clamp(||v||, eps)` produces NaN gradients.
    for mode in MODES:
        states = [torch.randn(batch, seq, dim, dtype=torch.float64, requires_grad=True)
                  for _ in range(num_loops)]
        zero_ins = list(states)
        zero_outs = list(states)      # out_r is in_r, so every delta is exactly zero
        params = {k: v.clone().requires_grad_(True)
                  for k, v in default_params(dim, num_loops, mode).items()}
        params["alpha"] = torch.full((1,), 0.7, dtype=torch.float64, requires_grad=True)

        delta_max = max(float((o - i).detach().abs().max())
                        for i, o in zip(zero_ins, zero_outs))
        check(delta_max == 0.0, f"{mode}: test setup did not produce zero deltas")

        y = reference_junction(params, zero_ins, zero_outs, mode)
        check(bool(torch.isfinite(y).all()),
              f"{mode}: non-finite output on an all-zero delta sequence")

        y.sum().backward()
        for name, tensor in list(params.items()) + [
                (f"state[{r}]", t) for r, t in enumerate(states)]:
            if tensor.grad is None:
                continue
            check(bool(torch.isfinite(tensor.grad).all()),
                  f"{mode}: non-finite gradient for {name} on an all-zero delta sequence")

    # 3b. Gradients are finite on ordinary input too.
    for mode in MODES:
        g_ins, g_outs = make_states(requires_grad=True)
        params = {k: v.clone().requires_grad_(True)
                  for k, v in default_params(dim, num_loops, mode).items()}
        params["alpha"] = torch.full((1,), 0.3, dtype=torch.float64, requires_grad=True)
        reference_junction(params, g_ins, g_outs, mode).sum().backward()
        for name, tensor in params.items():
            if tensor.grad is not None:
                check(bool(torch.isfinite(tensor.grad).all()),
                      f"{mode}: non-finite gradient for {name} on random input")

    # 4. Reduction identities. The four assertions above constrain shape, the identity
    #    branch and numerical hygiene, but nothing about the transitions themselves: a
    #    mode could scale its state by two and still pass all of them. These pin the
    #    VALUES, using only relations the specification states outright -- "gdn adds a
    #    scalar decay to deltanet", "retnet applies a frozen gamma", "kda is a diagonal
    #    decay applied before the rank-one erase", and so on. Drive each extra mechanism
    #    to its neutral setting and the mode must collapse onto its stated parent.
    big = 40.0          # sigmoid(40) rounds to exactly 1.0 in float64
    unit_eps = 1e-6
    # h_0 is set to zero so that alpha=1 makes y = h_0 + u = u EXACTLY. Recovering the
    # read as `y - h_0` against a generic h_0 would cancel catastrophically: mesa at a
    # large ridge has u ~ 1e-10, and subtracting two O(1) numbers to get it costs six
    # significant digits, which reads as a mesa bug when it is only the probe.
    r_ins = [torch.zeros_like(ins[0])] + list(ins[1:])
    r_outs = list(outs)

    def read(mode: str, fills: dict | None = None) -> Tensor:
        """The read `u` alone: alpha=1 against a zero pre-loop state."""
        params = default_params(dim, num_loops, mode)
        for name, value in (fills or {}).items():
            params[name] = torch.full_like(params[name], value)
        params["alpha"] = torch.ones(1, dtype=torch.float64)
        return reference_junction(params, r_ins, r_outs, mode, eps=unit_eps)

    def same(got: Tensor, want: Tensor, message: str, rtol=1e-9, atol=1e-11) -> None:
        check(torch.allclose(got, want, rtol=rtol, atol=atol),
              f"{message} (max |diff| = {float((got - want).abs().max()):.3e})")

    u_linattn = read("linattn")
    check(float(u_linattn.abs().max()) > 1e-3,
          "reduction baseline is ~zero, so these identities would be vacuous")

    # No decay, unit write scale, no erase => every gated mode IS linattn.
    same(read("retnet", {"gamma": 1.0}), u_linattn,
         "retnet with gamma=1 should equal linattn")
    same(read("gla", {"b_gate": big}), u_linattn,
         "gla with gate=1 should equal linattn")
    same(read("rwkv7", {"b_gate": big, "b_removal": -big}), u_linattn,
         "rwkv7 with gate=1 and zero removal should equal linattn")
    same(read("mamba2", {"A_log": -big, "dt_bias": inv_softplus(1.0)}), u_linattn,
         "mamba2 with dt=1 and zero A should equal linattn")
    same(read("lact", {"b_eta": inv_softplus(1.0)}), u_linattn,
         "lact with eta=1 is one step from an empty memory, i.e. linattn")

    # gdn and kda are deltanet plus a decay; neutralise the decay and they must coincide
    # with deltanet at the same beta.
    u_deltanet = read("deltanet")
    check(float((u_deltanet - u_linattn).abs().max()) > 1e-6,
          "deltanet and linattn coincide, so the gdn/kda identities would be vacuous")
    same(read("gdn", {"b_gate": big}), u_deltanet,
         "gdn with decay=1 should equal deltanet")
    same(read("kda", {"b_gate": big}), u_deltanet,
         "kda with decay=1 should equal deltanet")

    # beta -> 0 erases nothing and writes nothing, so the memory stays empty.
    same(read("deltanet", {"b_beta": -big}),
         torch.zeros_like(u_deltanet), "deltanet with beta=0 should read zero",
         atol=1e-15)

    # mesa at a huge ridge collapses onto a single small gradient step: the solution
    # tends to (1/lambda) * sum_j gamma_j v_j k_j^T, and gamma is sigmoid(0) = 1/2.
    lam = 1e10
    same(read("mesa", {"ridge": lam}) * lam, 0.5 * u_linattn,
         "mesa at a large ridge should collapse onto a scaled linattn read",
         rtol=1e-8, atol=1e-12)

    # With a SINGLE source the ridge solution has a closed form that can be derived by
    # hand, which pins lambda exactly -- including the RIDGE_MIN floor, which the
    # large-ridge check above is blind to. For unit k, the Sherman-Morrison inverse of
    # (gamma k k^T + lambda I) gives
    #     u = gamma / (gamma + lambda) * v * (k . q)
    one_in, one_out = [r_ins[0]], [r_outs[0]]
    p_one = default_params(dim, 1, "mesa")
    p_one["alpha"] = torch.ones(1, dtype=torch.float64)
    got_one = reference_junction(p_one, one_in, one_out, "mesa", eps=unit_eps)
    d1 = (r_outs[0] - r_ins[0]).to(torch.float64)
    k1 = _unit(d1, unit_eps)
    q1 = _unit(r_outs[0].to(torch.float64), unit_eps)
    gamma1, lambda1 = 0.5, RIDGE_INIT
    same(got_one,
         (gamma1 / (gamma1 + lambda1)) * d1 * (k1 * q1).sum(-1, keepdim=True),
         "mesa on a single source should match the closed-form ridge solution")

    # hgrn2's write vector is the complement of its forget gate. Drive the gate to zero:
    # every earlier source is wiped and the last one is written with an all-ones key, so
    # the read is exactly delta_last * sum(q).
    q_last = _unit(r_outs[-1].to(torch.float64), unit_eps)
    delta_last = (r_outs[-1] - r_ins[-1]).to(torch.float64)
    same(read("hgrn2", {"b_gate": -big}), delta_last * q_last.sum(-1, keepdim=True),
         "hgrn2 with forget=0 should read delta_last * sum(q)")

    # 4b. The identities above neutralise each mechanism, so on their own they cannot
    #     tell "gamma == 1" from "gamma is ignored". Pin the mechanisms at settings where
    #     they must BITE, with an analytic target on the other side.

    # gamma = 0 forgets everything: only the last source survives, so the read is
    # delta_last * (k_last . q).
    delta_last = (r_outs[-1] - r_ins[-1]).to(torch.float64)
    q_last = _unit(r_outs[-1].to(torch.float64), unit_eps)
    k_last = _unit(delta_last, unit_eps)
    same(read("retnet", {"gamma": 0.0}),
         delta_last * (k_last * q_last).sum(-1, keepdim=True),
         "retnet with gamma=0 should read only the last source")

    # dt is the write scale as well as the decay: at dt = 2 with A = 0 the memory does
    # not decay, so the read is exactly twice linattn's.
    same(read("mamba2", {"A_log": -big, "dt_bias": inv_softplus(2.0)}), 2.0 * u_linattn,
         "mamba2's dt should scale the write, not only the decay")

    # No mode may silently degenerate into linattn at its own initialisation. (lact is
    # excluded: eta = 1 at init makes one step from an empty memory exactly linattn.)
    for mode in MODES:
        if mode in ("linattn", "lact"):
            continue
        check(float((read(mode) - u_linattn).abs().max()) > 1e-6,
              f"{mode}: reads identically to linattn at init, so its mechanism is inert")

    # ...nor may a mode degenerate into the PARENT it is defined as extending. Comparing
    # only against linattn would let gdn collapse onto deltanet, or rwkv7 onto gla, and
    # still look alive.
    for child, parent, mechanism in (
            ("gdn", "deltanet", "the scalar decay gdn adds to deltanet"),
            ("kda", "deltanet", "the diagonal decay kda applies before the erase"),
            ("rwkv7", "gla", "rwkv7's decoupled rank-one removal"),
            ("hgrn2", "gla", "hgrn2's tie between the write and the gate complement"),
    ):
        check(float((read(child) - read(parent)).abs().max()) > 1e-6,
              f"{child}: reads identically to {parent} at init, so {mechanism} is inert")

    # 4c. The initialiser constants each mode's specification implies.
    softplus = torch.nn.functional.softplus
    p_mamba = default_params(dim, num_loops, "mamba2")
    step_one_decay = float(torch.exp(
        -softplus(p_mamba["dt_bias"][0, 0]) * torch.exp(p_mamba["A_log"][0])))
    check(abs(step_one_decay - SSM_DECAY) < 1e-12,
          f"mamba2 init step-one decay is {step_one_decay:.12f}, expected {SSM_DECAY}")

    eta0 = float(softplus(default_params(dim, num_loops, "lact")["b_eta"][0, 0]))
    check(abs(eta0 - LACT_ETA_INIT) < 1e-12,
          f"lact init eta is {eta0:.12f}, expected {LACT_ETA_INIT}")

    p_mesa = default_params(dim, num_loops, "mesa")
    lam0 = float(softplus(p_mesa["ridge"][0])) + RIDGE_MIN
    check(abs(lam0 - RIDGE_INIT) < 1e-12,
          f"mesa init ridge is {lam0:.12f}, expected {RIDGE_INIT}")
    gamma0 = float(torch.sigmoid(p_mesa["b_gamma"][0, 0]))
    check(abs(gamma0 - 0.5) < 1e-12, f"mesa init gamma is {gamma0}, expected 0.5")

    beta0 = float(torch.sigmoid(default_params(dim, num_loops, "deltanet")["b_beta"][0, 0]))
    check(abs(beta0 - 0.5) < 1e-12, f"deltanet init beta is {beta0}, expected 0.5")

    rho0 = float(torch.sigmoid(default_params(dim, num_loops, "rwkv7")["b_removal"][0, 0, 0]))
    check(abs(rho0 - 0.5) < 1e-12, f"rwkv7 init removal is {rho0}, expected 0.5")

    # 5. The gate bias initialiser retains 0.95 of the oldest source over the whole loop
    #    axis -- across max(L - 1, 1) TRANSITIONS, which is one fewer than the L visits.
    for loops in (2, 3, 4):
        bias = gate_bias_init(loops)
        steps = max(loops - 1, 1)
        retention = (1.0 / (1.0 + math.exp(-bias))) ** steps
        check(abs(retention - GATE_RETENTION_INIT) < 1e-12,
              f"L={loops}: sigmoid({bias:.6f})^{steps} = {retention:.12f}, "
              f"expected {GATE_RETENTION_INIT}")

    if failures:
        print(f"reference_linattn.self_check: {len(failures)} FAILURE(S)",
              file=sys.stderr)
        for message in failures:
            print(f"  - {message}", file=sys.stderr)
        sys.exit(1)

    print("reference_linattn.self_check: OK "
          f"({len(MODES)} modes: shape, alpha=0 identity, zero-delta forward/backward, "
          "reduction identities, gate bias retention)")
    return True


if __name__ == "__main__":
    self_check()
