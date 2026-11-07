"""Depth-memory readers on the loop axis, state interface (paper: tab:residual-families, app:readers).

The model applies its K shared blocks R times; the code's `num_loops` is the iteration count R. At the
boundary of iteration r (the code's "junction") the reader is handed the stored states v_0 .. v_r: h_0 and
the output of every iteration so far before its boundary blend, the last one being the current proposal.
Per head (eight heads in every released configuration, dh = dim // num_heads channels each)

    k_j = L2Norm(v_j),   v_j unprojected,   q = L2Norm(v_r)        (epsilon floor, no Q/K/V projections)
    S_i = S_{i-1} W_i + v_i b_i^T,   S_{-1} = 0,   o_r = S_r q
    h_{(r+1)K} = (1 - zeta) v_r + zeta o_r,     zeta (`alpha`) initialized to 0, i.e. Plain Loop.

S is never materialized: S_r = sum_i v_i w_{r,i}^T with w_{r,i} = W_r^T ... W_{i+1}^T b_i, so only the
write vectors are carried (`_read` applies every transport to them as W^T w) and o_r = sum_i (w_{r,i}.q) v_i.
Transport W_i and write b_i per reader, as in tab:residual-families (i indexes the stored source; the
gates are content-independent learned parameters indexed by junction, head and, where needed, channel):

    linattn   W = I                                b = k
    retnet    W = gamma_h I, gamma_h = 1 - 2^-(5+h) fixed    b = k
    mamba2    W = exp(-tau_i exp(A)) I             b = tau_i k         tau_i = softplus(dt_bias_i)
    gla       W = Diag(g_i)                        b = k
    hgrn2     W = Diag(g_i)                        b = 1 - g_i         the gate complement
    deltanet  W = I - eta_i k k^T                  b = eta_i k
    gdn       W = gamma_i (I - eta_i k k^T)        b = eta_i k         decay before correction
    kda       W = Diag(g_i) (I - eta_i k k^T)      b = eta_i k         decay before correction
    rwkv7     W = Diag(g_i) - kappa_i (a_i * kappa_i)^T    b = k      kappa_i = L2Norm(v_i - v_{i-1})
    mesa      S_r = V Omega K^T (K Omega K^T + mu I)^-1     weighted ridge regression over the prefix
    lact      one gradient step on the weighted prefix error from S_{r-1}, then Frobenius normalization

Parameters (rows index the junction: R + 1 stored sources and R transitions between them; the gates of
mamba2 and hgrn2 also set the write and so have a row per source), activation, initialization:

    alpha      (1,)                      zeta                        0
    b_beta     (R+1, H)                  eta = sigmoid               0.5                 deltanet, gdn, kda
    b_gate     (R, H) gdn | (R, H, dh) gla, kda, rwkv7 | (R+1, H, dh) hgrn2
                                         gamma, g = sigmoid          0.95^(1/R) per step: the oldest source
                                                                     retains 0.95 after R transitions
    A_log (H,), dt_bias (R+1, H)         tau = softplus(dt_bias)     A = 1, exp(-tau) = sqrt(1/5)   mamba2
    b_removal  (R, H, dh)                a = sigmoid                 0.5                 rwkv7
    b_eta      (R+1, H)                  eta = softplus              1                   lact
    b_gamma (R+1, H), ridge (H,)         omega = sigmoid, mu = 1e-4 + softplus(ridge)   0.5, 1   mesa
    gamma      buffer (1, 1, H, 1)       retnet's fixed schedule

Geometry runs in fp32 and the result is cast back to the input dtype.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

LINATTN_KINDS = ("linattn", "retnet", "mamba2", "gla", "hgrn2",
                 "deltanet", "gdn", "kda", "rwkv7", "lact", "mesa")

_USES_DELTA = {"deltanet", "gdn", "kda"}          # key-specific correction I - eta k k^T, write eta k
_ADDITIVE_DPLR = {"rwkv7"}                        # Diag(g) MINUS a rank-one term, not Diag(g) times one
_DECAY = {                                        # None | "fixed" | "scalar" | "diag"
    "linattn": None, "retnet": "fixed", "mamba2": "scalar", "gla": "diag",
    "hgrn2": "diag", "deltanet": None, "gdn": "scalar", "kda": "diag",
    "rwkv7": "diag", "lact": None, "mesa": None,
}
_TIED_WRITE = {"hgrn2"}                           # writes the gate complement 1 - g instead of a key
_DECOUPLED_ERASE = {"rwkv7"}                      # removal direction kappa is not the write key
_GATE_BY_SOURCE = {"mamba2", "hgrn2"}             # the gate also sets the write: one row per source

# Mamba-2 at init: A_log = 0 and dt_bias such that the step-one decay exp(-softplus(dt_bias)) is sqrt(1/5).
_SSM_DECAY = math.sqrt(1.0 / 5.0)
# A gated reader starts close to its ungated form: the gate bias is set so that the oldest source
# retains 0.95 after all R transitions, sigmoid(b)^R = 0.95.
_GATE_RETENTION_INIT = 0.95
# Mamba-2 bounds, each placed where fp32 arithmetic (not the model) would break: the decay exponent
# is floored at -30 so the decay never reaches exactly 0, exp(A_log) is bounded so it cannot be inf
# (clamping the product alone leaves a NaN gradient), and dt is bounded so dt * exp(A_log) cannot
# overflow. None of them binds in a healthy run.
_MAX_LOG_DECAY = 30.0
_A_LOG_MAX = 30.0
_DT_MAX = 1e10
# Mesa's ridge mu = _RIDGE_MIN + softplus(ridge): init 1 (the Gram matrix of unit keys has unit
# diagonal), with a floor so that collinear keys cannot make the n x n system singular in fp32.
_RIDGE_INIT = 1.0
_RIDGE_MIN = 1e-4
# LaCT-inspired step size eta = softplus(b_eta): init 1 (the exact minimizer for one orthogonal unit
# key), bounded so that eta v k^T cannot overflow before the Frobenius normalization.
_LACT_ETA_INIT = 1.0
_LACT_ETA_MAX = 1e6
# epsilon of the Frobenius normalization, nu = (||S||_F^2 + eps^2)^(-1/2); S is zero before the first write.
_LACT_NORM_EPS = 1e-6


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) == y."""
    return math.log(math.expm1(y))


def _gate_bias_init(num_loops: int) -> float:
    """logit(rho^(1/R)): the per-transition gate bias whose retention over R transitions is rho."""
    per_step = _GATE_RETENTION_INIT ** (1.0 / max(int(num_loops), 1))
    return math.log(per_step / (1.0 - per_step))


class LoopLinAttn(nn.Module):
    """One reader of LINATTN_KINDS over the stored states; see the module docstring.

    Args:
      dim: hidden width.  num_loops: the iteration count R.  kind: one of LINATTN_KINDS.
      num_heads: reader heads (8 in every released configuration).  eps: normalization floor.

    combine([v_0 .. v_r]) returns the blended state h_{(r+1)K} in the dtype of v_r.
    """

    def __init__(self, dim: int, num_loops: int, kind: str, num_heads: int = 8, eps: float = 1e-6):
        super().__init__()
        if kind not in LINATTN_KINDS:
            raise ValueError(f"kind must be one of {LINATTN_KINDS}, got {kind!r}")
        if dim % num_heads != 0:
            raise ValueError(f"model_dim ({dim}) must be divisible by num_heads ({num_heads})")
        if not (eps > 0 and math.isfinite(eps)):
            # at 0 an all-zero state gives 0/0; at inf every key normalizes to zero and the read is 0
            raise ValueError(f"eps must be positive and finite, got {eps!r}")
        if num_loops < 1:
            raise ValueError(f"num_loops must be at least 1, got {num_loops!r}")
        self.kind = kind
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_loops = num_loops
        self.eps = eps
        self.uses_delta = kind in _USES_DELTA
        self.decay = _DECAY[kind]
        self.tied_write = kind in _TIED_WRITE
        self.decoupled_erase = kind in _DECOUPLED_ERASE
        self.additive_dplr = kind in _ADDITIVE_DPLR
        self.gate_by_source = kind in _GATE_BY_SOURCE

        # Rows index the junction. The deepest junction sees R + 1 stored sources (h_0 and R iteration
        # outputs) and applies R transitions between them: write-side parameters have one row per
        # source, decay gates one per transition (none is applied to the first source, which is
        # written into an empty memory), except for mamba2 and hgrn2, whose gate also sets the write.
        # Rows are clamped at use, so a longer history reuses the last row.
        rows = num_loops + 1
        grows = rows if self.gate_by_source else num_loops
        H, dh = num_heads, self.head_dim

        if self.uses_delta:
            self.b_beta = nn.Parameter(torch.zeros(rows, H))                 # eta = sigmoid(b_beta)

        if self.decay == "fixed":
            # RetNet's per-head schedule gamma_h = 1 - 2^-(5+h), computed in float64; a head whose gamma
            # rounds to 1 in fp32 (h >= 18) would not decay at all, so it is refused. The buffer is
            # created empty and written by reset_parameters(): the trainer constructs the model on
            # the meta device and materializes it afterwards.
            g64 = 1.0 - torch.pow(2.0, -5.0 - torch.arange(H, dtype=torch.float64, device="cpu"))
            if bool((g64.to(torch.float32) >= 1.0).any()):
                raise ValueError(f"retnet: gamma_h = 1 - 2^-(5+h) rounds to 1 in fp32 for some head at "
                                 f"num_heads={H}; use fewer heads")
            self.register_buffer("gamma", torch.empty(1, 1, H, 1))
        elif kind == "mamba2":
            self.A_log = nn.Parameter(torch.zeros(H))
            self.dt_bias = nn.Parameter(
                torch.full((rows, H), _inv_softplus(-math.log(_SSM_DECAY))))
        elif self.decay == "scalar":
            self.b_gate = nn.Parameter(torch.full((grows, H), _gate_bias_init(num_loops)))
        elif self.decay == "diag":
            self.b_gate = nn.Parameter(torch.full((grows, H, dh), _gate_bias_init(num_loops)))

        if kind == "lact":
            self.b_eta = nn.Parameter(
                torch.full((rows, H), _inv_softplus(_LACT_ETA_INIT)))         # eta = softplus(b_eta)

        if self.additive_dplr:
            self.b_removal = nn.Parameter(torch.zeros(grows, H, dh))         # a = sigmoid(b_removal)

        if kind == "mesa":
            self.b_gamma = nn.Parameter(torch.zeros(rows, H))                # omega = sigmoid(b_gamma)
            self.ridge = nn.Parameter(
                torch.full((H,), _inv_softplus(_RIDGE_INIT - _RIDGE_MIN)))   # mu = 1e-4 + softplus(ridge)

        self.alpha = nn.Parameter(torch.zeros(1))                            # zeta: 0 => Plain Loop
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.alpha.zero_()
            if self.decay == "fixed":
                # written here rather than in __init__ because the trainer materializes the model from
                # the meta device and then calls reset_parameters(); skipped while still on meta
                H = self.num_heads
                src = (1.0 - torch.pow(2.0, -5.0 - torch.arange(H, dtype=torch.float64, device="cpu")))\
                    .to(torch.float32).view(1, 1, H, 1)
                if self.gamma.device.type != "meta":
                    self.gamma.copy_(src.to(self.gamma.device))
            if self.uses_delta:
                self.b_beta.zero_()
            if self.kind == "mamba2":
                self.A_log.zero_()
                self.dt_bias.fill_(_inv_softplus(-math.log(_SSM_DECAY)))
            if self.additive_dplr:
                self.b_removal.zero_()
            if self.kind == "lact":
                self.b_eta.fill_(_inv_softplus(_LACT_ETA_INIT))
            if self.kind == "mesa":
                self.b_gamma.zero_()
                self.ridge.fill_(_inv_softplus(_RIDGE_INIT - _RIDGE_MIN))
            if hasattr(self, "b_gate"):
                self.b_gate.fill_(_gate_bias_init(self.num_loops))

    # ---- shape helpers -------------------------------------------------------------

    def _split(self, t: Tensor) -> Tensor:
        b, s, d = t.shape
        return t.view(b, s, self.num_heads, self.head_dim)

    def _l2(self, t: Tensor) -> Tensor:
        # v / max(||v||, eps): exactly unit norm above the floor, so I - k k^T is a projection
        return F.normalize(t, dim=-1, eps=self.eps)

    def _row(self, p: Tensor, j: int) -> Tensor:
        """Row j of a per-junction parameter, clamped to the last row, broadcast over (batch, position)."""
        r = p[min(j, p.shape[0] - 1)]
        return r.view(1, 1, *r.shape)

    # ---- per-source quantities -----------------------------------------------------

    def _head_scalar(self, param: Tensor, j: int) -> Tensor:
        """Row j of a per-(junction, head) parameter as (1, 1, H, 1) logits."""
        return self._row(param, j).unsqueeze(-1)

    def _beta(self, j: int) -> Tensor:
        """eta_j in (0, 1): correction strength and write scale of the delta rule, (1, 1, H, 1)."""
        return torch.sigmoid(self._head_scalar(self.b_beta, j))

    def _dt(self, j: int) -> Tensor:
        """tau_j > 0 of Mamba-2, bounded; it sets both the decay and the write scale, (1, 1, H, 1)."""
        return F.softplus(self._head_scalar(self.dt_bias, j)).clamp(max=_DT_MAX)

    def _gate(self, j: int, dt=None):
        """The decay applied to the carried write vectors: (1, 1, H, 1) scalar or (1, 1, H, dh) diagonal."""
        if self.decay is None:
            return None
        if self.decay == "fixed":
            return self.gamma.float()
        if self.kind == "mamba2":
            dt = self._dt(j) if dt is None else dt
            A = torch.exp(self.A_log.float().clamp(max=_A_LOG_MAX)).view(1, 1, -1, 1)
            log_decay = -(dt * A)
            return torch.exp(log_decay.clamp(min=-_MAX_LOG_DECAY))
        # j indexes the source; the gate normally sits on the transition INTO it (row j - 1), except
        # for mamba2 and hgrn2, whose gate also sets the write and is read at j = 0 as well.
        t = j if self.gate_by_source else max(j - 1, 0)
        if self.decay == "scalar":
            return torch.sigmoid(self._head_scalar(self.b_gate, t))
        return torch.sigmoid(self._row(self.b_gate, t))

    def _write_vec(self, j: int, k: Tensor, beta=None, dt=None, gate=None) -> Tensor:
        """The write vector b_j, (.., H, dh): eta k (delta rule), tau k (mamba2), 1 - g (hgrn2), else k."""
        if self.tied_write:
            return 1.0 - gate
        if self.uses_delta:
            return (self._beta(j) if beta is None else beta) * k
        if self.kind == "mamba2":
            return (self._dt(j) if dt is None else dt) * k
        return k

    def _removal(self, j: int) -> Tensor:
        """a_j in (0, 1): RWKV-7's per-channel removal strength on the transition into source j."""
        return torch.sigmoid(self._row(self.b_removal, max(j - 1, 0)))

    def _erase_dir(self, hfs: list[Tensor], ks: list[Tensor], r: int) -> Tensor:
        """The direction the rank-one correction acts along, (.., H, dh): the write key, except for
        rwkv7, whose kappa_r is the normalized difference of adjacent stored states."""
        if not self.decoupled_erase:
            return ks[r]
        return self._l2(self._split(hfs[r] - hfs[r - 1]))

    def _read_lact(self, states: list[Tensor]) -> Tensor:
        """LaCT-inspired reader (app:regression): at junction r one gradient step from S_{r-1} on the
        prefix error sum_{j<=r} eta_j/2 ||S k_j - v_j||^2, then S <- S / sqrt(||S||_F^2 + eps^2).

        Factored over the carried write vectors w_i (every i <= r) and the keys:
            w_i <- w_i - sum_{j<=r} eta_j (w_i . k_j) k_j,     w_j <- w_j + eta_j k_j,
        and ||S||_F^2 = sum_{i,i'} (v_i . v_i')(w_i . w_i') is an (r+1) x (r+1) contraction.
        """
        n = len(states)
        hfs = [h.float() for h in states]
        K = torch.stack([self._l2(self._split(h)) for h in hfs], 0)   # (n,b,t,H,dh)
        V = torch.stack([self._split(h) for h in hfs], 0)
        q = self._l2(self._split(hfs[-1]))

        eta = torch.stack(
            [F.softplus(self._head_scalar(self.b_eta, j)).clamp(max=_LACT_ETA_MAX)
             for j in range(n)],
            dim=0).expand(n, *K.shape[1:-1], 1)                       # (n,b,t,H,1)
        E = eta.squeeze(-1).movedim(0, -1)                            # (b,t,H,n)

        W = None
        for r in range(n):
            m = r + 1
            new = torch.zeros_like(K[r]).unsqueeze(0)
            W = new if W is None else torch.cat([W, new], dim=0)      # (m,b,t,H,dh)
            Km, Vm = K[:m], V[:m]

            proj = torch.einsum("i...d,j...d->...ij", W, Km)          # (b,t,H,m,m)
            upd = torch.einsum("...ij,j...d->i...d",
                               proj * E[..., None, :m], Km)
            W = W - upd + eta[:m] * Km

            gv = torch.einsum("i...d,j...d->...ij", Vm, Vm)
            gw = torch.einsum("i...d,j...d->...ij", W, W)
            # eps inside the sqrt: d sqrt(x)/dx is infinite at 0, which an all-zero history reaches
            nrm2 = (gv * gw).sum((-1, -2)).clamp(min=0.0) + _LACT_NORM_EPS ** 2
            W = W / nrm2.sqrt().unsqueeze(0).unsqueeze(-1)

        return ((W * q.unsqueeze(0)).sum(-1, keepdim=True) * V).sum(0).reshape(
            states[-1].shape)

    def _read_mesa(self, states: list[Tensor]) -> Tensor:
        """Mesa (eq. ridge): S_r = V Omega K^T (K Omega K^T + mu I)^-1, evaluated as the equivalent
        n x n weighted Gram solve, n = r + 1, so no dh x dh matrix is formed:
            c = sqrt(Omega) (mu I + B B^T)^-1 B q,   B = sqrt(Omega) K,   o_r = sum_j c_j v_j.
        sqrt(omega) is built as exp(logsigmoid / 2) so a saturated omega keeps a finite gradient.
        """
        hfs = [h.float() for h in states]
        K = torch.stack([self._l2(self._split(h)) for h in hfs], dim=-2)  # (b,t,H,n,dh)
        V = torch.stack([self._split(h) for h in hfs], dim=-2)
        q = self._l2(self._split(hfs[-1]))                                # (b,t,H,dh)

        logit = torch.stack(
            [self._head_scalar(self.b_gamma, j).squeeze(-1)
             for j in range(len(states))], dim=-1)                        # (..,H,n)
        sq_gam = torch.exp(0.5 * F.logsigmoid(logit)).expand(*V.shape[:-1])   # sqrt(omega)
        lam = (_RIDGE_MIN + F.softplus(self.ridge.float())).view(1, 1, -1, 1, 1)

        sq = sq_gam.unsqueeze(-1)                                         # (..,H,n,1)
        B = sq * K                                                        # (b,t,H,n,dh)
        M = B @ B.transpose(-1, -2) + lam * torch.eye(
            len(states), dtype=K.dtype, device=K.device)
        rhs = (B * q.unsqueeze(-2)).sum(-1, keepdim=True)                 # (b,t,H,n,1)
        c = sq * torch.linalg.solve(M, rhs)
        return (c * V).sum(-2).reshape(states[-1].shape)

    # ---- the kernel ----------------------------------------------------------------

    def _read(self, states: list[Tensor]) -> Tensor:
        """o_r = S_r q with S_r held factored over the stored states. Returns fp32, shape of states[-1]."""
        if self.kind == "mesa":
            return self._read_mesa(states)
        if self.kind == "lact":
            return self._read_lact(states)
        n = len(states)
        hfs = [h.float() for h in states]
        ks = [self._l2(self._split(h)) for h in hfs]
        V = torch.stack([self._split(h) for h in hfs], dim=0)      # (n, b, t, H, dh)
        q = self._l2(self._split(hfs[-1]))                         # (b, t, H, dh)

        W = None                                                   # carried write vectors (m, b, t, H, dh)
        for r in range(n):
            beta = self._beta(r) if self.uses_delta else None
            dt = self._dt(r) if self.kind == "mamba2" else None
            # hgrn2 needs its gate at r = 0 as well, because the gate complement is the write vector
            g = (self._gate(r, dt=dt)
                 if self.decay is not None and (W is not None or self.tied_write)
                 else None)
            if W is not None:
                if self.additive_dplr:
                    # W_r = Diag(g) - kappa (a * kappa)^T, so W_r^T w = g * w - (a * kappa) (kappa . w)
                    ed = self._erase_dir(hfs, ks, r)
                    er = ed.unsqueeze(0)
                    proj = (W * er).sum(-1, keepdim=True)
                    W = (g.unsqueeze(0) * W
                         - (self._removal(r) * ed).unsqueeze(0) * proj)
                else:
                    # W_r = D (I - eta k k^T): decay first, then the correction. The order matters only
                    # for a diagonal D (kda); for a scalar one (gdn, mamba2) the two commute.
                    if g is not None:
                        W = g.unsqueeze(0) * W
                    if beta is not None:
                        er = self._erase_dir(hfs, ks, r).unsqueeze(0)
                        proj = (W * er).sum(-1, keepdim=True)
                        W = W - beta.unsqueeze(0) * proj * er
            new = self._write_vec(r, ks[r], beta=beta, dt=dt, gate=g).unsqueeze(0)
            W = new if W is None else torch.cat([W, new], dim=0)

        coef = (W * q.unsqueeze(0)).sum(-1, keepdim=True)          # (n, b, t, H, 1)
        return (coef * V).sum(0).reshape(states[-1].shape)

    def combine(self, states: list[Tensor]) -> Tensor:
        """states = [v_0 .. v_r]: h_0 and every iteration output so far before its boundary blend, the
        last being the current proposal. Every state is written, including the current one, and the
        current one is the query. Returns (1 - zeta) v_r + zeta o_r in the dtype of v_r."""
        if not states:
            raise ValueError("combine() needs at least one state")
        x = states[-1]
        u = self._read(states)
        alpha = self.alpha.float()
        return ((1.0 - alpha) * x.float() + alpha * u).to(x.dtype)
