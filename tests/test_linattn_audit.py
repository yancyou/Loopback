"""Audit of the two reader files against the paper's materialised form.

The reference here is written from the paper's equations (tab:residual-families, app:operator-mapping,
app:readers): S_j = S_{j-1} W_j + v_j b_j^T in value-by-key orientation and o = S q, with S literally
materialised in float64. It reads ONLY the module's parameter tensors (state_dict) and never calls the
module's gate helpers, so a wrong row, wrong activation, wrong sign or wrong decay/correction order
moves the reference and the implementation apart instead of together.

    python tests/test_linattn_audit.py --interface state   # loop_linattn_state.py: 8 heads over the stored states
    python tests/test_linattn_audit.py --interface delta   # loop_linattn_delta.py: 1 head over whole-iteration updates
"""

import argparse
import importlib
import math
import os
import sys

import torch

# --------------------------------------------------------------------------------
# interface selection (must happen before importing the reader module)
# --------------------------------------------------------------------------------
_ap = argparse.ArgumentParser()
_ap.add_argument("--interface", choices=("state", "delta"), default="state",
                 help="which reader file to audit")
_ap.add_argument("--max-sources", type=int, default=12,
                 help="deepest history audited, in stored sources")
ARGS = _ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LL = importlib.import_module("loop_linattn_state" if ARGS.interface == "state" else "loop_linattn_delta")
LoopLinAttn = LL.LoopLinAttn
KINDS = tuple(LL.LINATTN_KINDS)
UPDATE_IFACE = ARGS.interface == "delta"
VARIANT = ("update interface (whole-iteration updates, one head)" if UPDATE_IFACE
           else "state interface (stored states, 8 heads)")

torch.set_default_dtype(torch.float32)
B, T, D = 2, 5, 64
H = 1 if UPDATE_IFACE else 8
MAXN = ARGS.max_sources
# rows: the state interface stores R + 1 sources (h_0 .. h_R); the update interface stores R updates.
LOOPS = MAXN if UPDATE_IFACE else MAXN - 1
DELTA_KINDS = {"deltanet", "gdn", "kda"}
LACT_EPS = LL._LACT_NORM_EPS
RIDGE_MIN = LL._RIDGE_MIN


def make(kind, loops=LOOPS, heads=H, dim=D, seed=0):
    torch.manual_seed(seed)
    return LoopLinAttn(dim, loops, kind=kind, num_heads=heads)


def randomise_(mod, seed=2):
    """Every parameter off its init (gates NOT at init), rows ~ N(0, 0.7)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in mod.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.7)
    return mod


def rand_states(n, scale=1.0, seed=1, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(B, T, D, generator=g) * scale).to(dtype) for _ in range(n)]


def _unit(x, eps):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


# --------------------------------------------------------------------------------
# The paper's reader, materialised, from parameters only
# --------------------------------------------------------------------------------
class PaperReference:
    """S_j = S_{j-1} A_j + c_j v_j k_j^T (value-by-key), o = S q, in float64.

    Gate parameterisation re-derived by hand from the parameter tensors:
      beta_j   = sigmoid(b_beta[j])                            rows by SOURCE
      dt_j     = softplus(dt_bias[j]),  a_j = exp(-dt_j e^{A_log})      (mamba2)
      a_j      = sigmoid(b_gate[j-1])                          rows by TRANSITION (gla/gdn/kda/rwkv7)
      a_j      = sigmoid(b_gate[j])  and write = 1 - a_j       (hgrn2, rows by SOURCE)
      gamma_h  = 1 - 2^-(5+h) frozen buffer                    (retnet)
      rho_j    = sigmoid(b_removal[j-1])                       (rwkv7, per channel)
    Rows beyond the constructed depth clamp to the last row (as the code documents).
    """

    def __init__(self, mod, dtype=torch.float64):
        self.P = {k: v.detach().to(dtype) for k, v in mod.state_dict().items()}
        self.kind, self.H, self.dh, self.eps = mod.kind, mod.num_heads, mod.head_dim, mod.eps
        self.dtype = dtype

    def split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.H, self.dh)

    def row(self, name, j):
        p = self.P[name]
        return p[min(j, p.shape[0] - 1)]

    def scalar_gate(self, pname, j):                            # (1,1,H,1) logits
        return self.row(pname, j).view(1, 1, self.H, 1)

    def diag_gate(self, pname, j):                              # (1,1,H,dh) logits
        return self.row(pname, j).view(1, 1, self.H, self.dh)

    def kqv(self, sources, query):
        ks = [_unit(self.split(s.to(self.dtype)), self.eps) for s in sources]
        vs = [self.split(s.to(self.dtype)) for s in sources]
        q = _unit(self.split(query.to(self.dtype)), self.eps)
        return ks, vs, q

    # per-source quantities, by hand ------------------------------------------------
    def write_scalar_and_decay(self, j):
        """returns (c_j, a_j as (1,1,H,dh) or None, beta_j or None)"""
        kind = self.kind
        one = torch.ones(1, 1, self.H, 1, dtype=self.dtype)
        beta = torch.sigmoid(self.scalar_gate("b_beta", j)) if kind in DELTA_KINDS else None
        if kind == "mamba2":
            dt = torch.nn.functional.softplus(self.scalar_gate("dt_bias", j))
            a = torch.exp(-dt * torch.exp(self.P["A_log"]).view(1, 1, self.H, 1))
            return dt, a.expand(*a.shape[:-1], self.dh), None
        if kind == "retnet":
            a = self.P["gamma"].view(1, 1, self.H, 1)
            return one, a.expand(1, 1, self.H, self.dh), None
        if kind == "gdn":
            a = torch.sigmoid(self.scalar_gate("b_gate", j - 1))
            return beta, a.expand(*a.shape[:-1], self.dh), beta
        if kind in ("gla", "kda", "rwkv7"):
            a = torch.sigmoid(self.diag_gate("b_gate", j - 1))
            return (beta if beta is not None else one), a, beta
        if kind == "hgrn2":
            a = torch.sigmoid(self.diag_gate("b_gate", j))
            return None, a, None
        if kind == "deltanet":
            return beta, None, beta
        assert kind == "linattn", kind
        return one, None, None

    def dplr_read(self, sources, query, erase_dirs):
        ks, vs, q = self.kqv(sources, query)
        b, t = query.shape[:2]
        I = torch.eye(self.dh, dtype=self.dtype).view(1, 1, 1, self.dh, self.dh)
        S = torch.zeros(b, t, self.H, self.dh, self.dh, dtype=self.dtype)
        for j in range(len(sources)):
            c, a, beta = self.write_scalar_and_decay(j)
            k, v = ks[j], vs[j]
            if j > 0:
                if self.kind == "rwkv7":
                    rho = torch.sigmoid(self.diag_gate("b_removal", j - 1))
                    e = erase_dirs[j]
                    # A = Diag(a) - e (rho * e)^T   (tab:residual-families: Diag(g) - kappa (a * kappa)^T)
                    A = a.unsqueeze(-1) * I - e.unsqueeze(-1) * (rho * e).unsqueeze(-2)
                elif self.kind == "hgrn2":
                    A = a.unsqueeze(-1) * I
                else:
                    E = I
                    if beta is not None:
                        E = I - beta.unsqueeze(-1) * (k.unsqueeze(-1) * k.unsqueeze(-2))
                    A = (a.unsqueeze(-1) * E) if a is not None else E   # Diag(a) (I - beta k k^T)
                S = S @ A
            if self.kind == "hgrn2":
                S = S + v.unsqueeze(-1) * (1.0 - a).unsqueeze(-2)       # v (1 - a)^T
            else:
                S = S + c.unsqueeze(-1) * (v.unsqueeze(-1) * k.unsqueeze(-2))   # c v k^T
        return (S @ q.unsqueeze(-1)).squeeze(-1).reshape(query.shape)

    def lact_read(self, sources, query, normalise=True):
        """W <- W - sum_{j<=r} eta_j (W k_j - v_j) k_j^T  (cumulative prefix), then
        W <- W / sqrt(||W||_F^2 + eps^2) (eq. lact-prefix-map). normalise=False is the
        same step without the normalisation."""
        ks, vs, q = self.kqv(sources, query)
        b, t = query.shape[:2]
        W = torch.zeros(b, t, self.H, self.dh, self.dh, dtype=self.dtype)
        eta = [torch.nn.functional.softplus(self.scalar_gate("b_eta", j))
               for j in range(len(sources))]
        for r in range(len(sources)):
            grad = torch.zeros_like(W)
            for j in range(r + 1):
                resid = (W @ ks[j].unsqueeze(-1)).squeeze(-1) - vs[j]
                grad = grad + eta[j].unsqueeze(-1) * (resid.unsqueeze(-1) * ks[j].unsqueeze(-2))
            W = W - grad
            if normalise:
                W = W / torch.sqrt(W.flatten(-2).pow(2).sum(-1) + LACT_EPS ** 2)[..., None, None]
        return (W @ q.unsqueeze(-1)).squeeze(-1).reshape(query.shape)

    def mesa_read(self, sources, query):
        """S* = argmin sum_j gamma_j ||S k_j - v_j||^2 + lambda ||S||_F^2, materialised:
        S* = (sum_j gamma_j v_j k_j^T)(sum_j gamma_j k_j k_j^T + lambda I)^-1 (eq. ridge)."""
        ks, vs, q = self.kqv(sources, query)
        gam = torch.stack([torch.sigmoid(self.scalar_gate("b_gamma", j))
                           for j in range(len(sources))], dim=-2)             # (1,1,H,n,1)
        gam = gam.expand(*gam.shape[:-2], len(sources), 1)
        lam = (RIDGE_MIN + torch.nn.functional.softplus(self.P["ridge"])).view(1, 1, self.H, 1, 1)
        K = torch.stack(ks, dim=-2)
        V = torch.stack(vs, dim=-2)                                            # (b,t,H,n,dh)
        VGK = V.transpose(-1, -2) @ (gam * K)                                  # sum gamma v k^T
        KGK = K.transpose(-1, -2) @ (gam * K) + lam * torch.eye(self.dh, dtype=self.dtype)
        S = torch.linalg.solve(KGK, VGK.transpose(-1, -2)).transpose(-1, -2)  # VGK KGK^-1
        return (S @ q.unsqueeze(-1)).squeeze(-1).reshape(query.shape)

    def read(self, sources, query, erase_dirs=None):
        if self.kind == "lact":
            return self.lact_read(sources, query)
        if self.kind == "mesa":
            return self.mesa_read(sources, query)
        return self.dplr_read(sources, query, erase_dirs)


# --------------------------------------------------------------------------------
# Adapters: how each interface is called and what its sources are
# --------------------------------------------------------------------------------
def make_history(n, scale, seed, dtype=torch.float32):
    """Returns (call_kwargs, sources, query, x_cur, h0)."""
    if UPDATE_IFACE:
        ins = rand_states(n, scale, seed, dtype)
        outs = rand_states(n, scale, seed + 1000, dtype)
        sources = [(o.float() - i.float()) for i, o in zip(ins, outs)]
        return dict(states=outs, inputs=ins), sources, outs[-1].float(), outs[-1], ins[0]
    st = rand_states(n, scale, seed, dtype)
    sources = [s.float() for s in st]
    return dict(states=st), sources, st[-1].float(), st[-1], None


def erase_dirs_for(mod, sources):
    """RWKV-7's removal direction kappa as the paper defines it for each interface."""
    eps = mod.eps
    ks = [_unit(sources[j].to(torch.float64).view(B, T, mod.num_heads, mod.head_dim), eps)
          for j in range(len(sources))]
    if UPDATE_IFACE:
        return ks                                          # the normalised update, k_r itself
    dirs = [None]
    for j in range(1, len(sources)):
        d = (sources[j] - sources[j - 1]).to(torch.float64).view(B, T, mod.num_heads, mod.head_dim)
        dirs.append(_unit(d, eps))                        # normalised difference of adjacent states
    return dirs


def code_read(mod, kw):
    """The code's factorised o = S q (fp32), extracted through combine() with alpha=1."""
    with torch.no_grad():
        a0 = mod.alpha.clone()
        mod.alpha.fill_(1.0)
        y = mod.combine(**kw).float()
        mod.alpha.copy_(a0)
    if UPDATE_IFACE:                                      # y = h_0 + o  at alpha = 1
        return y - kw["inputs"][0].float()
    return y


# --------------------------------------------------------------------------------
# 1. naive (paper) vs factorised (code)
# --------------------------------------------------------------------------------
def audit_naive_vs_factorised():
    print(f"\n== [1] paper-form materialised S (float64) vs code's factorised read (fp32) "
          f"-- {VARIANT}, B={B} T={T} d={D} H={H}, n = 1..{MAXN} sources ==")
    print("   criterion: max|code - paper| < 1e-5 on unit-scale histories; at stream scale 300 the "
          "error is reported per unit of source scale (fp32 eps ~ 1e-7)")
    results = {}
    for kind in KINDS:
        mod = randomise_(make(kind))
        ref = PaperReference(mod)
        w_abs1 = w_per300 = 0.0
        for n in range(1, MAXN + 1):
            for scale in (1.0, 300.0):
                kw, sources, query, _, _ = make_history(n, scale, seed=10 + n)
                got = code_read(mod, kw)
                want = ref.read(sources, query, erase_dirs_for(mod, sources))
                err = float((got.double() - want).abs().max())
                if scale == 1.0:
                    w_abs1 = max(w_abs1, err)
                else:
                    w_per300 = max(w_per300, err / scale)
                assert torch.isfinite(got).all(), f"{kind}: non-finite read"
        results[kind] = (w_abs1, w_per300)
        ok = w_abs1 < 1e-5 and w_per300 < 1e-5
        print(f"  {kind:9s} abs@1 {w_abs1:.2e} err/scale@300 {w_per300:.2e}  {'PASS' if ok else 'FAIL'}")
    return results


# --------------------------------------------------------------------------------
# 2. init identity (gates at init, alpha = 0) -- bitwise
# --------------------------------------------------------------------------------
def audit_init_identity():
    print("\n== [2] init-equivalence: fresh module, alpha=0, output must equal the current "
          "state bitwise (fp32 + bf16) ==")
    res = {}
    for kind in KINDS:
        ok = True
        mod = make(kind)
        assert float(mod.alpha.detach()) == 0.0
        for dtype in (torch.float32, torch.bfloat16):
            for n in (1, 2, 4, MAXN):
                kw, _, _, x, _ = make_history(n, 300.0, seed=n, dtype=dtype)
                with torch.no_grad():
                    y = mod.combine(**kw)
                itype = torch.int32 if dtype == torch.float32 else torch.int16
                bit = torch.equal(y.view(itype), x.view(itype))
                ok = ok and y.dtype == dtype and bit
        res[kind] = ok
        print(f"  {kind:9s} {'PASS (bitwise)' if ok else 'FAIL'}")
    return res


# --------------------------------------------------------------------------------
# 3. newest write / self term, gate init values
# --------------------------------------------------------------------------------
def audit_newest_write_and_inits():
    print("\n== [3] newest write: coefficient c_r on the newest source and <k_r, q_r>; "
          "gate values AT INIT (R=3) ==")
    L3 = 3
    gb = LL._gate_bias_init(L3)
    steps = "R-1=2" if UPDATE_IFACE else "R=3"
    print(f"  _gate_bias_init(3) = {gb:.4f}  -> sigmoid = {torch.sigmoid(torch.tensor(gb)):.4f} "
          f"per transition (retention 0.95 over {steps} transitions)")
    out = {}
    for kind in KINDS:
        mod = make(kind, loops=L3)
        ref = PaperReference(mod)
        c, a, beta = (None, None, None)
        if kind not in ("lact", "mesa"):
            c, a, beta = ref.write_scalar_and_decay(1)
        # single source: o = c_0 <k_0, q> v_0 ; measure <k_r, q_r> on a generic history
        n = 3 if UPDATE_IFACE else 4
        kw, sources, query, _, _ = make_history(n, 1.0, seed=7)
        kq = float((_unit(sources[-1].double().view(B, T, mod.num_heads, mod.head_dim), mod.eps)
                    * _unit(query.double().view(B, T, mod.num_heads, mod.head_dim), mod.eps)).sum(-1).mean())
        desc = []
        if c is not None:
            desc.append(f"c={float(c.flatten()[0]):.4f}")
        if a is not None:
            desc.append(f"a={float(a.flatten()[0]):.4f}")
        if kind == "hgrn2":
            desc.append(f"write=1-a={1 - float(a.flatten()[0]):.4f}")
        if kind == "rwkv7":
            desc.append(f"rho={float(torch.sigmoid(mod.b_removal[0, 0, 0])):.4f}")
        if kind == "mesa":
            desc.append(f"gamma={float(torch.sigmoid(mod.b_gamma[0, 0])):.4f} "
                        f"lambda={RIDGE_MIN + float(torch.nn.functional.softplus(mod.ridge[0])):.4f}")
        if kind == "lact":
            desc.append(f"eta={float(torch.nn.functional.softplus(mod.b_eta[0, 0])):.4f}")
        out[kind] = (desc, kq)
        print(f"  {kind:9s} init: {' '.join(desc):48s} mean <k_newest, q> = {kq:+.3f}")
    return out


# --------------------------------------------------------------------------------
# 4. what the memory holds
# --------------------------------------------------------------------------------
def audit_sources():
    print("\n== [4] what the memory holds ==")
    if not UPDATE_IFACE:
        print("  state interface: combine(states) reads states[-1] as query and all of "
              "[h_0 .. h_r] as sources (checked in [1] by construction: the reference's "
              "sources ARE the states).")
        return {"variant": "state"}
    ok_all = True
    for kind in KINDS:
        mod = randomise_(make(kind))
        n = 6
        ins = rand_states(n, 1.0, seed=3)
        outs = rand_states(n, 1.0, seed=4)
        with torch.no_grad():
            mod.alpha.fill_(0.6)
            y = mod.combine(states=outs, inputs=ins).double()
            # (a) shift every (in_r, out_r) pair r < n-1 by a common vector: updates and
            #     query unchanged -> read unchanged (the gates do not see the state)
            sh = rand_states(n, 5.0, seed=9)
            ins2 = [i + s for i, s in zip(ins[:-1], sh[:-1])] + [ins[-1]]
            outs2 = [o + s for o, s in zip(outs[:-1], sh[:-1])] + [outs[-1]]
            y2 = mod.combine(states=outs2, inputs=ins2).double()
            # y differs only through h_0 = inputs[0] (added back outside the memory)
            shift_term = 0.6 * (ins2[0] - ins[0]).double()
            inv = float((y2 - shift_term - y).abs().max())
            # (b) paper form: y = (1-a) out_last + a (in_0 + o), o over delta_j = out_j - in_j
            deltas = [o.float() - i.float() for i, o in zip(ins, outs)]
            ref = PaperReference(mod)
            o = ref.read(deltas, outs[-1].float(), erase_dirs_for(mod, deltas))
            want = 0.4 * outs[-1].double() + 0.6 * (ins[0].double() + o)
            form = float((y - want).abs().max())
        ok = inv < 1e-4 and form < 1e-4
        ok_all = ok_all and ok
        print(f"  {kind:9s} shift-invariance of the read {inv:.1e} | paper blend form {form:.1e}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("  => sources are delta_r = out_r - in_r (whole-iteration updates); h_0 is added back "
          "outside the memory with coefficient alpha; the query is L2norm(out_last).")
    return {"variant": "update", "ok": ok_all}


# --------------------------------------------------------------------------------
# 5. parameter counts at d=768, R=3 against the closed-form layout
# --------------------------------------------------------------------------------
def expected_params(kind, R, heads, dim):
    """The parameter count the layout in the module docstring implies."""
    dh = dim // heads
    rows = R if UPDATE_IFACE else R + 1                   # one row per stored source
    grows = max(R - 1, 1) if UPDATE_IFACE else R          # one row per transition
    return {
        "linattn": 1, "retnet": 1,
        "mamba2": 1 + heads + rows * heads,
        "gla": 1 + grows * heads * dh,
        "hgrn2": 1 + rows * heads * dh,
        "deltanet": 1 + rows * heads,
        "gdn": 1 + rows * heads + grows * heads,
        "kda": 1 + rows * heads + grows * heads * dh,
        "rwkv7": 1 + 2 * grows * heads * dh,
        "lact": 1 + rows * heads,
        "mesa": 1 + rows * heads + heads,
    }[kind]


def audit_param_counts():
    print(f"\n== [5] parameter counts at d=768, R=3, num_heads={H} ==")
    res = {}
    for kind in KINDS:
        n = sum(p.numel() for p in make(kind, loops=3, heads=H, dim=768).parameters())
        want = expected_params(kind, 3, H, 768)
        res[kind] = (n, want)
        print(f"  {kind:9s} code={n:6d}  layout={want:6d}  {'match' if n == want else 'MISMATCH'}")
    return res


# --------------------------------------------------------------------------------
# 6. LaCT read is bounded in norm by 1 (Frobenius normalisation)
# --------------------------------------------------------------------------------
def audit_lact_bound():
    print(f"\n== [6] LaCT: per head ||o_h||_2 <= ||W_h||_F ||q_h||_2 = 1 after the Frobenius "
          f"normalisation, so ||o|| <= sqrt(H) = {math.sqrt(H):.2f} whatever the stream scale ==")
    mod = randomise_(make("lact"))
    ref = PaperReference(mod)
    out = {}
    for scale in (1.0, 30.0, 300.0):
        kw, sources, query, x, _ = make_history(4, scale, seed=5)
        o = code_read(mod, kw)
        lit = ref.lact_read(sources, query, normalise=False)
        on = float(o.norm(dim=-1).max())
        xn = float(x.float().norm(dim=-1).mean())
        assert on <= math.sqrt(H) * (1 + 1e-4), (on, H)
        out[scale] = (on, xn, float(lit.norm(dim=-1).mean()))
        print(f"  source scale {scale:6.1f}: max ||o|| = {on:.4f}  (stream ||h|| ~ {xn:.1f}, "
              f"ratio {on / xn:.2e});  the same step WITHOUT the normalisation gives "
              f"||o|| ~ {out[scale][2]:.1f}")
    return out


# --------------------------------------------------------------------------------
# 7. the h_0 path of the two blends (what an EMPTY / degenerate memory returns)
# --------------------------------------------------------------------------------
def audit_anchor_path():
    print("\n== [7] blend with a degenerate memory (every iteration writes nothing: out_r == in_r == h_0, "
          "the exact situation at training step 0) ==")
    a = 0.6
    h0 = rand_states(1, 300.0, seed=21)[0]
    for kind in KINDS:
        mod = randomise_(make(kind, loops=3))
        with torch.no_grad():
            mod.alpha.fill_(a)
            if UPDATE_IFACE:
                y = mod.combine(states=[h0] * 3, inputs=[h0] * 3)
            else:
                y = mod.combine(states=[h0] * 4)
        # write y = (1-a) h_r + a * (coef_h * h_0) per head; solve for coef_h by projection
        r = ((y - (1 - a) * h0) / a).view(B, T, mod.num_heads, mod.head_dim).double()
        hh = h0.view(B, T, mod.num_heads, mod.head_dim).double()
        coef = ((r * hh).sum(-1) / (hh * hh).sum(-1))                      # (B,T,H)
        resid = float((r - coef.unsqueeze(-1) * hh).abs().max() / hh.abs().max())
        print(f"  {kind:9s} y = (1-a) h_r + a * (coef h_0), coef in [{float(coef.min()):+.3f}, "
              f"{float(coef.max()):+.3f}] (off-h_0 residual {resid:.1e})"
              + ("   [memory empty: read == 0; a*h_0 is the hard-wired anchor path]"
                 if UPDATE_IFACE else "   [read over 4 identical states, no h_0 path outside the read]"))


def main():
    print(f"AUDIT {VARIANT}  kinds={KINDS}")
    r1 = audit_naive_vs_factorised()
    r2 = audit_init_identity()
    audit_newest_write_and_inits()
    r4 = audit_sources()
    r5 = audit_param_counts()
    audit_lact_bound()
    audit_anchor_path()
    bad = ([k for k, v in r1.items() if v[0] >= 1e-5 or v[1] >= 1e-5]
           + [k for k, v in r2.items() if not v]
           + [k for k, (n, want) in r5.items() if n != want]
           + (["sources"] if not r4.get("ok", True) else []))
    print("\nAUDIT " + ("PASSED" if not bad else f"FAILED: {bad}"))
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
