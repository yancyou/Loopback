"""Fused Triton kernels for Loopback's history read (`loop_residual_mode="delta_attnres"`, LoopDeltaAttnRes.combine).

One read at destination s has n = s + 1 sources, the raw block updates v_i (delta_i), and computes

    z_i   = rstd_i (q . v_i) / sqrt(D)                rstd_i the RMSNorm scale of v_i (an input with its own gradient)
    p     = softmax_i (z_i + lc_i),   p0 = softmax_i (lc_i),        lc_i = log max(c0_i, 1e-8),  M = sum_i c0_i
    tc_i  = c0_i + M (p_i - p0_i)                     the installed coefficients, returned for the next read
    out   = base + sum_i (tc_i - cbar_i) v_i          cbar_i the coefficients already installed by the previous read

exactly as LoopDeltaAttnRes.combine (the reference). The reference runs one GEMV per source for the scores and one
multiply-add per source into a (b, t, d) accumulator, so every source is read twice per read, the accumulator is
read and written once per source, and a read launches about 3n kernels; with KR reads over up to KR sources the
traffic grows with the loop depth squared (it cost the 191M arm 46 % of its training throughput, the 374M arm 49 %).

Here one kernel does a whole read: per tile of tokens it streams over the sources once, keeps two online softmaxes
(p and p0, so that a zero query still gives tc == c0 and out == base bitwise, as in the reference), and three fp32
accumulators (sum p_i v_i, sum p0_i v_i, sum (c0_i - cbar_i) v_i). The backward loads each source once more and
writes its gradient once:

    pass 1: gd_i = g . v_i,  gp_i = M (gd_i + gt_i),  Sacc = sum_i p_i gp_i,
            A = sum_i p_i rstd_i gp_i v_i,  B = sum_i p_i rstd_i v_i        ->  dq = sum_tok (A - Sacc B) / sqrt(D)
    pass 2: dz_i = p_i (gp_i - Sacc),  w_i = tc_i - cbar_i
            dv_i = w_i g + dz_i rstd_i q / sqrt(D)      (the RMSNorm path is autograd's, through rstd_i)
            drstd_i = dz_i z_i / rstd_i,   dcbar_i = -gd_i,   dbase = g
where g and gt are the upstream gradients of out and tc. Numerics: fp32 scores from the source dtype, fp32 softmax
and accumulation, one rounding of out to the source dtype (the reference accumulates the correction in the source
dtype, so on bf16 sources the fused read is the more accurate of the two).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_TILE = 2048            # BLOCK_T * BLOCK_D elements per tile (2026-11-07: 2048 with 4 warps; the released 4096 / 8 warps
                        # gives the same outputs bitwise; 1024 / 2 warps would be ~25% faster in the forward but changes the
                        # fp32 reduction order and so the bf16 outputs in the last bit -- not used)
_FWD_WARPS = 4          # 2026-11-07: launch settings as module globals so they can be tuned
_BWD_WARPS = 4
_BWD_TILES = 0          # tiles per backward program; 0 = the old rule (4 if n_tiles >= 4096 else 1)
_BWD2_TILE = 4096       # pass-2 tile (BLOCK_T * BLOCK_D); 4 tokens at d 1024 (sweep 11-07: 4096 / 4 warps)
_BWD2_WARPS = 4


@triton.jit
def _dattn_fwd_kernel(V_PTRS, R_PTRS, Q, BASE, CBAR, C0, LC, MPTR, OUT, TC, LSE, LSE0, LG, n_tok, S, INV,
                      D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr, IS_BF16: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    toks = pid * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    tmask = toks < n_tok
    dmask = offs < D
    mask2 = tmask[:, None] & dmask[None, :]
    q = tl.load(Q + offs, mask=dmask, other=0.0).to(tl.float32)
    M = tl.load(MPTR)
    base = tl.load(BASE + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
    # the scored softmax and the prior-only softmax run as the two lanes of one [.., 2] tensor, so that a zero query
    # (identical lanes) gives bitwise-identical accumulators and out == base exactly, whatever the compiler contracts
    acc2 = tl.zeros([BLOCK_T, BLOCK_D, 2], dtype=tl.float32)
    accc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    m2 = tl.full([BLOCK_T, 2], float("-inf"), tl.float32)
    lsum2 = tl.zeros([BLOCK_T, 2], dtype=tl.float32)
    for i in range(0, S):
        if IS_BF16:
            vp = tl.load(V_PTRS + i).to(tl.pointer_type(tl.bfloat16))
        else:
            vp = tl.load(V_PTRS + i).to(tl.pointer_type(tl.float32))
        rp = tl.load(R_PTRS + i).to(tl.pointer_type(tl.float32))
        v = tl.load(vp + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
        r = tl.load(rp + toks, mask=tmask, other=1.0)
        lc_i = tl.load(LC + i)
        c0_i = tl.load(C0 + i)
        cb_i = tl.load(CBAR + toks * S + i, mask=tmask, other=0.0)
        z = tl.sum(v * q[None, :], axis=1) * r * INV
        l = z + lc_i
        l0 = tl.zeros([BLOCK_T], dtype=tl.float32) + lc_i
        l2 = tl.join(l, l0)
        m2_new = tl.maximum(m2, l2)
        scale2 = tl.exp(m2 - m2_new)
        e2 = tl.exp(l2 - m2_new)
        acc2 = acc2 * scale2[:, None, :] + e2[:, None, :] * v[:, :, None]
        lsum2 = lsum2 * scale2 + e2
        m2 = m2_new
        accc += (c0_i - cb_i)[:, None] * v
        tl.store(LG + toks * S + i, l, mask=tmask)          # the logit, re-read below and by the backward
    lse2 = m2 + tl.log(lsum2)
    lse, lse0 = tl.split(lse2)
    pa, p0a = tl.split(acc2 / lsum2[:, None, :])
    out = base + accc + M * (pa - p0a)
    tl.store(OUT + toks[:, None] * D + offs[None, :], out.to(OUT.dtype.element_ty), mask=mask2)
    for i in range(0, S):
        l = tl.load(LG + toks * S + i, mask=tmask, other=0.0)
        lc_i = tl.load(LC + i)
        c0_i = tl.load(C0 + i)
        tc = c0_i + M * (tl.exp(l - lse) - tl.exp(lc_i - lse0))
        tl.store(TC + toks * S + i, tc, mask=tmask)
    tl.store(LSE + toks, lse, mask=tmask)
    tl.store(LSE0 + toks, lse0, mask=tmask)


@triton.jit
def _dattn_bwd_kernel(V_PTRS, R_PTRS, DV_PTRS, Q, G, GT, CBAR, C0, LC, MPTR, LSE, LSE0, LG, GD, DR, DCBAR, DQ_PART,
                      n_tok, S, INV,
                      D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr, TILES: tl.constexpr,
                      IS_BF16: tl.constexpr, ACCUM: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    dmask = offs < D
    q = tl.load(Q + offs, mask=dmask, other=0.0).to(tl.float32)
    M = tl.load(MPTR)
    dq = tl.zeros([BLOCK_D], dtype=tl.float32)
    for tb in range(0, TILES):
        toks = (pid * TILES + tb) * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
        tmask = toks < n_tok
        mask2 = tmask[:, None] & dmask[None, :]
        g = tl.load(G + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
        lse = tl.load(LSE + toks, mask=tmask, other=0.0)
        lse0 = tl.load(LSE0 + toks, mask=tmask, other=0.0)
        a_acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        b_acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        sacc = tl.zeros([BLOCK_T], dtype=tl.float32)
        # pass 1: the only read of the sources; the logits come from the forward (LG), g.v goes to scratch (GD)
        for i in range(0, S):
            if IS_BF16:
                vp = tl.load(V_PTRS + i).to(tl.pointer_type(tl.bfloat16))
            else:
                vp = tl.load(V_PTRS + i).to(tl.pointer_type(tl.float32))
            rp = tl.load(R_PTRS + i).to(tl.pointer_type(tl.float32))
            v = tl.load(vp + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
            r = tl.load(rp + toks, mask=tmask, other=1.0)
            gt_i = tl.load(GT + toks * S + i, mask=tmask, other=0.0)
            l = tl.load(LG + toks * S + i, mask=tmask, other=0.0)
            p = tl.exp(l - lse)
            gd = tl.sum(g * v, axis=1)
            gp = M * (gd + gt_i)
            sacc += p * gp
            pr = p * r
            a_acc += (pr * gp)[:, None] * v
            b_acc += pr[:, None] * v
            tl.store(GD + toks * S + i, gd, mask=tmask)
        dq += tl.sum(a_acc - sacc[:, None] * b_acc, axis=0) * INV
        # pass 2: the gradient of every source, rstd and installed coefficient, no source reads
        for i in range(0, S):
            rp = tl.load(R_PTRS + i).to(tl.pointer_type(tl.float32))
            r = tl.load(rp + toks, mask=tmask, other=1.0)
            lc_i = tl.load(LC + i)
            c0_i = tl.load(C0 + i)
            cb_i = tl.load(CBAR + toks * S + i, mask=tmask, other=0.0)
            gt_i = tl.load(GT + toks * S + i, mask=tmask, other=0.0)
            gd = tl.load(GD + toks * S + i, mask=tmask, other=0.0)
            l = tl.load(LG + toks * S + i, mask=tmask, other=0.0)
            p = tl.exp(l - lse)
            p0 = tl.exp(lc_i - lse0)
            w = c0_i - cb_i + M * (p - p0)
            gp = M * (gd + gt_i)
            dz = p * (gp - sacc)
            dv = w[:, None] * g + (dz * r * INV)[:, None] * q[None, :]
            if IS_BF16:
                dvp = tl.load(DV_PTRS + i).to(tl.pointer_type(tl.bfloat16))
                if ACCUM:   # prev + bf16(dv), rounded to bf16: what autograd's bf16 add_ of the returned dv did
                    prev = tl.load(dvp + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
                    dv = prev + dv.to(tl.bfloat16).to(tl.float32)
                tl.store(dvp + toks[:, None] * D + offs[None, :], dv.to(tl.bfloat16), mask=mask2)
            else:
                dvp = tl.load(DV_PTRS + i).to(tl.pointer_type(tl.float32))
                if ACCUM:
                    prev = tl.load(dvp + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0)
                    dv = prev + dv
                tl.store(dvp + toks[:, None] * D + offs[None, :], dv, mask=mask2)
            tl.store(DR + i * n_tok + toks, dz * (l - lc_i) / r, mask=tmask)
            tl.store(DCBAR + toks * S + i, -gd, mask=tmask)
    tl.store(DQ_PART + pid * D + offs, dq, mask=dmask)


def _ptr_array(tensors, device) -> torch.Tensor:
    # pinned + non_blocking: no stream synchronisation; the caching host allocator keeps the block alive for the copy
    host = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64, pin_memory=True)
    return host.to(device, non_blocking=True)


def _blocks(d: int):
    block_d = triton.next_power_of_2(d)
    block_t = max(1, _TILE // block_d)
    return block_d, block_t


def rstd_ref(delta: torch.Tensor, eps: float) -> torch.Tensor:
    """The RMSNorm scale of a source, (b, t, 1) fp32, differentiable: the key of the reference read is delta * rstd."""
    return torch.rsqrt(delta.float().pow(2).mean(-1, keepdim=True) + eps)


# 2026-11-07: the same scale in one Triton kernel each way. The PyTorch form above materialises an fp32 copy of the
# source and runs pow / mean / add / rsqrt (and their backward) as separate kernels, ~20% of Loopback's extra step time
# at 374M; here the bf16 source is read once and r = rsqrt(mean(x^2) + eps) is formed in fp32 registers. Backward:
# dx = g_r * (-r^3) * x / D. Same math; only the fp32 summation order differs (~1e-7 relative).
@triton.jit
def _rstd_fwd_kernel(X, R, n_tok, eps, D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    toks = pid * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    tmask = toks < n_tok
    mask2 = tmask[:, None] & (offs < D)[None, :]
    x = tl.load(X + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    tl.store(R + toks, r, mask=tmask)


@triton.jit
def _rstd_bwd_kernel(X, R, GR, GX, n_tok, D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    toks = pid * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    tmask = toks < n_tok
    mask2 = tmask[:, None] & (offs < D)[None, :]
    x = tl.load(X + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
    r = tl.load(R + toks, mask=tmask, other=0.0)
    gr = tl.load(GR + toks, mask=tmask, other=0.0)
    gx = (gr * (-r * r * r) / D)[:, None] * x
    tl.store(GX + toks[:, None] * D + offs[None, :], gx.to(GX.dtype.element_ty), mask=mask2)


class _Rstd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float):
        xc = x.contiguous()
        d = xc.shape[-1]
        n_tok = xc.numel() // d
        r = torch.empty(xc.shape[:-1] + (1,), device=xc.device, dtype=torch.float32)
        block_d, block_t = _blocks(d)
        _rstd_fwd_kernel[(triton.cdiv(n_tok, block_t),)](xc, r, n_tok, eps, D=d, BLOCK_D=block_d, BLOCK_T=block_t,
                                                         num_warps=4)
        ctx.save_for_backward(xc, r)
        return r

    @staticmethod
    def backward(ctx, gr: torch.Tensor):
        xc, r = ctx.saved_tensors
        d = xc.shape[-1]
        n_tok = xc.numel() // d
        gx = torch.empty_like(xc)
        block_d, block_t = _blocks(d)
        _rstd_bwd_kernel[(triton.cdiv(n_tok, block_t),)](xc, r, gr.contiguous(), gx, n_tok, D=d, BLOCK_D=block_d,
                                                         BLOCK_T=block_t, num_warps=4)
        return gx, None


def rstd(delta: torch.Tensor, eps: float) -> torch.Tensor:
    """The RMSNorm scale of a source, (b, t, 1) fp32, differentiable (fused kernels on CUDA, rstd_ref otherwise)."""
    if delta.is_cuda:
        return _Rstd.apply(delta, float(eps))
    return rstd_ref(delta, eps)


def _forward_impl(ctx, q, base, cbar, c0, srcs):
    n = len(srcs) // 2
    values, rstds = srcs[:n], srcs[n:]
    v0 = values[0]
    d = v0.shape[-1]
    n_tok = v0.numel() // d
    dev = v0.device
    qc = q.detach().float().contiguous()
    c0c = c0.detach().float().contiguous()
    lc = torch.log(c0c.clamp_min(1e-8))
    mvec = c0c.sum().reshape(1)
    cbar_c = cbar.detach().float().contiguous()
    base_c = base.detach().contiguous()
    block_d, block_t = _blocks(d)
    out = torch.empty_like(v0)
    tc = torch.empty((n_tok, n), device=dev, dtype=torch.float32)
    lse = torch.empty(n_tok, device=dev, dtype=torch.float32)
    lse0 = torch.empty(n_tok, device=dev, dtype=torch.float32)
    lg = torch.empty((n_tok, n), device=dev, dtype=torch.float32)     # the logits, kept for the backward
    vptr, rptr = _ptr_array(values, dev), _ptr_array(rstds, dev)
    grid = (triton.cdiv(n_tok, block_t),)
    _dattn_fwd_kernel[grid](vptr, rptr, qc, base_c, cbar_c, c0c, lc, mvec, out, tc, lse, lse0, lg, n_tok, n,
                            1.0 / (d ** 0.5), D=d, BLOCK_D=block_d, BLOCK_T=block_t,
                            IS_BF16=v0.dtype == torch.bfloat16, num_warps=_FWD_WARPS)
    ctx.save_for_backward(qc, cbar_c, c0c, lc, mvec, lse, lse0, lg, *values, *rstds)
    ctx.n, ctx.q_dtype, ctx.base_shape = n, q.dtype, base.shape
    return out, tc.view(*base.shape[:-1], n)


def _backward_impl(ctx, g, gt, dv, accum):
    """dv: one gradient buffer per source, written (accum False) or added into (accum True) by the kernel."""
    qc, cbar_c, c0c, lc, mvec, lse, lse0, lg, *rest = ctx.saved_tensors
    n = ctx.n
    values, rstds = rest[:n], rest[n:]
    v0 = values[0]
    d = v0.shape[-1]
    n_tok = v0.numel() // d
    dev = v0.device
    g = g.contiguous()
    gt_c = torch.zeros((n_tok, n), device=dev, dtype=torch.float32) if gt is None else gt.reshape(n_tok, n).float().contiguous()
    block_d, block_t = _blocks(d)
    n_tiles = triton.cdiv(n_tok, block_t)
    tiles = _BWD_TILES or (4 if n_tiles >= 4 * 1024 else 1)
    n_prog = triton.cdiv(n_tiles, tiles)
    dr = torch.empty((n, n_tok), device=dev, dtype=torch.float32)
    dcbar = torch.empty((n_tok, n), device=dev, dtype=torch.float32)
    dq_part = torch.empty((n_prog, d), device=dev, dtype=torch.float32)
    gd = torch.empty((n_tok, n), device=dev, dtype=torch.float32)     # scratch: g . v per (token, source)
    vptr, rptr, dvptr = _ptr_array(values, dev), _ptr_array(rstds, dev), _ptr_array(dv, dev)
    _dattn_bwd_kernel[(n_prog,)](vptr, rptr, dvptr, qc, g, gt_c, cbar_c, c0c, lc, mvec, lse, lse0, lg, gd, dr, dcbar,
                                 dq_part, n_tok, n, 1.0 / (d ** 0.5), D=d, BLOCK_D=block_d, BLOCK_T=block_t,
                                 TILES=tiles, IS_BF16=v0.dtype == torch.bfloat16, ACCUM=bool(accum),
                                 num_warps=_BWD_WARPS)
    dq = dq_part.sum(0).to(ctx.q_dtype)
    return dq, g, dcbar.view(*ctx.base_shape[:-1], n), [dr[i].view_as(rstds[i]) for i in range(n)]


class FusedDattnRead(torch.autograd.Function):
    """(out, tc) = FusedDattnRead.apply(q, base, cbar, c0, *values, *rstds); see the module docstring."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, base: torch.Tensor, cbar: torch.Tensor, c0: torch.Tensor, *srcs: torch.Tensor):
        return _forward_impl(ctx, q, base, cbar, c0, srcs)

    @staticmethod
    def backward(ctx, g: torch.Tensor, gt: torch.Tensor | None):
        n = ctx.n
        values = ctx.saved_tensors[8:8 + n]
        dv = [torch.empty_like(v) for v in values]   # one tensor per source: autograd releases each as soon as it can
        dq, g_out, dcbar_v, dr = _backward_impl(ctx, g, gt, dv, accum=False)
        return (dq, g_out, dcbar_v, None, *dv, *dr)


class DattnGradAccum:
    """Per-forward-pass state of the in-kernel gradient accumulation (2026-11-07). Source i (the update written at
    step i) is read by every later read, so autograd used to receive one full-size gradient tensor per (read, source)
    and add them up itself: at 374M about 210 extra bf16 adds per step, ~20% of Loopback's extra step time. With this
    object the backward of the last read (the first backward of the pass, which sees every source) allocates one
    gradient buffer per source; the earlier reads' backwards add their contribution into the buffers inside the
    kernel and return None for those inputs; the read that wrote source i last in backward order -- read i (and read
    1 for source 0, which read 0 never touches) -- returns the finished buffer. The chain of data dependencies fixes
    that order (read s's backward needs read s+1's), and the bf16 rounding after each addition is autograd's own, so
    every gradient is bitwise what the per-read tensors gave."""
    __slots__ = ("bufs",)

    def __init__(self):
        self.bufs = {}


class FusedDattnReadAcc(torch.autograd.Function):
    """FusedDattnRead with in-kernel gradient accumulation: apply(acc, step, q, base, cbar, c0, *values, *rstds)."""

    @staticmethod
    def forward(ctx, acc: DattnGradAccum, step: int, q, base, cbar, c0, *srcs):
        out = _forward_impl(ctx, q, base, cbar, c0, srcs)
        ctx.acc, ctx.step = acc, int(step)
        return out

    @staticmethod
    def backward(ctx, g: torch.Tensor, gt: torch.Tensor | None):
        acc, step, n = ctx.acc, ctx.step, ctx.n
        values = ctx.saved_tensors[8:8 + n]
        first = not acc.bufs
        if first:
            for i in range(n):
                acc.bufs[i] = torch.empty_like(values[i])
        elif any(i not in acc.bufs for i in range(n)):
            raise RuntimeError("delta_attnres gradient accumulation: a source without a buffer (reads out of order?)")
        dv = [acc.bufs[i] for i in range(n)]
        dq, g_out, dcbar_v, dr = _backward_impl(ctx, g, gt, dv, accum=not first)
        final = [i == step or (step == 1 and i == 0) for i in range(n)]
        return (None, None, dq, g_out, dcbar_v, None, *[dv[i] if final[i] else None for i in range(n)], *dr)


def fused_read_supported(base, values, rstds, cbar) -> bool:
    """CUDA, contiguous, same-shape same-dtype bf16/fp32 sources and base; contiguous fp32 (b, t, 1) rstds and a
    contiguous fp32 (b, t, n) cbar."""
    v0 = values[0]
    if not v0.is_cuda or v0.dtype not in (torch.bfloat16, torch.float32):
        return False
    shape = v0.shape
    if base.dtype != v0.dtype or base.shape != shape:
        return False
    for v in values:
        if v.dtype != v0.dtype or v.shape != shape or not v.is_contiguous():
            return False
    rshape = shape[:-1] + (1,)
    for r in rstds:
        if r.dtype != torch.float32 or r.shape != rshape or not r.is_contiguous():
            return False
    return cbar.dtype == torch.float32 and cbar.shape == shape[:-1] + (len(values),) and cbar.is_contiguous()


# ---------------------------------------------------------------------------------------------------------------
# 2026-11-07: the STACKED layout. The list layout above hands the kernels a pointer array and one tensor per source;
# its backward spent ~4.5 ms per pass on the per-source bookkeeping alone (pointer indirection, strided scalar loads),
# six times the forward. Here the model writes every block's update into its slice of one preallocated stack
# V[KR, b, t, d] (and the RMS scale into R[KR, b, t, 1]); the kernels index the stack directly: bitwise the same
# results, backward 1.5-3x faster. Gradients: dV / dR are stacks too, written by the first backward of the pass (the
# last read) and added into by the earlier reads inside the kernel (bf16 / fp32 rounding exactly as autograd's own
# accumulation of the per-read tensors); the reads return None for every update and scale input; the scale's
# backward (_RstdStk) adds its term into dV[i] and returns that slice as the update's one gradient, after the last
# read of the update (autograd runs it only once every read holding the scale has run). The stack's slices are views,
# so nothing here goes through save_for_backward (later slices are written in place; autograd's version check would
# refuse them); the backward of read s only touches slices 0..s, which are final by then.
@triton.jit
def _dattn_fwd_stk_kernel(V, R, Q, BASE, CBAR, C0, LC, MPTR, OUT, TC, LSE, LSE0, LG, n_tok, S, INV,
                          D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    toks = pid * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    tmask = toks < n_tok
    dmask = offs < D
    mask2 = tmask[:, None] & dmask[None, :]
    q = tl.load(Q + offs, mask=dmask, other=0.0).to(tl.float32)
    M = tl.load(MPTR)
    base = tl.load(BASE + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
    acc2 = tl.zeros([BLOCK_T, BLOCK_D, 2], dtype=tl.float32)
    accc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    m2 = tl.full([BLOCK_T, 2], float("-inf"), tl.float32)
    lsum2 = tl.zeros([BLOCK_T, 2], dtype=tl.float32)
    for i in range(0, S):
        v = tl.load(V + (i * n_tok + toks[:, None]) * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
        r = tl.load(R + i * n_tok + toks, mask=tmask, other=1.0)
        lc_i = tl.load(LC + i)
        c0_i = tl.load(C0 + i)
        cb_i = tl.load(CBAR + toks * S + i, mask=tmask, other=0.0)
        z = tl.sum(v * q[None, :], axis=1) * r * INV
        l = z + lc_i
        l0 = tl.zeros([BLOCK_T], dtype=tl.float32) + lc_i
        l2 = tl.join(l, l0)
        m2_new = tl.maximum(m2, l2)
        scale2 = tl.exp(m2 - m2_new)
        e2 = tl.exp(l2 - m2_new)
        acc2 = acc2 * scale2[:, None, :] + e2[:, None, :] * v[:, :, None]
        lsum2 = lsum2 * scale2 + e2
        m2 = m2_new
        accc += (c0_i - cb_i)[:, None] * v
        tl.store(LG + toks * S + i, l, mask=tmask)
    lse2 = m2 + tl.log(lsum2)
    lse, lse0 = tl.split(lse2)
    pa, p0a = tl.split(acc2 / lsum2[:, None, :])
    out = base + accc + M * (pa - p0a)
    tl.store(OUT + toks[:, None] * D + offs[None, :], out.to(OUT.dtype.element_ty), mask=mask2)
    for i in range(0, S):
        l = tl.load(LG + toks * S + i, mask=tmask, other=0.0)
        lc_i = tl.load(LC + i)
        c0_i = tl.load(C0 + i)
        tc = c0_i + M * (tl.exp(l - lse) - tl.exp(lc_i - lse0))
        tl.store(TC + toks * S + i, tc, mask=tmask)
    tl.store(LSE + toks, lse, mask=tmask)
    tl.store(LSE0 + toks, lse0, mask=tmask)


@triton.jit
def _dattn_bwd1_stk_kernel(V, R, G, GT, MPTR, LSE, LG, GD, SACC, DQ_PART, n_tok, S, INV,
                           D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr, TILES: tl.constexpr):
    """Pass 1 of the backward: the only read of the sources -- g.v (to GD), the per-token softmax term (SACC) and
    the query's gradient partials. Small tiles: it holds two [BLOCK_T, D] accumulators."""
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    dmask = offs < D
    M = tl.load(MPTR)
    dq = tl.zeros([BLOCK_D], dtype=tl.float32)
    for tb in range(0, TILES):
        toks = (pid * TILES + tb) * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
        tmask = toks < n_tok
        mask2 = tmask[:, None] & dmask[None, :]
        g = tl.load(G + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
        lse = tl.load(LSE + toks, mask=tmask, other=0.0)
        a_acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        b_acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        sacc = tl.zeros([BLOCK_T], dtype=tl.float32)
        for i in range(0, S):
            v = tl.load(V + (i * n_tok + toks[:, None]) * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
            r = tl.load(R + i * n_tok + toks, mask=tmask, other=1.0)
            gt_i = tl.load(GT + toks * S + i, mask=tmask, other=0.0)
            l = tl.load(LG + toks * S + i, mask=tmask, other=0.0)
            p = tl.exp(l - lse)
            gd = tl.sum(g * v, axis=1)
            gp = M * (gd + gt_i)
            sacc += p * gp
            pr = p * r
            a_acc += (pr * gp)[:, None] * v
            b_acc += pr[:, None] * v
            tl.store(GD + toks * S + i, gd, mask=tmask)
        dq += tl.sum(a_acc - sacc[:, None] * b_acc, axis=0) * INV
        tl.store(SACC + toks, sacc, mask=tmask)
    tl.store(DQ_PART + pid * D + offs, dq, mask=dmask)


@triton.jit
def _dattn_bwd2_stk_kernel(R, DV, Q, G, GT, CBAR, C0, LC, MPTR, LSE, LSE0, LG, GD, SACC, DR, DCBAR, n_tok, S, INV,
                           D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr, ACCUM: tl.constexpr):
    """Pass 2: the gradients of every source, scale and installed coefficient from the scalars of pass 1; no
    reductions and no accumulators, so it runs on wider token tiles."""
    pid = tl.program_id(0).to(tl.int64)
    toks = pid * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    tmask = toks < n_tok
    dmask = offs < D
    mask2 = tmask[:, None] & dmask[None, :]
    q = tl.load(Q + offs, mask=dmask, other=0.0).to(tl.float32)
    M = tl.load(MPTR)
    g = tl.load(G + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
    lse = tl.load(LSE + toks, mask=tmask, other=0.0)
    lse0 = tl.load(LSE0 + toks, mask=tmask, other=0.0)
    sacc = tl.load(SACC + toks, mask=tmask, other=0.0)
    for i in range(0, S):
        r = tl.load(R + i * n_tok + toks, mask=tmask, other=1.0)
        lc_i = tl.load(LC + i)
        c0_i = tl.load(C0 + i)
        cb_i = tl.load(CBAR + toks * S + i, mask=tmask, other=0.0)
        gt_i = tl.load(GT + toks * S + i, mask=tmask, other=0.0)
        gd = tl.load(GD + toks * S + i, mask=tmask, other=0.0)
        l = tl.load(LG + toks * S + i, mask=tmask, other=0.0)
        p = tl.exp(l - lse)
        p0 = tl.exp(lc_i - lse0)
        w = c0_i - cb_i + M * (p - p0)
        gp = M * (gd + gt_i)
        dz = p * (gp - sacc)
        dv = w[:, None] * g + (dz * r * INV)[:, None] * q[None, :]
        dvp = DV + (i * n_tok + toks[:, None]) * D + offs[None, :]
        if ACCUM:
            prev = tl.load(dvp, mask=mask2, other=0.0).to(tl.float32)
            dv = prev + dv.to(DV.dtype.element_ty).to(tl.float32)
        tl.store(dvp, dv.to(DV.dtype.element_ty), mask=mask2)
        dr = dz * (l - lc_i) / r
        if ACCUM:
            dr = tl.load(DR + i * n_tok + toks, mask=tmask, other=0.0) + dr
        tl.store(DR + i * n_tok + toks, dr, mask=tmask)
        tl.store(DCBAR + toks * S + i, -gd, mask=tmask)


@triton.jit
def _rstd_bwd_acc_kernel(X, R, GR, DV, n_tok, D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr):
    """DV += rounded(gr * (-r^3) * x / D), rounded: the scale's gradient term added to the update's gradient buffer."""
    pid = tl.program_id(0).to(tl.int64)
    toks = pid * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    tmask = toks < n_tok
    mask2 = tmask[:, None] & (offs < D)[None, :]
    x = tl.load(X + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
    r = tl.load(R + toks, mask=tmask, other=0.0)
    gr = tl.load(GR + toks, mask=tmask, other=0.0)
    gx = ((gr * (-r * r * r) / D)[:, None] * x).to(DV.dtype.element_ty).to(tl.float32)
    prev = tl.load(DV + toks[:, None] * D + offs[None, :], mask=mask2, other=0.0).to(tl.float32)
    tl.store(DV + toks[:, None] * D + offs[None, :], (prev + gx).to(DV.dtype.element_ty), mask=mask2)


def _alias(stack_tensor: torch.Tensor, s: int) -> torch.Tensor:
    """Slice s of a stack as a tensor that shares its storage but is not an autograd view of it (set_ on a fresh
    tensor): later slices are written in place, and autograd forbids a custom Function to return a view whose base
    is then modified."""
    sl = stack_tensor[s]
    out = torch.empty(0, dtype=stack_tensor.dtype, device=stack_tensor.device)
    out.set_(stack_tensor.untyped_storage(), sl.storage_offset(), sl.shape, sl.stride())
    return out


class DattnStack:
    """The stacked storage of one forward pass: V[n_steps, b, t, d] (the updates), R[n_steps, b, t, 1] (their RMS
    scales), and, from the first backward on, their gradient stacks dV / dR."""
    __slots__ = ("V", "R", "dV", "dR", "n_steps", "n_tok", "d")

    def __init__(self, n_steps: int, shape, dtype, device):
        b, t, d = shape
        self.n_steps, self.n_tok, self.d = int(n_steps), b * t, d
        self.V = torch.empty((n_steps, b, t, d), dtype=dtype, device=device)
        self.R = torch.empty((n_steps, b, t, 1), dtype=torch.float32, device=device)
        self.dV = self.dR = None

    def write(self, s: int, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """V[s] = h - x, returned as the update tensor (a view of the stack, with the autograd edge to h and x)."""
        return _WriteDelta.apply(self, s, h, x)

    def rstd(self, s: int, delta: torch.Tensor, eps: float) -> torch.Tensor:
        """R[s] = the RMS scale of V[s], returned as a view of the stack (differentiable through `delta`)."""
        return _RstdStk.apply(self, s, delta, float(eps))

    def grads(self):
        if self.dV is None:
            self.dV = torch.empty_like(self.V)
            self.dR = torch.empty_like(self.R)
            return False
        return True


class _WriteDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, stack: DattnStack, s: int, h: torch.Tensor, x: torch.Tensor):
        out = _alias(stack.V, s)
        torch.sub(h, x, out=out)
        return out

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        return None, None, g, -g


class _RstdStk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, stack: DattnStack, s: int, x: torch.Tensor, eps: float):
        r = _alias(stack.R, s)
        d = x.shape[-1]
        block_d, block_t = _blocks(d)
        _rstd_fwd_kernel[(triton.cdiv(stack.n_tok, block_t),)](x, r, stack.n_tok, eps, D=d, BLOCK_D=block_d,
                                                               BLOCK_T=block_t, num_warps=4)
        ctx.stack, ctx.s, ctx.x = stack, s, x
        return r

    @staticmethod
    def backward(ctx, gr: torch.Tensor):
        stack, s, x = ctx.stack, ctx.s, ctx.x
        if stack.dV is None:
            raise RuntimeError("delta_attnres stacked read: the scale's backward ran before any read's backward")
        d = x.shape[-1]
        block_d, block_t = _blocks(d)
        _rstd_bwd_acc_kernel[(triton.cdiv(stack.n_tok, block_t),)](x, stack.R[s], gr.contiguous(), stack.dV[s],
                                                                   stack.n_tok, D=d, BLOCK_D=block_d, BLOCK_T=block_t,
                                                                   num_warps=4)
        return None, None, _alias(stack.dV, s), None


class FusedDattnReadStk(torch.autograd.Function):
    """(out, tc) = apply(stack, step, q, base, cbar, c0, *updates, *scales): the updates and scales are the stack's
    slices 0..step (inputs only for autograd's bookkeeping; the kernels read the stack)."""

    @staticmethod
    def forward(ctx, stack: DattnStack, step: int, q, base, cbar, c0, *srcs):
        n = len(srcs) // 2
        d, n_tok, dev = stack.d, stack.n_tok, stack.V.device
        if base.dtype != stack.V.dtype or not base.is_contiguous() or base.shape[:-1] != stack.V.shape[1:3]:
            raise RuntimeError("delta_attnres stacked read: base must be contiguous, the stack's dtype and shape")
        qc = q.detach().float().contiguous()
        c0c = c0.detach().float().contiguous()
        lc = torch.log(c0c.clamp_min(1e-8))
        mvec = c0c.sum().reshape(1)
        cbar_c = cbar.detach().float().contiguous()
        block_d, block_t = _blocks(d)
        out = torch.empty_like(base)
        tc = torch.empty((n_tok, n), device=dev, dtype=torch.float32)
        lse = torch.empty(n_tok, device=dev, dtype=torch.float32)
        lse0 = torch.empty(n_tok, device=dev, dtype=torch.float32)
        lg = torch.empty((n_tok, n), device=dev, dtype=torch.float32)
        _dattn_fwd_stk_kernel[(triton.cdiv(n_tok, block_t),)](stack.V, stack.R, qc, base.detach(), cbar_c, c0c, lc, mvec,
                                                              out, tc, lse, lse0, lg, n_tok, n, 1.0 / (d ** 0.5),
                                                              D=d, BLOCK_D=block_d, BLOCK_T=block_t, num_warps=_FWD_WARPS)
        ctx.save_for_backward(qc, cbar_c, c0c, lc, mvec, lse, lse0, lg)
        ctx.stack, ctx.step, ctx.n, ctx.q_dtype, ctx.base_shape = stack, int(step), n, q.dtype, base.shape
        return out, tc.view(*base.shape[:-1], n)

    @staticmethod
    def backward(ctx, g: torch.Tensor, gt: torch.Tensor | None):
        qc, cbar_c, c0c, lc, mvec, lse, lse0, lg = ctx.saved_tensors
        stack, step, n = ctx.stack, ctx.step, ctx.n
        d, n_tok, dev = stack.d, stack.n_tok, stack.V.device
        accum = stack.grads()
        g = g.contiguous()
        gt_c = torch.zeros((n_tok, n), device=dev, dtype=torch.float32) if gt is None else gt.reshape(n_tok, n).float().contiguous()
        block_d, block_t = _blocks(d)
        n_tiles = triton.cdiv(n_tok, block_t)
        tiles = _BWD_TILES or (4 if n_tiles >= 4 * 1024 else 1)
        n_prog = triton.cdiv(n_tiles, tiles)
        dcbar = torch.empty((n_tok, n), device=dev, dtype=torch.float32)
        dq_part = torch.empty((n_prog, d), device=dev, dtype=torch.float32)
        gd = torch.empty((n_tok, n), device=dev, dtype=torch.float32)
        sacc = torch.empty(n_tok, device=dev, dtype=torch.float32)
        _dattn_bwd1_stk_kernel[(n_prog,)](stack.V, stack.R, g, gt_c, mvec, lse, lg, gd, sacc, dq_part, n_tok, n,
                                          1.0 / (d ** 0.5), D=d, BLOCK_D=block_d, BLOCK_T=block_t, TILES=tiles,
                                          num_warps=_BWD_WARPS)
        block_t2 = max(1, _BWD2_TILE // block_d)
        _dattn_bwd2_stk_kernel[(triton.cdiv(n_tok, block_t2),)](stack.R, stack.dV, qc, g, gt_c, cbar_c, c0c, lc, mvec,
                                                                lse, lse0, lg, gd, sacc, stack.dR, dcbar, n_tok, n,
                                                                1.0 / (d ** 0.5), D=d, BLOCK_D=block_d, BLOCK_T=block_t2,
                                                                ACCUM=bool(accum), num_warps=_BWD2_WARPS)
        dq = dq_part.sum(0).to(ctx.q_dtype)
        final = [i == step or (step == 1 and i == 0) for i in range(n)]   # the last read of scale i, in backward order
        return (None, None, dq, g, dcbar.view(*ctx.base_shape[:-1], n), None,
                *[None] * n, *[_alias(stack.dR, i) if final[i] else None for i in range(n)])
