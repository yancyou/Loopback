#!/usr/bin/env python3
"""Validation loss of a trained looped checkpoint run with a different number of iterations of its shared stack, without
retraining (the paper's "robustness to additional visits" figure and the 3B extra-visit table).

The model is rebuilt with num_hidden_layers = P + C + K * R'. Prelude blocks, the embedding, the final norm and the head
load as trained; middle position i loads shared block (i - P) mod K; coda block j loads the trained coda block at the
same distance from the end. Loopback's queries (K R - 1 trained rows, one per destination after the first read) become
K R' - 1 rows:
  --query last   a destination beyond the trained depth reuses the trained query of the same block position in the
                 last trained iteration (destination s -> row (R - 1) K + s mod K - 1); the paper's setting
  --query zero   it gets a zero query (the decay prior alone)
R' < R keeps the first K R' - 1 rows. A Standard checkpoint (no shared blocks) is evaluated once. The loss is the paper's
validation protocol (scripts/loopback_eval.val_windows: the held-out shard packed as in training, 1,024 windows of 4,096
tokens, bf16, one sequence per forward).

    python scripts/eval_extra_visits.py --dump <run dir> --step 7280 --config configs/191m_d768x20/lt124m_ours_p2k4l4c2_4klen.json \\
        --tokenizer trainer/transformer-1.3B-100B --data_glob "$DATA_ROOT/val/*.arrow" \\
        --iterations 1,2,3,4,5,6,8 --query last,zero --out extra_visits.json [--impl trainer|port]
"""
import argparse
import copy
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loopback_eval import IMPLS, load_model, load_tokenizer, val_loss, val_windows  # noqa: E402

LAYER = re.compile(r"^(.*\blayers\.)(\d+)(\..*)$")


def topology(cfg):
    P, C, L = cfg.num_prelude_layers, cfg.num_coda_layers, cfg.num_hidden_layers
    K = cfg.num_middle_param_layers
    return P, C, L, K, ((L - P - C) // K if K else None)


def remap(sd0, P, C, L0, K, R0, R, query):
    """The state dict of the R'-iteration model from the trained one (R0 iterations)."""
    L = P + C + K * R

    def src(i):
        if i < P:
            return i
        if i < L - C:
            return P + (i - P) % K
        return L0 - (L - i)

    out, layer_keys = {}, {}
    for k, v in sd0.items():
        m = LAYER.match(k)
        if m:
            layer_keys.setdefault(int(m.group(2)), []).append((k, m.group(1), m.group(3)))
        elif k.endswith("loop_dattn.query"):
            q = torch.zeros(K * R - 1, v.shape[1], dtype=v.dtype)
            for s in range(1, K * R):                       # destination s uses row s - 1
                if s < K * R0:
                    q[s - 1] = v[s - 1]
                elif query == "last":
                    row = (R0 - 1) * K + s % K - 1
                    if row >= 0:
                        q[s - 1] = v[row]
            out[k] = q
        else:
            out[k] = v
    for i in range(L):                                      # every position takes its source block's tensors
        for k0, head, tail in layer_keys[src(i)]:
            out[f"{head}{i}{tail}"] = sd0[k0]
    return out


def rebuild(model, impl, R, query):
    """A copy of `model` with R iterations, weights remapped from the trained checkpoint."""
    cfg0 = model.config
    P, C, L0, K, R0 = topology(cfg0)
    sd0 = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if impl == "port":
        from looped_transformer import LoopedTransformerLM
        cfg = copy.deepcopy(cfg0)
        cfg.num_hidden_layers = P + C + K * R
        m = LoopedTransformerLM(cfg)
    else:
        from transformers import AutoModelForCausalLM
        cfg = copy.deepcopy(cfg0)
        cfg.num_hidden_layers = P + C + K * R
        m = AutoModelForCausalLM.from_config(cfg)
    sd = remap(sd0, P, C, L0, K, R0, R, query)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and not missing, (missing[:5], unexpected[:5])
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data_glob", required=True)
    ap.add_argument("--iterations", default="1,2,3,4,5,6,8", help="comma-separated R' values")
    ap.add_argument("--query", default="last", help="last | zero | last,zero (Loopback only)")
    ap.add_argument("--n_seq", type=int, default=1024)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--impl", choices=IMPLS, default="trainer")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    t0 = time.time()
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    model = load_model(a.dump, a.step, a.config, a.impl)
    cfg = model.config
    P, C, L0, K, R0 = topology(cfg)
    mode = cfg.loop_residual_mode
    tok = load_tokenizer(a.tokenizer)
    seqs = val_windows(a.data_glob, tok, a.seq_len, a.n_seq)
    res = dict(name=a.name, dump=a.dump, step=a.step, config=a.config, impl=a.impl, mode=mode, P=P, K=K, C=C,
               R_trained=R0, n_seq=int(seqs.shape[0]), seq_len=a.seq_len, rows=[])
    if K is None:                                           # Standard: nothing to iterate
        v = val_loss(model.to(dtype).to(a.device), seqs, a.device)
        res["rows"].append(dict(R=None, query="-", val_loss=v))
        print(f"[extra_visits] {a.name or a.dump}: Standard, val {v:.4f}", flush=True)
    else:
        queries = a.query.split(",") if mode == "delta_attnres" else ["-"]
        for R in [int(x) for x in a.iterations.split(",")]:
            for qm in queries:
                if mode == "delta_attnres" and R <= R0 and qm == "zero":
                    continue                                # identical to "last" there
                t = time.time()
                m = rebuild(model, a.impl, R, qm).to(dtype).to(a.device)
                v = val_loss(m, seqs, a.device)
                del m
                if a.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                res["rows"].append(dict(R=R, query=qm, val_loss=v))
                print(f"[extra_visits] {a.name or a.dump}: trained R={R0} K={K}  R'={R} query={qm:4s}  val {v:.4f}  "
                      f"({time.time() - t:.0f} s)", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
