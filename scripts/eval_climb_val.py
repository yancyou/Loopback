#!/usr/bin/env python3
"""The paper's validation loss of a trained checkpoint: the held-out ClimbMix shard packed as in training (documents
tokenized with the beginning-of-sequence marker, concatenated, cut into 4,096-token windows), the first 1,024 windows,
one window per forward in bf16, mean next-token cross-entropy in nats (perplexity = exp). Every model is scored on the
same windows, so the comparison is paired.

    python scripts/eval_climb_val.py --dump <run dir> --step 7280 --config configs/191m_d768x20/lt124m_ours_p2k4l4c2_4klen.json \\
        --tokenizer trainer/transformer-1.3B-100B --data_glob "$DATA_ROOT/val/*.arrow" --out val.json \\
        [--impl trainer|port] [--n_seq 1024 --seq_len 4096] [--dtype bf16|fp32] [--device cuda]

impl "trainer" (default) is the implementation the paper's numbers used; impl "port" (looped_transformer.py) needs no
flame dependencies, runs on CPU, and agrees with it to four decimals (scripts/loopback_eval.py).
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loopback_eval import IMPLS, bos_id, load_model, load_tokenizer, val_loss, val_windows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="run directory holding checkpoint/step-N")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--config", required=True, help="the model config JSON the run was trained with")
    ap.add_argument("--tokenizer", required=True, help="the tokenizer directory training used")
    ap.add_argument("--data_glob", required=True, help="arrow shards of the held-out split")
    ap.add_argument("--n_seq", type=int, default=1024)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--impl", choices=IMPLS, default="trainer")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    seqs = val_windows(args.data_glob, tok, args.seq_len, args.n_seq)
    bos = int((seqs == bos_id(tok)).sum()) / seqs.shape[0]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = load_model(args.dump, args.step, args.config, args.impl).to(dtype).to(args.device)
    loss = val_loss(model, seqs, args.device)
    print(f"VAL {args.arm or args.dump} loss={loss:.4f} ppl={math.exp(loss):.2f} n={seqs.shape[0]} seq_len={args.seq_len} "
          f"impl={args.impl} bos_per_seq={bos:.1f}", flush=True)
    if args.out:
        json.dump(dict(arm=args.arm, dump=args.dump, step=args.step, config=args.config, impl=args.impl, n_seq=int(seqs.shape[0]),
                       seq_len=args.seq_len, val_loss=loss, ppl=math.exp(loss), bos_per_seq=bos), open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
