#!/usr/bin/env python3
"""Mathematical and code text (appendix "Mathematical and code text"): the teacher-forced loss of reference solutions
and five-shot numeric-answer accuracy of a trained checkpoint, and the paired bootstrap between two result files.

Reference-solution loss: GSM8K test (1,319 problems; calculator annotations <<...>> removed, the final "#### N" line
standardized to "The answer is N."), MATH-500 (500), HumanEval (164) and sanitized MBPP test (257; the prompt and the
first test as a docstring). Each sequence is BOS + problem + reference solution, cut to its last 4,096 tokens; the
loss is the NLL of the solution tokens given the problem, in nats per token (token-weighted) and bits per byte; the
four-set mean weights the sets equally.

Five-shot numeric-answer accuracy: SingleOp (562), SingleEq (508), AddSub (395), MultiArith (600), the numeric-answer
subset of ASDiv (2,084 of 2,215) and SVAMP (1,000), from ToRA's evaluation files. The prompt is the first five
standard-prompting exemplars of Wei et al. (2022), "Q: ...\\nA: The answer is N.", then the problem. Greedy exact
match: every token of the reference number is the argmax, the argmax after the number is not a digit, and when it is
"." the one after that is not a digit either. The mean NLL of the answer tokens is kept as a continuous measure; the
six-set mean weights the sets equally.

Context and continuation follow lm-eval's rule: the pair is tokenized once and the continuation is what follows the
context's own encoding. The model runs in bf16 on a GPU (fp32 on CPU), one sequence per forward, with the reference
Loopback read unless LOOP_DATTN_FUSED=1. Every item's result is kept in the JSON for the paired bootstrap.

  python scripts/eval_math_code.py --dump <run dir> --step 7280 --config <model json> --tokenizer <dir> \\
      --data <dir written by scripts/fetch_eval_data.py> --out <json> [--arm NAME] [--impl trainer|port] \\
      [--device cuda] [--limit N]
  python scripts/eval_math_code.py --paired A.json B.json [--draws 2000] [--seed 0] [--out diff.json]
      every metric's difference A - B with its paired bootstrap standard error: the problems of each set are
      resampled with replacement (2,000 draws, the same draw for both models), the set means are recomputed, and the
      standard error is the standard deviation of the differences over draws.

`--impl trainer` (default) is the training model, the implementation the paper's numbers used; it needs a GPU.
`--impl port` is looped_transformer.py, plain PyTorch on CPU or GPU.
"""
import argparse
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loopback_eval as LE  # noqa: E402

EXEMPLARS = [   # Wei et al. 2022, standard prompting: the first five math-word-problem exemplars
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will "
     "be 21 trees. How many trees did the grove workers plant today?", "6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give "
     "to Denny?", "8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "9"),
]
PREFIX = "".join(f"Q: {q}\nA: The answer is {a}.\n\n" for q, a in EXEMPLARS)
MWP_SETS = ("singleop", "singleeq", "addsub", "multiarith", "asdiv", "svamp")
NLL_SETS = ("gsm8k", "math500", "humaneval", "mbpp")


def fmt_num(x):   # 14.0 -> "14", 2.50 -> "2.5"
    x = float(x)
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.6f}".rstrip("0").rstrip(".")


def _rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def load_mwp(d):
    """{set: [(question, answer string)]}, and the number of ASDiv items skipped for a non-numeric answer."""
    sets = {n: [(r["input"].strip(), fmt_num(r["target"])) for r in _rows(os.path.join(d, f"tora_mawps_{n}.jsonl"))]
            for n in ("singleop", "singleeq", "addsub", "multiarith")}
    sets["asdiv"], skipped = [], 0
    for r in _rows(os.path.join(d, "tora_asdiv_test.jsonl")):
        m = re.match(r"^\s*(-?[\d,]*\.?\d+)\s*(\(.*\))?\s*$", r["answer"])
        if not m:
            skipped += 1
            continue
        sets["asdiv"].append((f'{r["body"].strip()} {r["question"].strip()}', fmt_num(m.group(1).replace(",", ""))))
    sets["svamp"] = [(f'{r["Body"].strip()} {r["Question"].strip()}', fmt_num(r["Answer"]))
                     for r in _rows(os.path.join(d, "tora_svamp_test.jsonl"))]
    return sets, skipped


def load_solutions(d):
    """{set: [(context, reference solution)]}."""
    import pandas as pd

    def gsm(a):
        a = re.sub(r"<<[^>]*>>", "", a.strip())
        return re.sub(r"\n#### *(.+)$", lambda m: f"\nThe answer is {m.group(1).strip()}.", a)

    g = pd.read_parquet(os.path.join(d, "gsm8k_test.parquet"))
    m = _rows(os.path.join(d, "math500_test.jsonl"))
    h = pd.read_parquet(os.path.join(d, "humaneval_test.parquet"))
    b = pd.read_parquet(os.path.join(d, "mbpp_sanitized_test.parquet"))
    return {"gsm8k": [(f"Question: {q}\nAnswer:", " " + gsm(a)) for q, a in zip(g.question, g.answer)],
            "math500": [(f"Problem: {r['problem']}\nSolution:", " " + r["solution"]) for r in m],
            "humaneval": list(zip(h.prompt, h.canonical_solution)),
            "mbpp": [(f'"""\n{p}\n{t[0]}\n"""\n', c) for p, t, c in zip(b.prompt, b.test_list, b.code)]}


def split_enc(tok, ctx, cont):
    """lm-eval's rule: encode context+continuation once; the continuation is what follows the context's own encoding."""
    enc = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
    whole, c = enc(ctx + cont), enc(ctx)
    return (c, whole[len(c):]) if whole[:len(c)] == c else (c, enc(cont))


@torch.inference_mode()
def logprobs(model, ids, device):
    logits = LE.forward(model, torch.tensor([ids], device=device))[0][0].float()
    return F.log_softmax(logits, dim=-1)


def evaluate(model, tok, data, device, limit=None, max_len=4096):
    """The result dict: per-set aggregates with every item's result, and the two means."""
    bos = LE.bos_id(tok)
    digit = {i for i in range(len(tok)) if (s := tok.convert_ids_to_tokens(i).lstrip("▁")) and s[0].isdigit()}
    dot = set(tok("x.", add_special_tokens=False).input_ids[-1:])
    res = {}

    sets, skipped = load_mwp(data)
    res["mwp"], res["mwp_asdiv_skipped_non_numeric"] = {}, skipped
    for name, items in sets.items():
        items = items[:limit] if limit else items
        out = []
        for q, gold in items:
            ctx = PREFIX + f"Q: {q}\nA: The answer is"
            c, num = split_enc(tok, ctx, " " + gold)
            _, full = split_enc(tok, ctx, " " + gold + ".")
            tail = full[len(num):] if full[:len(num)] == num else []
            ids = ([bos] + c + num + tail)[-max_len:]
            lp = logprobs(model, ids, device)
            p0 = len(ids) - len(num) - len(tail)                     # position of the first answer token
            rows, tgt = lp[p0 - 1:p0 - 1 + len(num)], torch.tensor(num, device=lp.device)
            nll = -float(rows.gather(1, tgt[:, None]).sum())
            ok = bool((rows.argmax(-1) == tgt).all())
            nxt = int(lp[p0 + len(num) - 1].argmax())               # what follows the number
            if ok and nxt in digit:
                ok = False
            elif ok and nxt in dot:                                  # "14." -- the token after the dot must not be a digit
                ok = bool(tail) and tail[0] == nxt and int(lp[p0 + len(num)].argmax()) not in digit
            out.append([int(ok), round(nll, 5), len(num)])
        n = len(out)
        res["mwp"][name] = dict(n=n, acc=100.0 * sum(r[0] for r in out) / n,
                                answer_nll_per_token=sum(r[1] for r in out) / sum(r[2] for r in out), items=out)
        print(f"[mwp] {name:11s} n={n:5d} acc {res['mwp'][name]['acc']:6.2f}  "
              f"answer nll/tok {res['mwp'][name]['answer_nll_per_token']:.4f}", flush=True)
    res["mwp_avg_acc"] = sum(v["acc"] for v in res["mwp"].values()) / len(res["mwp"])

    res["nll"] = {}
    for name, items in load_solutions(data).items():
        items = items[:limit] if limit else items
        out = []
        for ctx, sol in items:
            c, s = split_enc(tok, ctx, sol)
            ids = ([bos] + c + s)[-max_len:]
            s = s[-(len(ids) - 1):] if len(s) > len(ids) - 1 else s
            lp = logprobs(model, ids, device)
            p0 = len(ids) - len(s)
            tot = -float(lp[p0 - 1:p0 - 1 + len(s)].gather(1, torch.tensor(s, device=lp.device)[:, None]).sum())
            out.append([round(tot, 4), len(s), len(sol.encode("utf-8"))])
        T, B, N = sum(r[1] for r in out), sum(r[2] for r in out), sum(r[0] for r in out)
        res["nll"][name] = dict(n=len(out), tokens=T, nats_per_token=N / T, bits_per_byte=N / math.log(2) / B, items=out)
        print(f"[nll] {name:10s} n={len(out):5d} nats/tok {N / T:.4f}  bits/byte {N / math.log(2) / B:.4f}", flush=True)
    res["nll_avg_nats_per_token"] = sum(v["nats_per_token"] for v in res["nll"].values()) / len(res["nll"])
    return res


# ---- paired bootstrap between two result files ----

def _stat(part, items, idx):
    """The set's metric on the resampled items: accuracy (%) or token-weighted NLL per token."""
    if part == "mwp":
        return 100.0 * items[idx, 0].mean(axis=-1)
    return items[idx, 0].sum(axis=-1) / items[idx, 1].sum(axis=-1)


def paired_bootstrap(ra, rb, draws=2000, seed=0):
    """Differences A - B of every set metric and of the two means, with paired bootstrap standard errors over
    problems: the same resampled problem indices for both models, the set means recomputed per draw, and the
    standard error the standard deviation of the per-draw differences (the 2.5th and 97.5th percentiles as well)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = dict(a=ra.get("arm") or ra.get("model"), b=rb.get("arm") or rb.get("model"), draws=draws, seed=seed)
    for part, mean_key, sets in (("mwp", "mwp_avg_acc", MWP_SETS), ("nll", "nll_avg_nats_per_token", NLL_SETS)):
        out[part], per_set = {}, []
        for name in sets:
            A, B = (np.asarray(r[part][name]["items"], dtype=float) for r in (ra, rb))
            if A.shape != B.shape:
                raise SystemExit(f"{part}/{name}: {len(A)} items in A, {len(B)} in B; the files must score the same problems")
            n = len(A)
            idx = rng.integers(0, n, size=(draws, n))
            d = _stat(part, A, idx) - _stat(part, B, idx)
            per_set.append(d)
            out[part][name] = dict(n=n, diff=float(_stat(part, A, slice(None)) - _stat(part, B, slice(None))),
                                   se=float(d.std(ddof=1)), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
        d = np.mean(per_set, axis=0)
        out[mean_key] = dict(diff=float(ra[mean_key] - rb[mean_key]), se=float(d.std(ddof=1)),
                             ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    return out


def print_paired(p):
    print(f"A = {p['a']}\nB = {p['b']}\npaired bootstrap, {p['draws']} draws over problems (seed {p['seed']}); A - B")
    for part, mean_key, unit in (("mwp", "mwp_avg_acc", "accuracy (points)"), ("nll", "nll_avg_nats_per_token", "nats per token")):
        print(f"  {part} {unit}:")
        for name, v in p[part].items():
            print(f"    {name:11s} n={v['n']:5d}  diff {v['diff']:+.4f}  se {v['se']:.4f}  95% [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
        v = p[mean_key]
        print(f"    {'mean':11s}          diff {v['diff']:+.4f}  se {v['se']:.4f}  95% [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", help="run directory holding checkpoint/step-N")
    ap.add_argument("--step", type=int)
    ap.add_argument("--config", help="the model config json the run was trained with")
    ap.add_argument("--tokenizer", help="the tokenizer directory training used")
    ap.add_argument("--data", help="the directory written by scripts/fetch_eval_data.py")
    ap.add_argument("--out", help="result JSON (evaluation) or difference JSON (--paired)")
    ap.add_argument("--arm", default="", help="name recorded in the result JSON")
    ap.add_argument("--impl", choices=LE.IMPLS, default="trainer")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="items per set (testing only)")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--paired", nargs=2, metavar=("A.json", "B.json"), help="paired bootstrap of A - B instead of an evaluation")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.paired:
        p = paired_bootstrap(json.load(open(a.paired[0])), json.load(open(a.paired[1])), a.draws, a.seed)
        print_paired(p)
        if a.out:
            json.dump(p, open(a.out, "w"), indent=2)
            print(f"wrote {a.out}")
        return
    for k in ("dump", "step", "config", "tokenizer", "data", "out"):
        if getattr(a, k) is None:
            ap.error(f"--{k} is required for an evaluation")
    t0 = time.time()
    tok = LE.load_tokenizer(a.tokenizer)
    model = LE.load_model(a.dump, a.step, a.config, a.impl)
    dtype = torch.float32 if torch.device(a.device).type == "cpu" else torch.bfloat16
    model = model.to(dtype).to(a.device).eval()
    res = dict(arm=a.arm, model=f"{a.dump}/checkpoint/step-{a.step}", config=a.config,
               params=sum(p.numel() for p in model.parameters()), impl=a.impl, device=a.device, dtype=str(dtype),
               limit=a.limit, max_len=a.max_len, LOOP_DATTN_FUSED=os.environ.get("LOOP_DATTN_FUSED", "0"),
               host=os.uname().nodename,
               gpu=torch.cuda.get_device_name(a.device) if torch.device(a.device).type == "cuda" else None,
               evaluator="loopback/scripts/eval_math_code.py")
    res.update(evaluate(model, tok, a.data, a.device, a.limit, a.max_len))
    res["seconds"] = round(time.time() - t0, 1)
    print(f"[done] mwp avg acc {res['mwp_avg_acc']:.2f}  solution nll avg {res['nll_avg_nats_per_token']:.4f}  "
          f"({res['seconds']} s)", flush=True)
    json.dump(res, open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
