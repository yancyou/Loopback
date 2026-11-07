#!/usr/bin/env python3
"""Downstream tasks of the paper (lm-evaluation-harness 0.4.12) for a trained checkpoint, through the lm-eval model
class whose request handling produced the paper's tables.

Protocol (appendix, "Downstream tasks"): zero-shot prompts, the full evaluation sets, 1,000 bootstrap iterations for
the standard errors, and the beginning-of-sequence token at the start of every context -- training prepends it to
every document, and a prompt without it is an input no model saw in training (`--bos 0` is the earlier behaviour,
BOS only for an empty context, kept for re-deriving old tables). Scoring: context and continuation are tokenized
separately (no special tokens), concatenated after BOS, cut to the last 4,096 tokens with BOS kept in front, and the
continuation's log-probability is the sum over its tokens of the fp32 log-softmax of the model's logits (bf16 on a
GPU, fp32 on CPU); requests are batched by length and right-padded, which a causal model does not see. WikiText word
perplexity scores each document in non-overlapping 4,095-token windows, the first from BOS, each later one with the
previous window's last token carried as context. Generation (no default task needs it) is greedy and recomputes the
prefix at every step, capped at 48 new tokens.

The default task list is the paper's: the thirteen accuracy tasks of the main table and WikiText. HellaSwag,
OpenBookQA, ARC-Easy and ARC-Challenge are read as acc_norm, the others as acc, MMLU as the harness's "mmlu"
aggregate over 57 subjects; `scripts/aggregate_downstream.py` turns the JSONs into the table rows.

  python scripts/eval_downstream.py --dump <run dir> --step 7280 \
      --config configs/191m_d768x20/lt124m_ours_p2k4l4c2_4klen.json --tokenizer trainer/transformer-1.3B-100B \
      --out <dir> --arm loopback_191m [--impl trainer|port] [--device cuda] [--tasks a,b,c] [--limit N] \
      [--bootstrap_iters 1000]

`--impl trainer` (default) is the training model, the implementation every number in the paper used; it needs a GPU.
`--impl port` is looped_transformer.py, plain PyTorch on CPU or GPU. Writes <out>/<arm>.json: every numeric metric
of every task with its stderr, and the harness version, sample counts, task versions and group aggregates.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loopback_eval as LE  # noqa: E402
from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.api.model import LM  # noqa: E402

TASKS = ("piqa,hellaswag,winogrande,openbookqa,commonsense_qa,boolq,race,arc_easy,arc_challenge,mmlu,"
         "anli_r1,rte,lambada_openai,wikitext")


class LoopbackLM(LM):
    """lm-eval model over either implementation of the checkpoint (`loopback_eval.forward`)."""

    def __init__(self, model, tokenizer, device="cuda", max_len=4096, batch_tokens=8192, max_batch=16,
                 max_gen_toks=48, bos=1):
        super().__init__()
        self.tok = tokenizer
        self.bos = int(bos)
        self.bos_id = LE.bos_id(self.tok)
        dtype = torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16
        self.model = model.to(dtype).to(device).eval()
        self._device, self.max_len = device, max_len  # the LM base class owns a read-only `device`
        self.batch_tokens, self.max_batch, self.max_gen_toks = batch_tokens, max_batch, max_gen_toks

    def _enc(self, s):
        return self.tok(s, add_special_tokens=False).input_ids

    def _logits(self, x):
        return LE.forward(self.model, x.to(self._device))[0].float()

    @torch.no_grad()
    def _score(self, rows):
        """rows = [(idx, ids, n_cont)]; returns {idx: (logprob, is_greedy)}."""
        out = {}
        rows = sorted(rows, key=lambda r: -len(r[1]))
        i = 0
        while i < len(rows):
            batch, L = [], len(rows[i][1])
            while i < len(rows) and len(batch) < self.max_batch and (len(batch) + 1) * L <= self.batch_tokens:
                batch.append(rows[i])
                i += 1
            if not batch:
                batch, i = [rows[i]], i + 1
            L = max(len(r[1]) for r in batch)
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            x = torch.full((len(batch), L), pad, dtype=torch.long)
            for j, (_, ids, _) in enumerate(batch):
                x[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            logp = torch.log_softmax(self._logits(x), dim=-1)
            for j, (idx, ids, n_cont) in enumerate(batch):
                n = len(ids)
                tgt = torch.tensor(ids[n - n_cont: n], dtype=torch.long, device=logp.device)
                lp = logp[j, n - n_cont - 1: n - 1, :]
                out[idx] = (float(lp.gather(-1, tgt[:, None]).sum()), bool((lp.argmax(-1) == tgt).all()))
        return out

    def loglikelihood(self, requests):
        rows = []
        for k, req in enumerate(requests):
            ctx, cont = req.args
            if self.bos:
                c_ids = [self.bos_id] + self._enc(ctx)
            else:
                c_ids = self._enc(ctx) if ctx else [self.tok.bos_token_id or 1]
            k_ids = self._enc(cont)
            if not k_ids:
                k_ids = [self.tok.eos_token_id or 2]
            ids = (c_ids + k_ids)[-self.max_len:]
            if self.bos and ids[0] != self.bos_id:  # a context longer than the window: keep BOS in front
                ids = [self.bos_id] + ids[1:]
            rows.append((k, ids, min(len(k_ids), len(ids) - 1)))
        sc = self._score(rows)
        return [sc[k] for k in range(len(requests))]

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        """Total log-probability of each whole document (WikiText perplexity): non-overlapping windows, the first
        token of the document scored from BOS, the first token of every later window from the previous window's
        last token."""
        out = []
        for req in requests:
            (text,) = req.args
            ids = self._enc(text)
            if not ids:
                out.append(0.0)
                continue
            total, prev = 0.0, self.bos_id
            for i in range(0, len(ids), self.max_len - 1):
                win = ids[i: i + self.max_len - 1]
                x = torch.tensor([[prev] + win], dtype=torch.long)
                logp = torch.log_softmax(self._logits(x), dim=-1)[0, :-1, :]
                tgt = torch.tensor(win, dtype=torch.long, device=logp.device)
                total += float(logp.gather(-1, tgt[:, None]).sum())
                prev = win[-1]
            out.append(total)
        return out

    @torch.no_grad()
    def generate_until(self, requests):
        """Greedy generation, one request at a time, recomputing the prefix each step (there is no incremental KV
        path), so it is affordable only for answers of a few dozen tokens: `max_gen_toks` caps the task's budget.
        Stops on any `until` string or on EOS. No task of the default list generates."""
        out = []
        eos = self.tok.eos_token_id
        for req in requests:
            ctx, kwargs = req.args
            kwargs = dict(kwargs or {})
            until = kwargs.get("until") or []
            if isinstance(until, str):
                until = [until]
            max_new = min(int(kwargs.get("max_gen_toks", 32)), self.max_gen_toks)
            ids = self._enc(ctx)[-(self.max_len - max_new):]
            if self.bos:
                ids = [self.bos_id] + ids[-(self.max_len - max_new - 1):]
            gen = []
            for _ in range(max_new):
                x = torch.tensor([ids + gen], dtype=torch.long)
                nxt = int(self._logits(x)[0, -1].argmax())
                if eos is not None and nxt == eos:
                    break
                gen.append(nxt)
                txt = self.tok.decode(gen)
                if any(u and u in txt for u in until):
                    break
            txt = self.tok.decode(gen)
            for u in until:
                if u and u in txt:
                    txt = txt.split(u)[0]
            out.append(txt)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", required=True, help="run directory holding checkpoint/step-N")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--config", required=True, help="the model config json the run was trained with")
    ap.add_argument("--tokenizer", required=True, help="the tokenizer directory training used")
    ap.add_argument("--out", required=True, help="directory of the result JSONs")
    ap.add_argument("--arm", required=True, help="name of the result file, <out>/<arm>.json")
    ap.add_argument("--impl", choices=LE.IMPLS, default="trainer")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--limit", type=int, default=None, help="examples per task (testing only)")
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--bootstrap_iters", type=int, default=1000, help="bootstrap iterations for the stderrs; 0 = none")
    ap.add_argument("--bos", type=int, default=1, choices=(0, 1),
                    help="1 = BOS at the start of every context (the paper); 0 = BOS only for an empty context")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lm = LoopbackLM(LE.load_model(a.dump, a.step, a.config, a.impl), LE.load_tokenizer(a.tokenizer),
                    device=a.device, bos=a.bos)
    res = simple_evaluate(model=lm, tasks=a.tasks.split(","), num_fewshot=a.num_fewshot, limit=a.limit,
                          bootstrap_iters=a.bootstrap_iters, verbosity="ERROR")
    table = {}
    print(f"\n=== {a.arm} ({a.impl}, step {a.step}, bos {a.bos}) ===")
    for t, r in sorted(res["results"].items()):
        keep = {k: v for k, v in r.items() if k != "alias" and not isinstance(v, str)}  # every numeric metric (+ stderr)
        table[t] = keep
        head = keep.get("acc_norm,none", keep.get("acc,none"))
        if head is not None:
            print(f"  {t:<18} {100 * head:6.2f}%   {keep}")
        else:
            print(f"  {t:<18} {'':>7}   {keep}")
    import lm_eval as _le
    meta = dict(lm_eval_version=getattr(_le, "__version__", "?"), num_fewshot=a.num_fewshot,
                bootstrap_iters=a.bootstrap_iters, n_samples=res.get("n-samples"), versions=res.get("versions"),
                higher_is_better=res.get("higher_is_better"), groups=res.get("groups"),
                group_subtasks=res.get("group_subtasks"))
    with open(os.path.join(a.out, f"{a.arm}.json"), "w") as f:
        json.dump(dict(arm=a.arm, step=a.step, limit=a.limit, bos=a.bos, impl=a.impl, dump=a.dump, config=a.config,
                       results=table, meta=meta), f, indent=2)
    print(f"wrote {a.out}/{a.arm}.json")


if __name__ == "__main__":
    main()
