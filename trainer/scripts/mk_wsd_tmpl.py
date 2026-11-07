"""Derive a per-arm job template from the recipe template by replacing whole lines (every replaced key must exist
exactly once, so a template drift fails loudly instead of silently keeping the old value). Defaults = the paper's recipe."""
import re, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
ap.add_argument("--micro", type=int, required=True); ap.add_argument("--accum", type=int, required=True)
ap.add_argument("--lr", required=True); ap.add_argument("--warmup", type=int, required=True)
ap.add_argument("--decay_ratio", default="0.1"); ap.add_argument("--decay_type", default="sqrt"); ap.add_argument("--lr_min", default="0.0")
ap.add_argument("--wd", default="0.0"); ap.add_argument("--beta1", default="0.9"); ap.add_argument("--beta2", default="0.95"); ap.add_argument("--eps", default="1e-15")
ap.add_argument("--interval", type=int, required=True); ap.add_argument("--keep", type=int, default=0); ap.add_argument("--log_freq", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--wd_params", default="all", choices=["all", "matrices", "matrices_no_embed"])
a = ap.parse_args()
t = open(a.src).read()
def rep(key, val):
    global t
    n = len(re.findall(rf"(?m)^{re.escape(key)} = .*$", t))
    assert n == 1, f"{key}: {n} occurrences in {a.src}"
    t = re.sub(rf"(?m)^{re.escape(key)} = .*$", f"{key} = {val}", t)
rep("batch_size", a.micro); rep("gradient_accumulation_steps", a.accum); rep("seed", a.seed)
rep("weight_decay", a.wd); rep("wd_params", f'"{a.wd_params}"'); rep("beta1", a.beta1); rep("beta2", a.beta2); rep("eps", a.eps); rep("lr", a.lr)
rep("warmup_steps", a.warmup); rep("decay_ratio", a.decay_ratio); rep("decay_type", f'"{a.decay_type}"'); rep("lr_min", a.lr_min)
rep("interval", a.interval); rep("keep_latest_k", a.keep); rep("log_freq", a.log_freq)
open(a.dst, "w").write(t)
print(a.dst)
