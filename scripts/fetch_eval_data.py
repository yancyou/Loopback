#!/usr/bin/env python3
"""The public evaluation files of `scripts/eval_math_code.py`, downloaded into the layout it reads, with the item
counts of the paper checked by its own loaders.

  python scripts/fetch_eval_data.py --data <dir> [--force]

Word problems: ToRA's evaluation data on GitHub (microsoft/ToRA, src/data/{mawps,asdiv,svamp}/test.jsonl).
mawps/test.jsonl is split by its "type" field into tora_mawps_{singleop,singleeq,addsub,multiarith}.jsonl (the
"input" and "target" fields, in file order: 562 / 508 / 395 / 600 problems); asdiv/test.jsonl (2,215; 2,084 with a
numeric answer) and svamp/test.jsonl (1,000) are stored as they are. Reference solutions, from the Hugging Face hub
through `datasets`: openai/gsm8k (main, test; 1,319) -> gsm8k_test.parquet, HuggingFaceH4/MATH-500 (test; 500) ->
math500_test.jsonl, openai/openai_humaneval (test; 164) -> humaneval_test.parquet, google-research-datasets/mbpp
(sanitized, test; 257) -> mbpp_sanitized_test.parquet. Files already present are kept unless --force. Needs network
access to raw.githubusercontent.com and huggingface.co; no token.
"""
import argparse
import json
import os
import sys
import urllib.request

TORA = "https://raw.githubusercontent.com/microsoft/ToRA/main/src/data/{}/test.jsonl"
MAWPS_TYPES = ("singleop", "singleeq", "addsub", "multiarith")
HF = [  # (file, repo, config, split)
    ("gsm8k_test.parquet", "openai/gsm8k", "main", "test"),
    ("math500_test.jsonl", "HuggingFaceH4/MATH-500", None, "test"),
    ("humaneval_test.parquet", "openai/openai_humaneval", None, "test"),
    ("mbpp_sanitized_test.parquet", "google-research-datasets/mbpp", "sanitized", "test"),
]
EXPECTED = dict(singleop=562, singleeq=508, addsub=395, multiarith=600, asdiv=2084, svamp=1000,
                gsm8k=1319, math500=500, humaneval=164, mbpp=257)
EXPECTED_ASDIV_SKIPPED = 2215 - 2084


def fetch(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def tora(d, force):
    out = [os.path.join(d, f"tora_mawps_{t}.jsonl") for t in MAWPS_TYPES]
    if force or not all(os.path.exists(p) for p in out):
        rows = [json.loads(l) for l in fetch(TORA.format("mawps")).decode("utf-8").splitlines() if l.strip()]
        for t, p in zip(MAWPS_TYPES, out):
            with open(p, "w") as f:
                for r in rows:
                    if r["type"] == t:
                        f.write(json.dumps({"input": r["input"], "target": r["target"]}) + "\n")
            print(f"wrote {p}")
    for name in ("asdiv", "svamp"):
        p = os.path.join(d, f"tora_{name}_test.jsonl")
        if force or not os.path.exists(p):
            with open(p, "wb") as f:
                f.write(fetch(TORA.format(name)))
            print(f"wrote {p}")


def hub(d, force):
    from datasets import load_dataset

    for file, repo, config, split in HF:
        p = os.path.join(d, file)
        if not force and os.path.exists(p):
            continue
        ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
        if file.endswith(".jsonl"):
            with open(p, "w") as f:
                for r in ds:
                    f.write(json.dumps(r) + "\n")
        else:
            ds.to_pandas().to_parquet(p, index=False)
        print(f"wrote {p} ({len(ds)} rows)")


def check(d):
    """Read the files with eval_math_code.py's loaders and compare the item counts with the paper's."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eval_math_code as E

    sets, skipped = E.load_mwp(d)
    counts = {k: len(v) for k, v in sets.items()} | {k: len(v) for k, v in E.load_solutions(d).items()}
    bad = {k: (counts.get(k), n) for k, n in EXPECTED.items() if counts.get(k) != n}
    if skipped != EXPECTED_ASDIV_SKIPPED:
        bad["asdiv_skipped_non_numeric"] = (skipped, EXPECTED_ASDIV_SKIPPED)
    for k in EXPECTED:
        print(f"  {k:11s} {counts.get(k):5d}  (expected {EXPECTED[k]})")
    print(f"  asdiv items skipped for a non-numeric answer: {skipped} (expected {EXPECTED_ASDIV_SKIPPED})")
    if bad:
        raise SystemExit(f"item counts differ from the paper's: {bad}")
    print(f"all item counts match the paper's; data in {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="destination directory (the --data of scripts/eval_math_code.py)")
    ap.add_argument("--force", action="store_true", help="download again even when a file is present")
    a = ap.parse_args()
    os.makedirs(a.data, exist_ok=True)
    tora(a.data, a.force)
    hub(a.data, a.force)
    check(a.data)


if __name__ == "__main__":
    main()
