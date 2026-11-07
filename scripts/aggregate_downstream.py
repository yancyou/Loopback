#!/usr/bin/env python3
"""The paper's downstream table rows from `scripts/eval_downstream.py` result JSONs.

Per task, the metric the paper reports, in percent with the harness's bootstrap standard error: length-normalized
accuracy (acc_norm) for HellaSwag, OpenBookQA, ARC-Easy and ARC-Challenge, accuracy (acc) for the other nine, MMLU
as the harness's "mmlu" aggregate over the 57 subjects. WikiText word perplexity is listed separately, and the last
row is the unweighted mean of the thirteen accuracy tasks (MMLU counts once; WikiText is excluded); when a JSON lacks
some of the thirteen, the mean is over the tasks it has and says so. A JSON whose "bos" field is not 1 was not
produced under the paper's protocol (BOS at the start of every context) and is flagged.

  python scripts/aggregate_downstream.py <out dir>/*.json [--csv table.csv] [--json table.json]

The screen table has one column per JSON (its "arm"); the CSV has one row per JSON.
"""
import argparse
import csv
import json
import math
import os

# (task key in the harness, metric, label); the order of the paper's table
TASKS = [("piqa", "acc", "PIQA"), ("hellaswag", "acc_norm", "HellaSwag"), ("winogrande", "acc", "WinoGrande"),
         ("openbookqa", "acc_norm", "OpenBookQA"), ("commonsense_qa", "acc", "CommonsenseQA"), ("boolq", "acc", "BoolQ"),
         ("race", "acc", "RACE"), ("arc_easy", "acc_norm", "ARC-Easy"), ("arc_challenge", "acc_norm", "ARC-Challenge"),
         ("mmlu", "acc", "MMLU"), ("anli_r1", "acc", "ANLI-R1"), ("rte", "acc", "RTE"), ("lambada_openai", "acc", "LAMBADA")]
WIKITEXT = ("wikitext", "word_perplexity", "WikiText word ppl")


def summarize(path):
    """One JSON -> {"arm", "bos", "step", "tasks": {key: (percent, stderr percent) | None}, "wikitext", "mean", "n_mean"}."""
    d = json.load(open(path))
    res = d["results"]
    tasks = {}
    for key, metric, _ in TASKS:
        r = res.get(key)
        if r is None or f"{metric},none" not in r:
            tasks[key] = None
            continue
        se = r.get(f"{metric}_stderr,none")
        tasks[key] = (100.0 * r[f"{metric},none"], None if se is None or (isinstance(se, float) and math.isnan(se)) else 100.0 * se)
    wiki = res.get(WIKITEXT[0], {}).get(f"{WIKITEXT[1]},none")
    have = [v[0] for v in tasks.values() if v is not None]
    return dict(path=path, arm=d.get("arm") or os.path.splitext(os.path.basename(path))[0], bos=d.get("bos"),
                step=d.get("step"), limit=d.get("limit"), tasks=tasks, wikitext=wiki,
                mean=sum(have) / len(have) if have else None, n_mean=len(have))


def cell(v):
    if v is None:
        return "--"
    if isinstance(v, tuple):
        return f"{v[0]:.2f}" if v[1] is None else f"{v[0]:.2f} ± {v[1]:.2f}"
    return f"{v:.2f}"


def print_table(rows):
    w = max(12, *(len(r["arm"]) for r in rows))
    print(f"{'':<20}" + "".join(f"{r['arm']:>{w + 2}}" for r in rows))
    for key, _, label in TASKS:
        print(f"{label:<20}" + "".join(f"{cell(r['tasks'][key]):>{w + 2}}" for r in rows))
    print(f"{WIKITEXT[2]:<20}" + "".join(f"{cell(r['wikitext']):>{w + 2}}" for r in rows))
    means = [cell(r["mean"]) + ("" if r["n_mean"] == len(TASKS) else f" ({r['n_mean']}/13)") for r in rows]
    print(f"{'Mean (13 tasks)':<20}" + "".join(f"{m:>{w + 2}}" for m in means))
    for r in rows:
        notes = []
        if r["bos"] != 1:
            notes.append(f"bos={r['bos']}: not the paper's protocol (BOS at the start of every context)")
        if r["limit"]:
            notes.append(f"limit={r['limit']}: a subset of every task")
        if r["n_mean"] != len(TASKS):
            notes.append(f"mean over {r['n_mean']} of the 13 tasks")
        if notes:
            print(f"note {r['arm']}: " + "; ".join(notes))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jsons", nargs="+", help="result JSONs of scripts/eval_downstream.py")
    ap.add_argument("--csv", default="", help="write one row per JSON (value and stderr columns)")
    ap.add_argument("--json", default="", help="write the summaries as JSON")
    a = ap.parse_args()
    rows = [summarize(p) for p in a.jsons]
    print_table(rows)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "bos", "step"] + [x for k, _, _ in TASKS for x in (k, k + "_stderr")]
                       + ["wikitext_word_ppl", "mean13", "mean_n_tasks"])
            for r in rows:
                vals = []
                for k, _, _ in TASKS:
                    v = r["tasks"][k]
                    vals += ["", ""] if v is None else [f"{v[0]:.4f}", "" if v[1] is None else f"{v[1]:.4f}"]
                w.writerow([r["arm"], r["bos"], r["step"]] + vals
                           + ["" if r["wikitext"] is None else f"{r['wikitext']:.4f}",
                              "" if r["mean"] is None else f"{r['mean']:.4f}", r["n_mean"]])
        print(f"wrote {a.csv}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=2)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
