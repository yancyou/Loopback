#!/usr/bin/env python3
"""Training and validation text in the layout the trainer reads: Arrow shards with `text` and `cluster_id` columns under
<root>/train/ and <root>/val/, sampled from the public text release of ClimbMix.

The paper trained on ClimbMix (Diao et al., 2025). NVIDIA publishes it as GPT-2 token ids (nvidia/Nemotron-ClimbMix,
with a detokenization script); OptimalScale/ClimbMix is a community copy detokenized to text: 100 jsonl parts, 1.69 TB,
about 386B Llama tokens. The parts are sorted by topic cluster (part 0 is cluster 1, part 99 clusters 19 and 20), so a
few whole parts would cover a few topics only. The paper's runs used a private re-sharding of the whole corpus into
3,584 shards, each a uniform sample of every cluster (shards 0-239, 6.7% of the corpus, for 191M-778M; 0-559 for 1.4B
and 3B; shard 3583 for validation). This script draws the same kind of sample: it reads --fraction of every part, as
--chunks evenly spaced byte ranges (only that fraction is downloaded), deals the documents at random into training
shards of about 150,000 documents, shuffles each shard, and holds out a separate shuffled validation shard read from
just after each training range. The result has the corpus's cluster mixture (printed at the end); its documents are a
different sample than the paper's, so a run on it agrees with the paper to within run-to-run variation (the seed
table in results/README.md).

    python scripts/fetch_data.py --root /data/climb_text                    # --fraction 0.067: 113 GB read, about 26B
                                                                            # Llama tokens: 191M-778M at 20 TPP
    python scripts/fetch_data.py --root /data/climb_text --fraction 0.17    # 288 GB, about 66B: 1.4B (28.3B tokens)
                                                                            # and 3B (59.8B)
Then DATA_ROOT=/data/climb_text for the ladder (train/data-*.arrow, val/data-val.arrow). The sample is kept per part
under <root>/download/ until the shards are written, so an interrupted run resumes with the parts it has not finished.
"""
import argparse
import collections
import json
import math
import os
import resource
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

REPO = "OptimalScale/ClimbMix"
SCHEMA = pa.schema([("text", pa.string()), ("cluster_id", pa.int32())])
SHARD_BYTES = 470e6        # jsonl bytes per training shard: about 150,000 documents (the paper's shards hold ~154,000)
LLAMA_PER_BYTE = 0.228     # Llama tokens (with <s>) per jsonl byte, measured on 44,400 documents of six parts


def batch(docs):
    return pa.record_batch([pa.array([t for t, _ in docs], pa.string()), pa.array([c for _, c in docs], pa.int32())],
                           schema=SCHEMA)


def write_arrow(path, table):
    tmp = path + ".tmp"
    with pa.OSFile(tmp, "wb") as f, ipc.new_stream(f, SCHEMA) as w:
        w.write_table(table)
    os.replace(tmp, path)


def read_arrow(path):
    with pa.memory_map(path) as f:
        return ipc.open_stream(f).read_all()


def fetch(fs, path, start, end, tries=8):
    for k in range(tries):
        try:
            return fs.cat_file(path, start=start, end=end)
        except Exception as e:  # network hiccups: retry with backoff
            if k == tries - 1:
                raise
            print(f"  retry {path} [{start}, {end}): {e!r}", flush=True)
            time.sleep(2 ** k)


def read_window(fs, path, size, start, n_train, n_val):
    """The documents whose first byte lies in [start, start + n_train) (training) and in the next n_val bytes
    (validation); a line straddling a boundary belongs to the range holding its first byte."""
    lo = max(start - 1, 0)                       # one byte back: is `start` itself the beginning of a line?
    buf = fetch(fs, path, lo, min(start + n_train + n_val + (1 << 20), size))

    def extend():
        nonlocal buf
        if lo + len(buf) >= size:
            return False
        buf += fetch(fs, path, lo + len(buf), min(lo + len(buf) + (1 << 22), size))
        return True

    if start == 0:
        pos = 0
    else:
        while b"\n" not in buf:
            if not extend():
                return [], []
        pos = buf.index(b"\n") + 1
    t_end, v_end = start + n_train - lo, start + n_train + n_val - lo
    train, val = [], []
    while pos < v_end and lo + pos < size:
        nl = buf.find(b"\n", pos)
        while nl < 0 and extend():               # finish a long last line
            nl = buf.find(b"\n", pos)
        line = buf[pos:nl if nl >= 0 else len(buf)]
        if line.strip():
            d = json.loads(line)
            (train if pos < t_end else val).append((d["text"], int(d["cluster_id"])))
        if nl < 0:
            break
        pos = nl + 1
    return train, val


def download_part(fs, name, size, a, out):
    """Sample one part into <out>/<name>.arrow (training documents) and <name>.val.arrow; skipped if both exist."""
    tr_path, va_path = os.path.join(out, name + ".arrow"), os.path.join(out, name + ".val.arrow")
    if os.path.exists(tr_path) and os.path.exists(va_path):
        return
    path = f"datasets/{REPO}/{name}"
    step = size / a.chunks
    n_train, n_val = int(a.fraction * step), int(a.val_fraction * step)
    with pa.OSFile(tr_path + ".tmp", "wb") as ft, ipc.new_stream(ft, SCHEMA) as wt, \
         pa.OSFile(va_path + ".tmp", "wb") as fv, ipc.new_stream(fv, SCHEMA) as wv, \
         ThreadPoolExecutor(a.workers) as pool:
        for train, val in pool.map(lambda j: read_window(fs, path, size, int(j * step), n_train, n_val),
                                   range(a.chunks)):
            if train:
                wt.write_batch(batch(train))
            if val:
                wv.write_batch(batch(val))
    os.replace(va_path + ".tmp", va_path)
    os.replace(tr_path + ".tmp", tr_path)


def mixture(counter):
    n = sum(counter.values())
    return " ".join(f"{c}:{100 * k / n:.1f}%" for c, k in sorted(counter.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fraction", type=float, default=0.067, help="share of every part read for training")
    ap.add_argument("--val_fraction", type=float, default=0.00028, help="share read for validation (~150,000 docs)")
    ap.add_argument("--chunks", type=int, default=64, help="evenly spaced byte ranges per part")
    ap.add_argument("--workers", type=int, default=16, help="parallel range requests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_downloads", action="store_true", help="keep <root>/download/ after writing the shards")
    a = ap.parse_args()
    assert 0 < a.fraction and 0 < a.val_fraction and a.fraction + a.val_fraction <= 1
    from huggingface_hub import HfApi, HfFileSystem

    sizes = {f.path: f.size for f in HfApi().list_repo_tree(REPO, repo_type="dataset")
             if f.path.startswith("part_") and f.path.endswith(".jsonl")}
    names = sorted(sizes, key=lambda p: int(p[5:-6]))
    total = sum(sizes.values())
    print(f"{REPO}: {len(names)} parts, {total / 1e12:.2f} TB; reading {a.fraction:g} of every part for training "
          f"({a.fraction * total / 1e9:.1f} GB, about {a.fraction * total * LLAMA_PER_BYTE / 1e9:.1f}B Llama tokens) "
          f"and {a.val_fraction:g} for validation", flush=True)
    dl, train_dir, val_dir = (os.path.join(a.root, d) for d in ("download", "train", "val"))
    for d in (dl, train_dir, val_dir):
        os.makedirs(d, exist_ok=True)
    fs = HfFileSystem()
    for i, name in enumerate(names):
        t0 = time.time()
        download_part(fs, name, sizes[name], a, dl)
        print(f"[{i + 1}/{len(names)}] {name} sampled ({time.time() - t0:.0f}s)", flush=True)

    # deal the training documents into shards at random (one open file per shard), then shuffle each shard
    n_shards = max(1, math.ceil(a.fraction * total / SHARD_BYTES))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < n_shards + 64:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard if hard == resource.RLIM_INFINITY else min(hard, max(soft, n_shards + 64)), hard))
    deal = [os.path.join(train_dir, f"deal-{s:05d}.arrow") for s in range(n_shards)]
    sinks = [pa.OSFile(p, "wb") for p in deal]
    writers = [ipc.new_stream(s, SCHEMA) for s in sinks]
    train_mix = collections.Counter()
    for i, name in enumerate(names):
        t = read_arrow(os.path.join(dl, name + ".arrow"))
        train_mix.update(t.column("cluster_id").to_pylist())
        dest = np.random.default_rng([a.seed, i]).integers(n_shards, size=t.num_rows)
        order = np.argsort(dest, kind="stable")
        bounds = np.searchsorted(dest[order], np.arange(n_shards + 1))
        for s in range(n_shards):
            if bounds[s + 1] > bounds[s]:
                writers[s].write_table(t.take(order[bounds[s]:bounds[s + 1]]))
    for w, s in zip(writers, sinks):
        w.close()
        s.close()
    for s, p in enumerate(deal):
        t = read_arrow(p)
        write_arrow(os.path.join(train_dir, f"data-{s:05d}.arrow"),
                    t.take(np.random.default_rng([a.seed, 1 << 20, s]).permutation(t.num_rows)))
        os.remove(p)

    v = pa.concat_tables([read_arrow(os.path.join(dl, name + ".val.arrow")) for name in names])
    v = v.take(np.random.default_rng([a.seed, 1 << 21]).permutation(v.num_rows))
    write_arrow(os.path.join(val_dir, "data-val.arrow"), v)
    val_mix = collections.Counter(v.column("cluster_id").to_pylist())
    if not a.keep_downloads:
        shutil.rmtree(dl)
    print(f"{n_shards} training shards, {sum(train_mix.values()):,} documents, in {train_dir}")
    print(f"validation: {v.num_rows:,} documents in {val_dir}/data-val.arrow")
    print(f"cluster mixture (training): {mixture(train_mix)}")
    print(f"cluster mixture (validation): {mixture(val_mix)}", flush=True)


if __name__ == "__main__":
    main()
