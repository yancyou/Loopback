#!/usr/bin/env python3
"""Validation loss of a trained checkpoint on the held-out ClimbMix shard (DCP-direct load, single GPU).

Builds the model from its config, loads the torch distributed checkpoint (<dump>/checkpoint/step-<step>) directly,
and reports the mean token cross-entropy (val loss) and perplexity over n_seq held-out windows packed exactly as the
training packer packs them. The same windows are used for every arm, so the numbers are directly comparable.

Usage:
  python scripts/eval_val_loss.py --dump <exp_dir> --step 7280 --config <model.json> \
    --tokenizer transformer-1.3B-100B --data_glob "<DATA_ROOT>/val/*.arrow" \
    --n_seq 1024 --seq_len 4096 --out val_<arm>.json
"""
import argparse, glob, json, math, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # trainer/, for flame.custom_models


def load_model_from_dcp(dump, step, config_path):
    import tempfile
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    from transformers import AutoConfig, AutoModelForCausalLM
    import flame.custom_models  # noqa: F401  registers model_type looped_transformer
    cfg = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_config(cfg)
    ckpt = os.path.join(dump, f"checkpoint/step-{step}")
    with tempfile.TemporaryDirectory() as td:
        pt = os.path.join(td, "ckpt.pt")
        dcp_to_torch_save(ckpt, pt)
        sd = torch.load(pt, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    param_names = {name for name, _ in model.named_parameters()}
    missing_params = [k for k in missing if k in param_names]
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"
    assert not missing_params, f"missing parameters: {missing_params[:8]}"
    return model


def build_val_sequences(data_glob, tokenizer, seq_len, n_seq, packing="train"):
    """Pack the validation stream. packing="train" (the default) reproduces flame's
    OnlineTokenizedIterableDataset.tokenize exactly: documents are tokenised in batches of 64 with
    add_special_tokens=True, so each one starts with the tokenizer's BOS, and the concatenated stream is cut into
    seq_len windows. packing="eval" concatenates the documents with no separator at all, an input no model sees in
    training; it inflates every loss by an architecture-dependent amount and is not comparable across arms.
    """
    from datasets import load_dataset
    assert packing in ("train", "eval"), packing
    files = sorted(glob.glob(data_glob))
    assert files, f"no files matched {data_glob}"
    ds = load_dataset("arrow", data_files=files, split="train", streaming=True)
    text_key, buf, seqs, docs = None, [], [], []

    def flush(docs):
        # one tokenizer call per batch of documents, exactly as the training packer does
        for ids in tokenizer(docs, return_attention_mask=False)["input_ids"]:
            buf.extend(ids)

    for ex in ds:
        if text_key is None:
            text_key = "text" if "text" in ex else next(k for k, v in ex.items() if isinstance(v, str))
        if packing == "train":
            docs.append(ex[text_key])
            if len(docs) < 64:
                continue
            flush(docs); docs = []
        else:
            buf.extend(tokenizer(ex[text_key], add_special_tokens=False).input_ids)
        while len(buf) >= seq_len:
            seqs.append(buf[:seq_len]); del buf[:seq_len]
            if len(seqs) >= n_seq:
                return torch.tensor(seqs, dtype=torch.long)
    if docs:
        flush(docs)
        while len(buf) >= seq_len and len(seqs) < n_seq:
            seqs.append(buf[:seq_len]); del buf[:seq_len]
    assert seqs, "not enough tokens"
    return torch.tensor(seqs, dtype=torch.long)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--config", required=True); ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data_glob", required=True); ap.add_argument("--n_seq", type=int, default=1024)
    ap.add_argument("--seq_len", type=int, default=4096); ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="")
    ap.add_argument("--packing", choices=("train", "eval"), default="train",
                    help="how the validation stream is packed; train = same as the training packer (default)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model_from_dcp(args.dump, args.step, args.config).to(torch.bfloat16).cuda().eval()
    seqs = build_val_sequences(args.data_glob, tok, args.seq_len, args.n_seq, args.packing)

    tot, n = 0.0, 0
    for i in range(seqs.shape[0]):
        ids = seqs[i:i + 1].cuda()
        out = model(ids, labels=ids.clone())
        tot += float(out.loss.item()); n += 1
    val = tot / n
    res = dict(arm=args.arm, dump=args.dump, n_seq=int(seqs.shape[0]),
               seq_len=args.seq_len, val_loss=val, ppl=math.exp(val), packing=args.packing)
    print(f"[val] {args.arm:10s} val_loss={val:.4f} ppl={math.exp(val):.3f} (n={n})", flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
