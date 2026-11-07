"""Shared pieces of the evaluation scripts: loading a trained checkpoint into either implementation of the model, one
forward signature for both, and the paper's validation windows.

A run directory holds torch.distributed.checkpoint shards under <dump>/checkpoint/step-N; the model config is the JSON
the run was trained with (configs/).

  impl "trainer"  the training model (trainer/flame/custom_models/looped_transformer): the implementation every number
                  in the paper was computed with. Needs flame's dependencies and a GPU (its RMSNorm and SwiGLU are
                  Triton kernels).
  impl "port"     looped_transformer.py, plain PyTorch, CPU or GPU. Bitwise the trainer model in fp32
                  (tests/test_looped_transformer_port.py); in bf16 its validation loss agrees with the trainer's to
                  four decimals.

Validation windows follow training: documents are tokenized with the beginning-of-sequence marker (the tokenizer's
default, 64 documents per call), concatenated, and cut into seq_len-token windows; every model is scored on the same
first n_seq windows of the held-out shard.
"""
import glob
import importlib.util
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER = os.path.join(ROOT, "trainer")
IMPLS = ("trainer", "port")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dcp_state_dict(dump, step):
    """The model state dict of a distributed checkpoint, exported with PyTorch's own converter."""
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    ckpt = os.path.join(dump, f"checkpoint/step-{step}")
    assert os.path.isdir(ckpt), f"no checkpoint at {ckpt}"
    with tempfile.TemporaryDirectory() as td:
        pt = os.path.join(td, "ckpt.pt")
        dcp_to_torch_save(ckpt, pt)
        return torch.load(pt, map_location="cpu", weights_only=False)["model"]


def load_model(dump, step, config, impl="trainer"):
    """A trained checkpoint as the trainer model (impl "trainer") or the plain-PyTorch port (impl "port"), on CPU,
    in the checkpoint's fp32."""
    assert impl in IMPLS, impl
    sd = dcp_state_dict(dump, step)
    if impl == "trainer":
        if TRAINER not in sys.path:
            sys.path.insert(0, TRAINER)
        import flame.custom_models  # noqa: F401  registers the model type
        from transformers import AutoConfig, AutoModelForCausalLM

        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(config))
    else:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from looped_transformer import LoopedTransformerLM

        model = LoopedTransformerLM.from_json(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"
    assert not missing, f"missing keys: {missing[:8]}"
    return model.eval()


def is_port(model):
    return type(model).__name__ == "LoopedTransformerLM"


def forward(model, input_ids, labels=None):
    """(logits, loss) from either implementation; `labels` aligned with `input_ids` (shifted inside the model)."""
    if is_port(model):
        out = model(input_ids, labels=labels)
        return out["logits"], out["loss"]
    out = model(input_ids=input_ids, labels=labels)
    return out.logits, out.loss


def load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def bos_id(tokenizer):
    return tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1


def val_windows(data_glob, tokenizer, seq_len=4096, n_seq=1024):
    """The first n_seq windows of seq_len tokens of the held-out shards, packed as in training."""
    from datasets import load_dataset

    files = sorted(glob.glob(data_glob))
    assert files, f"no files matched {data_glob}"
    ds = load_dataset("arrow", data_files=files, split="train", streaming=True)
    text_key, buf, seqs, docs = None, [], [], []

    def flush():
        for ids in tokenizer(docs, return_attention_mask=False)["input_ids"]:
            buf.extend(ids)
        docs.clear()

    for ex in ds:
        if text_key is None:
            text_key = "text" if "text" in ex else next(k for k, v in ex.items() if isinstance(v, str))
        docs.append(ex[text_key])
        if len(docs) < 64:
            continue
        flush()
        while len(buf) >= seq_len:
            seqs.append(buf[:seq_len])
            del buf[:seq_len]
            if len(seqs) >= n_seq:
                return torch.tensor(seqs, dtype=torch.long)
    if docs:
        flush()
        while len(buf) >= seq_len and len(seqs) < n_seq:
            seqs.append(buf[:seq_len])
            del buf[:seq_len]
    assert len(seqs) >= n_seq, f"only {len(seqs)} windows of {seq_len} tokens in {data_glob}"
    return torch.tensor(seqs, dtype=torch.long)


@torch.no_grad()
def val_loss(model, seqs, device="cuda"):
    """Mean next-token cross-entropy over the windows, one sequence per forward (the paper's protocol)."""
    total = 0.0
    for i in range(seqs.shape[0]):
        ids = seqs[i:i + 1].to(device)
        total += float(forward(model, ids, labels=ids.clone())[1].item())
    return total / seqs.shape[0]
