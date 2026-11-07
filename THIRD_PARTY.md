# Third-party code

This project's own code is under the MIT licence in `LICENSE`. The parts below come from other projects and keep their
licences; the full texts are in `licenses/`.

| files | origin | licence |
|---|---|---|
| `trainer/flame/` | a trimmed copy of [fla-org/flame](https://github.com/fla-org/flame), the torchtitan-based trainer of flash-linear-attention | MIT, Copyright (c) 2023-2025 Songlin Yang, Yu Zhang (`licenses/flame-MIT.txt`) |
| `trainer/flame/train.py`, `trainer/flame/config_manager.py`, `trainer/flame/components/checkpoint.py`, `trainer/flame/models/parallelize_fla.py`, `trainer/flame/tools/utils.py` | files that flame took from [pytorch/torchtitan](https://github.com/pytorch/torchtitan); the "LICENSE file in the root directory" named in their headers is torchtitan's | BSD-3-Clause, (c) Meta Platforms, Inc. and affiliates (`licenses/torchtitan-BSD-3-Clause.txt`) |
| `trainer/flame/custom_models/looped_transformer/rotary.py` | the rotary embedding of [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | MIT, Copyright (c) 2023-2025 Songlin Yang (`licenses/flash-linear-attention-MIT.txt`) |

This project's additions inside `trainer/` are the `looped_transformer` model
(`trainer/flame/custom_models/looped_transformer/`, apart from `rotary.py`), an exact document-level data resume
(`trainer/flame/data.py`) and the WSD ladder driver `trainer/queue_wsd_ladder.sh`.

- At run time the trainer uses [pytorch/torchtitan](https://github.com/pytorch/torchtitan) (BSD-3-Clause) and
  [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT) as installed packages.
- The tokenizer used by every run is the Llama 32k tokenizer published with fla's
  [`fla-hub/transformer-1.3B-100B`](https://huggingface.co/fla-hub/transformer-1.3B-100B) checkpoint; it is downloaded,
  not redistributed here (see README, "Data and tokenizer").
- The fused read kernel (`fused_dattn.py`) is written with [Triton](https://github.com/triton-lang/triton) (MIT).
