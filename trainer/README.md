# trainer/ — the flame trainer the paper's runs executed

A trimmed copy of [fla-org/flame](https://github.com/fla-org/flame) (torchtitan + flash-linear-attention, MIT),
reduced to what the paper's training runs used:

- `flame/custom_models/looped_transformer/` — the training model as a HuggingFace custom model (model type
  `looped_transformer`: Standard, Plain Loop, Loopback and the loop-axis readers): `modeling_looped_transformer.py`, `configuration_looped_transformer.py`, `rotary.py`, `fused_dattn.py`
  (the fused Triton read, `LOOP_DATTN_FUSED=1`), `loop_linattn_delta.py` / `loop_linattn_state.py` (the readers;
  byte-identical to the copies at the repository root, checked by `tests/test_vendored_copies.sh`). The pure-PyTorch
  `looped_transformer.py` at the repository root is checked against this model bitwise
  (`tests/test_looped_transformer_port.py`).
- `flame/train.py` — the training loop (`torchrun -m flame.train`): FSDP2 with bf16 parameters and fp32 gradient
  reductions, gradient accumulation, global-norm clipping, torchtitan's warmup-stable-decay schedule, checkpoints on
  a fixed interval. `flame/config_manager.py` holds every option (toml sections, overridden by the command line);
  `flame/components/optimizer.py` builds AdamW (`optimizer.wd_params` selects the parameters weight decay applies
  to; the paper's runs used weight decay 0); `flame/models/parallelize_fla.py` applies FSDP (the looped blocks,
  whose parameters are shared, stay in the root FSDP unit), activation checkpointing and `torch.compile`;
  `flame/checkpoint_manager.py` and `flame/components/checkpoint.py` save and load the torch distributed
  checkpoints (model, optimizer, schedule, dataloader and train state).
- `flame/data.py` — the online tokenizer and packer (the documents of the shuffled shards in order, each starting
  with the tokenizer's `<s>`, concatenated and cut into 4,096-token sequences) with a document-exact resume: a
  resumed run continues at the document it stopped at on every worker, which is what lets a ladder rung fork from a
  checkpoint and reproduce the uninterrupted run.
- `queue_wsd_ladder.sh` — the WSD tokens-per-parameter ladder driver: one call trains every arm of a table to the
  requested tokens-per-parameter budgets, later rungs forking from the previous rung's last pre-decay checkpoint,
  and evaluates each finished rung; its header documents every variable and the settings of every table.
- `scripts/run_local_train.sh` — the torchrun launcher the ladder calls (it also runs a single arm);
  `scripts/mk_wsd_tmpl.py` — derives each arm's job toml from `tmpl_vanilla_lr1e3_e1x.toml.tmpl`, the recipe
  template; `scripts/eval_val_loss.py` — the validation evaluator the ladder runs at the end of every rung.
- `transformer-1.3B-100B/` — the Llama 32k tokenizer, filled by `scripts/fetch_tokenizer.sh` at the repository
  root (the scripts here look for it at this path by default).

## The recipe

Shared by every run: AdamW with β = (0.9, 0.95), ε = 1e-15, weight decay 0, gradient clipping at global norm 1.0,
sequence length 4,096, seed 42, linear warmup over 128 steps, the peak learning rate held, the last 10 % of the
steps decayed to 0 along 1 − √u (`decay_type = "sqrt"`, `decay_ratio = 0.1`), no activation checkpointing
(`ACMODE=none`), no `torch.compile`, 8 GPUs on one node, FSDP with bf16 parameters and fp32 reductions. The budget
is 20 tokens per parameter (TPP), counted on `NREF` = the Standard model's total parameters and rounded to a
20-step grid by the ladder. Only the peak learning rate differs between the arms of a table.

| scale | micro × accum per GPU | tokens / step | steps | tokens | peak lr Standard / looped arms | `NREF` | training text | `TPP_UNIT`, `TPPS` |
|---|---|---|---|---|---|---|---|---|
| 191M | 16 × 1 | 524,288 | 7,280 | 3.82B | 1e-3 / 3e-3 | 190,741,248 | ~26B tokens (`fetch_data.py`) | 20, `"20"` |
| 374M | 8 × 4 | 1,048,576 | 7,120 | 7.47B | 2e-3 / 2e-3 | 373,867,520 | ~26B tokens | 5, `"5 20"` |
| 778M | 8 × 8 | 2,097,152 | 7,440 | 15.60B | 1e-3 / 1.5e-3 | 777,856,512 | ~26B tokens | 5, `"5 20"` |
| 1.4B | 8 × 15 | 3,932,160 | 7,200 | 28.31B | 1e-3 / 1.5e-3 | 1,415,682,048 | ~66B tokens (`--fraction 0.17`) | 20, `"20"` |
| 3B | 4 × 60 | 7,864,320 | 7,600 | 59.77B | 1e-3 / 1e-3 | 2,914,667,520 | ~66B tokens (`--fraction 0.17`) | 1, `"1 20"` |

The paper's runs read shards 0–239 (191M–778M) or 0–559 (1.4B and 3B) of a private 3,584-shard split of ClimbMix and
validated on its shard 3583; `scripts/fetch_data.py` samples the public text release into shards of the same format
and cluster mixture, with other documents (see the repository README).

The ladder computes the step count as T = round(U · NREF / tokens-per-step / 20) · 20 · (20 / U) with U = `TPP_UNIT`,
so the unit fixes the exact number of steps: 374M and 778M were screened at 5 TPP and their 20-TPP run forked from
that rung (`TPP_UNIT=5 TPPS="20"` trains the identical 7,120 / 7,440-step run as a single rung); 3B was run as a
1-TPP rung (380 steps) plus the fork to 20 TPP (`TPP_UNIT=1 TPPS="20"` gives the same 7,600 steps); `TPP_UNIT=20`
would give 7,140 / 7,420 / 7,420 steps at 374M / 778M / 3B. Within a table every arm reads the same tokens in the
same order; the document order is a function of the shard list and of the number of data-parallel ranks.

## Environment

The paper's runs used Python 3.10, torch 2.9 / 2.10, flash-attn 2.8.3, flash-linear-attention 0.4.1, torchtitan at
commit 0b44d4c, transformers 5.x and datasets 3.4 / 5.0, plus `torchdata` (the stateful dataloader) and `tomli` on
Python < 3.11. Install torch with CUDA, then:

```bash
pip install flash-linear-attention==0.4.1 flash-attn==2.8.3 transformers datasets huggingface-hub torchdata tomli
pip install git+https://github.com/pytorch/torchtitan@0b44d4c
bash   scripts/fetch_tokenizer.sh                           # from the repository root: -> trainer/transformer-1.3B-100B/
python scripts/fetch_data.py --root $DATA_ROOT                  # ClimbMix text, ~26B tokens -> $DATA_ROOT/train/data-*.arrow, $DATA_ROOT/val/
python scripts/fetch_data.py --root $DATA_ROOT --fraction 0.17  # the 1.4B / 3B budget, ~66B tokens
```

Every command below is run from `trainer/`; the launcher puts `trainer/` on `PYTHONPATH` itself.

## One table: the ladder

```bash
SELF_TAG=t191 NREF=190741248 MICRO=16 ACCUM=1 NGPU=8 TPPS="20" ORDER=arm LOOP_DATTN_FUSED=1 \
DATA_ROOT=$DATA_ROOT OUT_DIR=$OUT_DIR \
ARMS="standard:../configs/191m_d768x20/lt124m_dense20_4klen.json:1e-3 \
      plain_loop:../configs/191m_d768x20/lt124m_plain_p2k4l4c2_4klen.json:3e-3 \
      loopback:../configs/191m_d768x20/lt124m_ours_p2k4l4c2_4klen.json:3e-3" \
bash queue_wsd_ladder.sh
```

`ARMS` lists `name:config:peak-lr` triples; `MICRO`, `ACCUM`, `NREF`, `TPP_UNIT`, `TPPS` and `DATA` come from the
table above (374M: `MICRO=8 ACCUM=4 TPP_UNIT=5 TPPS="5 20" NREF=373867520`; 1.4B: `MICRO=8 ACCUM=15 NREF=1415682048`,
and so on; `DATA` defaults to `$DATA_ROOT/train/data-*.arrow`). For every arm the driver writes a per-arm template, checks it
line by line against the recipe, runs a 20-step smoke test, then trains each rung (resuming its own newest
checkpoint after a crash, forking the next rung from the pre-decay checkpoint `step-F_k`), keeps only `step-F_k`
and the decayed endpoint `step-T_k`, and evaluates the endpoint on the held-out shard in the background on
`FIRST_GPU`. Per rung it leaves `$OUT_DIR/wsd_<arm>_tpp<k>/` (checkpoints), `wsd_<arm>_tpp<k>.log`,
`wsd_<arm>_tpp<k>_climbval.json` and a row of `$OUT_DIR/climb4k_wsd_<SELF_TAG>_results.md`; the log line
`RESULT <arm> rung <k> ... climb=<loss>` repeats the row. A rung whose val json exists is skipped, so a call can be
repeated to resume a table or to add rungs. `LOOP_DATTN_FUSED=1` selects Loopback's fused Triton read (what the
paper's throughput numbers use); without it the reference read runs.

## One arm

```bash
GPUS=0,1,2,3,4,5,6,7 MODEL_CONFIG=../configs/191m_d768x20/lt124m_ours_p2k4l4c2_4klen.json STEPS=7280 \
DATA_ROOT=$DATA_ROOT DUMP=$OUT_DIR/loopback_191m bash scripts/run_local_train.sh
```

The default template `tmpl_vanilla_lr1e3_e1x.toml.tmpl` is the recipe at the 191M batch (16 × 1 per GPU, peak lr
1e-3, a checkpoint every 2,184 steps). Another batch, learning rate or checkpoint interval is another template:
`python scripts/mk_wsd_tmpl.py --src tmpl_vanilla_lr1e3_e1x.toml.tmpl --dst my.toml.tmpl --micro 16 --accum 1
--lr 3e-3 --warmup 128 --interval 2184`, then `TOML_TMPL=my.toml.tmpl`. The job toml is written to `DUMP/train.toml`;
a restarted run resumes from the newest checkpoint in `DUMP/checkpoint`.

## The evaluator

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_val_loss.py --dump $OUT_DIR/wsd_loopback_tpp20 --step 7280 \
  --config ../configs/191m_d768x20/lt124m_ours_p2k4l4c2_4klen.json --tokenizer transformer-1.3B-100B \
  --data_glob "$DATA_ROOT/val/*.arrow" --out val.json
```

It loads the torch distributed checkpoint `<dump>/checkpoint/step-<step>` into the model, packs 1,024 windows of
4,096 tokens from the held-out shard exactly as the training packer does (`--n_seq`, `--seq_len`), and writes the
mean token cross-entropy and perplexity to the json (bf16, one GPU). `scripts/eval_climb_val.py` at the repository
root computes the same number with the pure-PyTorch model.
