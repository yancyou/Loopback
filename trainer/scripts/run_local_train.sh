#!/bin/bash
# Runs one training job on the GPUs of this machine: writes the job toml from TOML_TMPL (the ladder passes its
# per-arm template; the default is the paper's recipe at the 191M batch) and starts torchrun with flame.train,
# one rank per card. Arrow-shard datasets are loaded by flame/data.py.
#
# Usage (from trainer/):
#   GPUS=0,1,2,3,4,5,6,7 MODEL_CONFIG=../configs/191m_d768x20/lt124m_dense20_4klen.json STEPS=7280 bash scripts/run_local_train.sh
# Required: GPUS (comma-separated device ids), MODEL_CONFIG (model config json), STEPS (optimizer steps)
# Optional: DATA_ROOT (/data/climb_text, the directory scripts/fetch_data.py filled) and DATA ($DATA_ROOT/train/data-*.arrow,
#           the training shards), TOK (trainer/transformer-1.3B-100B, the tokenizer), DUMP (the output directory;
#           checkpoints under DUMP/checkpoint, the job toml at DUMP/train.toml, rank logs under DUMP/logs),
#           TOML_TMPL (tmpl_vanilla_lr1e3_e1x.toml.tmpl), ACMODE (activation checkpointing: none | selective | full;
#           the paper's runs used none), WANDB_MODE (offline)
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
: "${GPUS:?set GPUS=comma-separated device ids, e.g. 0,1,2,3,4,5,6,7}"
: "${MODEL_CONFIG:?set MODEL_CONFIG=path/to/model_config.json}"
: "${STEPS:?set STEPS=<optimizer steps>}"
test -f "$MODEL_CONFIG" || { echo "no such config: $MODEL_CONFIG" >&2; exit 1; }
DATA_ROOT="${DATA_ROOT:-/data/climb_text}"
DATA="${DATA:-$DATA_ROOT/train/data-*.arrow}"
TOK="${TOK:-$REPO/transformer-1.3B-100B}"
JOB="$(basename "$MODEL_CONFIG" .json)"
DUMP="${DUMP:-$HOME/experiments/$JOB}"; mkdir -p "$DUMP/logs"
NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
TOML="$DUMP/train.toml"
TOML_TMPL="${TOML_TMPL:-$REPO/tmpl_vanilla_lr1e3_e1x.toml.tmpl}"
sed -e "s|__STEPS__|$STEPS|" -e "s|__DATA__|$DATA|g" -e "s|__TOK__|$TOK|g" \
    "$TOML_TMPL" > "$TOML"
echo "[run_local_train] job=$JOB gpus=$GPUS ngpu=$NGPU steps=$STEPS config=$MODEL_CONFIG toml=$TOML"
exec torchrun --nnodes=1 --nproc_per_node="$NGPU" \
  --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 --log-dir "$DUMP/logs" \
  -m flame.train \
  --job.config_file "$TOML" \
  --job.dump_folder "$DUMP" \
  --model.config "$MODEL_CONFIG" \
  --model.tokenizer_path "$TOK" \
  --activation_checkpoint.mode "${ACMODE:-none}"
