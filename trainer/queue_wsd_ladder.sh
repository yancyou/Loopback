#!/usr/bin/env bash
# queue_wsd_ladder.sh: the WSD tokens-per-parameter ladder. One call trains every arm of a table to the requested
# tokens-per-parameter (TPP) budgets and evaluates each finished rung on the held-out ClimbMix shard.
#
# One RUNG per (arm, TPP k) = a complete WSD run to T_k optimizer steps: linear warmup WARMUP steps, the peak lr
# held, the last D_k = round(DECAY_RATIO * T_k) steps decayed to LR_MIN (torchtitan's linear_warmup_stable_decay,
# flame [lr_scheduler] decay_ratio / decay_type / lr_min). Rung k > first FORKS from rung k-1's checkpoint at
# F_{k-1} = T_{k-1} - D_{k-1}: every step up to F_{k-1} ran at the identical (warmup, peak) lr in both schedules, the
# checkpoint carries the lr the next step uses (peak), the optimizer state and the dataloader position, so rung k is
# step for step the uninterrupted WSD run to T_k.
#   * TPP is counted on NREF parameters (one number for every arm: the ladder is iso-token, and for iso-depth arms
#     iso-FLOP); T_k = k * T_U / U with U = TPP_UNIT and T_U = the multiple of 20 steps nearest U*NREF/TOKSTEP, so
#     DECAY_RATIO 0.1 gives integer D_k and every fork F_k is a multiple of F_U.
#   * checkpoint interval = the largest divisor of F_U not above IV_MAX, so every fork step is a regular interval
#     save (the dataloader snapshot interval is the same number, so its state there is exact); keep_latest_k = KEEP
#     while a rung trains (a crash resumes from the newest save); when a rung is complete only step-F_k (the next
#     rung's fork) and step-T_k (the decayed model) are kept.
#   * requires flame/data.py with the document-exact resume (hf_table_start): a branch would otherwise skip data at
#     every fork.
#
# Variables (defaults = the paper's recipe):
#   SELF_TAG  a short name for this table; it tags every log line, the templates and the results table (required)
#   ARMS      "name:config.json:lr ..." (config paths relative to trainer/; required). Arm names must be unique per
#             lr/seed: a later call with the same arm name continues that arm's ladder from its last fork.
#   NREF      the parameter count TPP is counted on: the Standard model's total parameters (required)
#   TPPS      rungs, ascending multiples of TPP_UNIT (default "20"); a rung whose val json exists is skipped
#   TPP_UNIT  the rung grid (default 20); T_U is rounded to a 20-step grid, so the unit fixes the exact step count
#   ORDER     rung (default: every arm's first rung, then every arm's next rung, ...) | arm (one arm's whole ladder,
#             then the next arm's)
#   MICRO / ACCUM  per-GPU micro-batch (sequences) and gradient-accumulation steps (default 16 / 1)
#   NGPU      ranks (default 8; tokens per step = MICRO x ACCUM x NGPU x 4096); GPU_LIST (0..NGPU-1) the cards used
#   WARMUP (128), DECAY_RATIO (0.1), DECAY_TYPE (sqrt), LR_MIN (0.0): the WSD schedule
#   WD (0.0), WD_PARAMS (all | matrices | matrices_no_embed; all), BETA1 (0.9), BETA2 (0.95), EPS (1e-15): AdamW
#   SEED (42); IV_MAX (2200, the largest checkpoint interval); ACMODE (activation checkpointing: none)
#   SMOKE     1 (default) runs 20 steps of every arm before its first rung, so a broken arm fails in minutes
#   KEEP      keep_latest_k while a rung trains (0 = keep every save until the rung ends; a small disk sets e.g. 5,
#             which must exceed the number of saves between the fork step and the end of the rung)
#   FIRST_GPU the card the per-rung evaluation runs on in the background (0); VAL_NSEQ (1024) its number of windows
#   TB (0; 1 = TensorBoard event files under <dump>/tb), WANDB (0; 1 = wandb; needs WANDB_MODE=online, WANDB_API_KEY
#             and WANDB_PROJECT in the environment)
#   DATA_ROOT (/data/climb_text, the directory scripts/fetch_data.py filled), DATA ($DATA_ROOT/train/data-*.arrow,
#             the training shards), CLIMBVAL ($DATA_ROOT/val/*.arrow, the held-out shard),
#             TOK (trainer/transformer-1.3B-100B, the tokenizer), BASE_TMPL (tmpl_vanilla_lr1e3_e1x.toml.tmpl, the
#             recipe template every per-arm template is derived from)
#   OUT_DIR   dumps, logs, templates and the results table (required); SUMMARY_MD ($OUT_DIR/climb4k_wsd_<SELF_TAG>_results.md)
#   ENVDIR    the python environment (default: the one python3 resolves to)
#
# The paper's tables (8 GPUs; lr = Standard / looped arms; NREF = the Standard model's total parameters):
#   scale  MICRO x ACCUM  tok/step   NREF           TPP_UNIT  TPPS    steps  lr             training text
#   191M   16 x 1           524,288    190,741,248  20        "20"    7,280  1e-3 / 3e-3    ~26B tokens (fetch_data.py)
#   374M    8 x 4         1,048,576    373,867,520   5        "5 20"  7,120  2e-3 / 2e-3    ~26B tokens
#   778M    8 x 8         2,097,152    777,856,512   5        "5 20"  7,440  1e-3 / 1.5e-3  ~26B tokens
#   1.4B    8 x 15        3,932,160  1,415,682,048  20        "20"    7,200  1e-3 / 1.5e-3  ~66B tokens (fetch_data.py --fraction 0.17)
#   3B      4 x 60        7,864,320  2,914,667,520   1        "1 20"  7,600  1e-3 / 1e-3    ~66B tokens (fetch_data.py --fraction 0.17)
# T_U is rounded to the 20-step grid, so TPP_UNIT decides the exact step count: 374M and 778M were screened at 5 TPP
# and their 20-TPP rung forked from that rung's pre-decay checkpoint (TPP_UNIT=5 TPPS="20" trains the identical
# 7,120 / 7,440-step run as one rung); 3B was run as a 1-TPP rung (380 steps) plus the fork to 20 TPP (TPP_UNIT=1
# TPPS="20" gives the same 7,600 steps). TPP_UNIT=20 would give 7,140 / 7,420 / 7,420 steps at 374M / 778M / 3B.
# The paper's runs read shards 0..239 (191M-778M) or 0..559 (1.4B, 3B) of a private 3,584-shard split of ClimbMix;
# scripts/fetch_data.py samples the public text release into shards of the same format and cluster mixture.
# Launch from trainer/ (it cds there itself), e.g. the 191M table:
#   SELF_TAG=t191 NREF=190741248 MICRO=16 ACCUM=1 OUT_DIR=/data/runs DATA_ROOT=/data/climb_text \
#     ARMS="standard:../configs/191m_d768x20/lt124m_dense20_4klen.json:1e-3 loopback:../configs/191m_d768x20/lt124m_ours_p2k4l4c2_4klen.json:3e-3" \
#     bash queue_wsd_ladder.sh
# Per rung: $OUT_DIR/wsd_<arm>_tpp<k>/ (the checkpoints step-F_k and step-T_k), wsd_<arm>_tpp<k>.log,
# wsd_<arm>_tpp<k>_climbval.json (the validation loss) and a row of the results table; the log line
# "RESULT <arm> rung <k> ..." repeats the row.
set -uo pipefail
: "${SELF_TAG:?}" "${ARMS:?}" "${NREF:?NREF (the parameter count TPP is counted on) is required}"
T="[$SELF_TAG]"
TPPS="${TPPS:-20}"; ORDER="${ORDER:-rung}"; TPP_UNIT="${TPP_UNIT:-20}"
MICRO="${MICRO:-16}"; ACCUM="${ACCUM:-1}"; NGPU="${NGPU:-8}"; SEQ=4096; NSEQ="${VAL_NSEQ:-1024}"
GPU_LIST="${GPU_LIST:-$(seq -s, 0 $((NGPU - 1)))}"   # the cards the ranks use; default the first NGPU
WARMUP="${WARMUP:-128}"; DECAY_RATIO="${DECAY_RATIO:-0.1}"; DECAY_TYPE="${DECAY_TYPE:-sqrt}"; LR_MIN="${LR_MIN:-0.0}"
WD="${WD:-0.0}"; WD_PARAMS="${WD_PARAMS:-all}"; BETA1="${BETA1:-0.9}"; BETA2="${BETA2:-0.95}"; EPS="${EPS:-1e-15}"; SEED="${SEED:-42}"; IV_MAX="${IV_MAX:-2200}"
TB="${TB:-0}"; WANDB="${WANDB:-0}"
KEEP="${KEEP:-0}"
TOKSTEP=$((MICRO * ACCUM * NGPU * SEQ))
declare -A CFG_OF LR_OF; ARM_LIST=""
for kv in $ARMS; do
  a=${kv%%:*}; rest=${kv#*:}; f=${rest%%:*}; l=${rest#*:}
  { [ "$l" != "$rest" ] && [[ "$l" =~ ^[0-9.]+e?-?[0-9]*$ ]]; } || { echo "$T ARMS entry '$kv' is not name:config:lr -- abort"; exit 3; }
  [ -z "${CFG_OF[$a]:-}" ] || { echo "$T duplicate arm name $a -- abort"; exit 3; }
  CFG_OF[$a]=$f; LR_OF[$a]=$l; ARM_LIST="$ARM_LIST $a"
done; ARM_LIST=${ARM_LIST# }
case "$ORDER" in rung|arm) ;; *) echo "$T ORDER must be rung|arm -- abort"; exit 3;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO"
ENVDIR="${ENVDIR:-$(python3 -c "import sys, os; print(os.path.dirname(os.path.dirname(sys.executable)))")}"   # the python env (default: the one python3 resolves to)
export PATH="$ENVDIR/bin:$PATH" PYTHONNOUSERSITE=1 WANDB_MODE="${WANDB_MODE:-offline}"
export ACMODE="${ACMODE:-none}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
OUT="${OUT_DIR:?set OUT_DIR=<directory for the dumps, logs and result tables>}"
export WANDB_DIR="$OUT/wandb" TMPDIR="${TMPDIR:-$OUT/tmp}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$OUT/hf_cache}"
mkdir -p "$OUT" "$WANDB_DIR" "$TMPDIR"
# data: DATA_ROOT is the directory scripts/fetch_data.py filled (train/ and val/); every path is overridable.
TOK="${TOK:-$REPO/transformer-1.3B-100B}"
DATA_ROOT="${DATA_ROOT:-/data/climb_text}"
CLIMBVAL="${CLIMBVAL:-$(ls $DATA_ROOT/val/*.arrow 2>/dev/null | head -n 1)}"
export DATA="${DATA:-$DATA_ROOT/train/data-*.arrow}"
BASE_TMPL="${BASE_TMPL:-$REPO/tmpl_vanilla_lr1e3_e1x.toml.tmpl}"
SUMMARY="${SUMMARY_MD:-$OUT/climb4k_wsd_${SELF_TAG}_results.md}"
VAL_PIDS=""

# ---- the dataloader must resume document-exact (a branch would otherwise skip data at every fork) ----
grep -q "hf_table_start" flame/data.py || { echo "$T flame/data.py lacks the document-exact resume (hf_table_start) -- abort"; exit 3; }
test -f "$BASE_TMPL" || { echo "$T missing BASE_TMPL $BASE_TMPL -- abort"; exit 3; }

# ---- ladder arithmetic: T_k, D_k, F_k per rung and the checkpoint interval (torchtitan's own rounding) ----
LADDER=$(python3 - "$NREF" "$TOKSTEP" "$DECAY_RATIO" "$IV_MAX" "$WARMUP" "$TPP_UNIT" $TPPS <<'PY'
import sys
nref, tokstep, ratio, iv_max, warmup, unit = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
tpps = [int(x) for x in sys.argv[7:]]
assert unit > 0 and tpps == sorted(tpps) and len(set(tpps)) == len(tpps) and all(k > 0 and k % unit == 0 for k in tpps), f"TPPS must be ascending multiples of TPP_UNIT={unit}: {tpps}"
t20 = int(round(unit * nref / tokstep / 20.0)) * 20   # the unit rung's steps (= T_20 when unit is 20)
assert t20 >= 20
rows, forks = [], []
for k in tpps:
    T = t20 * k // unit
    D = round(T * ratio)                          # build_lr_schedulers: decay_steps = round(steps * decay_ratio)
    F = T - D
    assert D >= 2 and F > warmup, (k, T, D, F)
    rows.append((k, T, D, F)); forks.append(F)
f20 = t20 - round(t20 * ratio)
iv = max(d for d in range(1, min(f20, iv_max) + 1) if f20 % d == 0)
assert all(F % iv == 0 for F in forks), (forks, iv)
print(f"T{unit}={t20} IV={iv}")
for k, T, D, F in rows:
    print(f"RUNG {k} {T} {D} {F} {T * tokstep / 1e9:.3f} {T * tokstep / nref:.2f}")
PY
) || { echo "$T ladder arithmetic failed -- abort"; exit 3; }
IV=$(echo "$LADDER" | head -1 | grep -oE 'IV=[0-9]+' | cut -d= -f2)
declare -A T_OF D_OF F_OF TOKB_OF TPPX_OF
while read -r tag k Tk Dk Fk tb tx; do
  [ "$tag" = RUNG ] || continue
  T_OF[$k]=$Tk; D_OF[$k]=$Dk; F_OF[$k]=$Fk; TOKB_OF[$k]=$tb; TPPX_OF[$k]=$tx
done <<< "$LADDER"
echo "$T $(date -Is) ladder: NREF=$NREF, $TOKSTEP tok/step (micro $MICRO x accum $ACCUM x $NGPU), warmup $WARMUP, decay last $DECAY_RATIO $DECAY_TYPE -> $LR_MIN, checkpoint interval $IV, TPP unit $TPP_UNIT"
for k in $TPPS; do echo "$T   rung $k TPP: T=${T_OF[$k]} steps (${TOKB_OF[$k]}B tok, ${TPPX_OF[$k]} TPP), decay ${D_OF[$k]} steps, fork for the next rung at step ${F_OF[$k]}"; done

# ---- per-rung recipe templates, checked line by line after generation ----
tmpl_of() { echo "$OUT/tmpl_wsd_${SELF_TAG}_$1.toml.tmpl"; }   # one per arm (the recipe does not change across rungs)
for a in $ARM_LIST; do
  test -f "${CFG_OF[$a]}" || { echo "$T missing config for $a: ${CFG_OF[$a]} -- abort"; exit 3; }
  python3 scripts/mk_wsd_tmpl.py --src "$BASE_TMPL" --dst "$(tmpl_of $a)" --micro "$MICRO" --accum "$ACCUM" --lr "${LR_OF[$a]}" \
    --warmup "$WARMUP" --decay_ratio "$DECAY_RATIO" --decay_type "$DECAY_TYPE" --lr_min "$LR_MIN" --wd "$WD" --beta1 "$BETA1" \
    --beta2 "$BETA2" --eps "$EPS" --wd_params "$WD_PARAMS" --interval "$IV" --keep "$KEEP" --log_freq 10 --seed "$SEED" > /dev/null || { echo "$T template generation failed for $a -- abort"; exit 3; }
  TM="$(tmpl_of $a)"
  if [ "$TB" = 1 ]; then
    grep -q "^enable_tensorboard" "$TM" || sed -i "s/^enable_wandb = .*/&\nenable_tensorboard = false\nsave_tb_folder = \"tb\"/" "$TM"
    sed -i "s/^enable_tensorboard = .*/enable_tensorboard = true/" "$TM"
  fi
  [ "$WANDB" = 1 ] && sed -i "s/^enable_wandb = .*/enable_wandb = true/" "$TM"
  for line in "batch_size = $MICRO" "gradient_accumulation_steps = $ACCUM" 'seq_len = 4096' 'context_len = 4096' 'steps = __STEPS__' \
              'name = "AdamW"' "lr = ${LR_OF[$a]}" "weight_decay = $WD" "wd_params = \"$WD_PARAMS\"" "beta1 = $BETA1" "beta2 = $BETA2" "eps = $EPS" \
              "warmup_steps = $WARMUP" "decay_ratio = $DECAY_RATIO" "decay_type = \"$DECAY_TYPE\"" "lr_min = $LR_MIN" 'max_norm = 1.0' \
              "seed = $SEED" "interval = $IV" "keep_latest_k = $KEEP" 'enable_checkpoint = true' 'compile = false'; do
    grep -qxF "$line" "$(tmpl_of $a)" || { echo "$T template $(tmpl_of $a) lacks '$line' -- abort"; exit 3; }
  done
  python3 -c "import json,sys
c=json.load(open(sys.argv[1])); sys.exit(0 if (c.get('embed_init_std') is None and not c.get('zero_init_residual_proj') and not c.get('logit_softcap') and not c.get('tie_word_embeddings')) else 1)" "${CFG_OF[$a]}" \
    || { echo "$T $a: ${CFG_OF[$a]} is not the plain model package (embed_init_std/zero_init/softcap/tie must be off) -- abort"; exit 3; }
done
echo "$T $(date -Is) recipe ok: AdamW wd $WD on $WD_PARAMS betas ($BETA1, $BETA2) eps $EPS, ACMODE=$ACMODE; arms: $(for a in $ARM_LIST; do printf '%s(lr %s) ' "$a" "${LR_OF[$a]}"; done)"

[ -f "$SUMMARY" ] || cat > "$SUMMARY" <<EOH
# ClimbMix WSD ladder [$SELF_TAG]: decayed checkpoints at fixed tokens-per-parameter budgets

Recipe (flame looped_transformer, plain Llama package: N(0,0.02) untied embeddings, no softcap / zero-init /
QK norm; RoPE 1e6, Llama 32k, seq 4096): AdamW wd $WD on $WD_PARAMS parameters, betas ($BETA1, $BETA2), eps $EPS, clip 1.0,
$TOKSTEP tok/step (micro $MICRO x accum $ACCUM x $NGPU), seed $SEED. WSD: linear warmup $WARMUP steps, peak lr
held, the last $DECAY_RATIO of each rung's steps decayed to $LR_MIN along $DECAY_TYPE. TPP counted on
NREF = $NREF parameters for every arm. Rung k forks from rung k-1's last pre-decay checkpoint, so each row
is the uninterrupted WSD run to that budget. val = the held-out ClimbMix shard, $NSEQ x 4096 windows
packed as in training.

| arm | lr | TPP | tokens (B) | steps | tail-100 | val loss | val ppl |
|---|---|---|---|---|---|---|---|
EOH

losses() { grep -oE 'step: *[0-9]+ +.?\[32mloss: *[0-9.]+' "$1" 2>/dev/null | grep -oE 'loss: *[0-9.]+' | grep -oE '[0-9.]+'; }
tail_steps100() { losses "$1" | tail -10 | awk '{s+=$1;n++} END{if(n) printf "%.4f", s/n; else printf "n/a"}'; }
laststep()      { sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null | grep -oE 'step: *[0-9]+' | tail -1 | grep -oE '[0-9]+'; }
jget()          { python3 -c "import json,math;d=json.load(open('$1'));print(f\"{d['val_loss']:.4f}\" if '$2'=='loss' else f\"{math.exp(d['val_loss']):.2f}\")" 2>/dev/null || echo n/a; }
has_ckpt()      { [ -f "$1/.metadata" ]; }

# launch_train CFG DUMP LOG NSTEPS RESUME STALL_POLLS  (stall poll = 10 s). 0 iff rc=0 and checkpoint/step-NSTEPS exists.
launch_train() {
  local CFG=$1 DUMP=$2 LOG=$3 NSTEPS=$4 RESUME=$5 STALL=$6 attempt=0 pid last cur stale rc p
  while [ $attempt -lt 4 ]; do
    if [ "$RESUME" -eq 0 ] && [ $attempt -eq 0 ]; then rm -rf "$DUMP"; fi
    echo "$T $(date -Is) launch attempt $attempt -> $DUMP (steps $NSTEPS, template $TOML_TMPL)"
    setsid env GPUS="$GPU_LIST" MODEL_CONFIG="$CFG" DUMP="$DUMP" STEPS=$NSTEPS \
      bash scripts/run_local_train.sh >> "$LOG" 2>&1 < /dev/null &
    pid=$!; last=-1; stale=0
    while kill -0 $pid 2>/dev/null; do
      sleep 10
      cur=$(laststep "$LOG"); cur=${cur:-0}
      if [ "$cur" != "$last" ]; then last=$cur; stale=0; else stale=$((stale+1)); fi
      if [ $stale -ge "$STALL" ]; then
        echo "$T $(date -Is) STALL: no step progress for $((STALL/6)) min at step $cur -> killing process group $pid"
        kill -TERM -- -$pid 2>/dev/null; sleep 30; kill -KILL -- -$pid 2>/dev/null; sleep 10
        for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
          tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q -- "$DUMP" && { echo "$T killing leftover rank $p"; kill -KILL $p; }
        done
        break
      fi
    done
    wait $pid; rc=$?
    cur=$(laststep "$LOG"); cur=${cur:-0}
    if [ $rc -eq 0 ] && has_ckpt "$DUMP/checkpoint/step-$NSTEPS"; then return 0; fi
    echo "$T $(date -Is) attempt $attempt ended rc=$rc at step $cur (want $NSTEPS, checkpoint/step-$NSTEPS $(has_ckpt "$DUMP/checkpoint/step-$NSTEPS" && echo present || echo missing)); resuming from the newest checkpoint"
    RESUME=1; attempt=$((attempt+1)); sleep 30
  done
  return 1
}

# run_rung ARM K PREV_K(or "-")
run_rung() {
  local ARM=$1 K=$2 PK=$3
  local CFG="${CFG_OF[$ARM]}" LR="${LR_OF[$ARM]}" TK=${T_OF[$2]} FK=${F_OF[$2]}
  local DUMP="$OUT/wsd_${ARM}_tpp$K" LOG="$OUT/wsd_${ARM}_tpp$K.log" VALC="$OUT/wsd_${ARM}_tpp${K}_climbval.json"
  export TOML_TMPL; TOML_TMPL="$(tmpl_of "$ARM")"
  # a rung is done when its val exists
  if [ -f "$VALC" ]; then echo "$T $(date -Is) SKIP $ARM rung $K (val exists)"; return 0; fi
  if [ ! -f "$OUT/wsd_${SELF_TAG}_${ARM}_smoke.ok" ] && [ "${SMOKE:-1}" = 1 ]; then
    echo "$T $(date -Is) === SMOKE $ARM ($NGPU ranks, 20 steps, template $TOML_TMPL) ==="
    rm -f "$OUT/smokefull_wsd_$ARM.log"
    if ! launch_train "$CFG" "$OUT/smokefull_wsd_$ARM" "$OUT/smokefull_wsd_$ARM.log" 20 0 120; then
      echo "$T $(date -Is) SMOKE FAILED $ARM -- arm skipped"; tail -25 "$OUT/smokefull_wsd_$ARM.log"; return 1
    fi
    echo "$T $(date -Is) smoke ok: step-20 loss $(losses "$OUT/smokefull_wsd_$ARM.log" | tail -1)  $(sed 's/\x1b\[[0-9;]*m//g' "$OUT/smokefull_wsd_$ARM.log" | grep -oE 'memory: *[0-9.]+GiB\([0-9.]+%\) +tps: *[0-9,]+' | tail -1)"
    touch "$OUT/wsd_${SELF_TAG}_${ARM}_smoke.ok"; rm -rf "$OUT/smokefull_wsd_$ARM"
  fi
  if ! has_ckpt "$DUMP/checkpoint/step-$TK"; then
    local RES=0
    if [ -n "$(find "$DUMP/checkpoint" -maxdepth 2 -name .metadata 2>/dev/null | head -1)" ]; then
      RES=1; echo "$T $(date -Is) $ARM rung $K: resuming its own newest checkpoint"
    elif [ "$PK" != "-" ]; then
      local SRC="$OUT/wsd_${ARM}_tpp$PK/checkpoint/step-${F_OF[$PK]}"
      has_ckpt "$SRC" || { echo "$T $(date -Is) FATAL $ARM rung $K: fork checkpoint $SRC is missing -- rung not run"; return 1; }
      mkdir -p "$DUMP/checkpoint" && cp -al "$SRC" "$DUMP/checkpoint/" || { echo "$T FATAL $ARM rung $K: cannot hardlink $SRC"; return 1; }
      RES=1; echo "$T $(date -Is) $ARM rung $K: forked from rung $PK's step ${F_OF[$PK]} (pre-decay, peak lr)"
    fi
    echo "AC=$ACMODE ${MICRO}x${ACCUM} opt=AdamW lr=$LR wd=$WD/$WD_PARAMS betas=($BETA1,$BETA2) eps=$EPS wsd=warmup${WARMUP}+${DECAY_TYPE}${DECAY_RATIO}->${LR_MIN} rung=${K}TPP T=$TK fork=$( [ "$PK" = - ] && echo scratch || echo "tpp$PK@${F_OF[$PK]}" )  (template $TOML_TMPL, config $CFG, $(date -Is))" > "$OUT/wsd_${ARM}_tpp${K}_recipe.txt"
    echo "$T $(date -Is) === TRAIN $ARM rung $K: steps -> $TK (${TOKB_OF[$K]}B tok), lr $LR ==="
    if ! launch_train "$CFG" "$DUMP" "$LOG" "$TK" "$RES" 180; then echo "$T $(date -Is) TRAIN FAILED $ARM rung $K, see $LOG"; return 1; fi
    echo "$T $(date -Is) === TRAIN $ARM rung $K done -> val in background on card ${FIRST_GPU:-0} ==="
  fi
  # the next rung needs step-F_K; the decayed model is step-T_K; every other save of this rung goes
  has_ckpt "$DUMP/checkpoint/step-$FK" || { echo "$T $(date -Is) FATAL $ARM rung $K: fork checkpoint step-$FK missing after training (interval $IV)"; return 1; }
  local d
  for d in "$DUMP"/checkpoint/step-*; do
    case "$(basename "$d")" in "step-$FK"|"step-$TK") ;; *) rm -rf "$d" ;; esac
  done
  (
    if [ ! -f "$VALC" ]; then
      CUDA_VISIBLE_DEVICES=${FIRST_GPU:-0} python3 scripts/eval_val_loss.py --dump "$DUMP" --step "$TK" --config "$CFG" --tokenizer "$TOK" \
        --data_glob "$CLIMBVAL" --n_seq $NSEQ --seq_len $SEQ --arm "wsd_${ARM}_tpp${K}_climb" --out "$VALC" >> "$LOG" 2>&1 || echo "$T VAL rc=$? ($ARM rung $K)"
    fi
    TT=$(tail_steps100 "$LOG"); VC=$(jget "$VALC" loss); PC=$(jget "$VALC" ppl)
    echo "| $ARM | $LR | $K | ${TOKB_OF[$K]} | $TK | $TT | $VC | $PC |" >> "$SUMMARY"
    echo "$T $(date -Is) RESULT $ARM rung $K (${TOKB_OF[$K]}B tok): tail100=$TT climb=$VC ppl=$PC  [$(cut -d'(' -f1 "$OUT/wsd_${ARM}_tpp${K}_recipe.txt" 2>/dev/null | sed 's/ *$//')]"
  ) &
  VAL_PIDS="$VAL_PIDS $!"
}

# ---- data present ----
[ -n "$(ls $DATA 2>/dev/null | head -1)" ] || { echo "$T no training shards match $DATA"; exit 3; }
test -s "$CLIMBVAL" || { echo "$T missing val shard $CLIMBVAL"; exit 3; }
echo "$T $(date -Is) $(ls $DATA | wc -l) train shards ($DATA) + val shard $CLIMBVAL"

if [ "$ORDER" = rung ]; then
  PK=-
  for K in $TPPS; do
    for a in $ARM_LIST; do run_rung "$a" "$K" "$PK" || echo "$T $(date -Is) RUNG FAILED $a $K"; done
    PK=$K
  done
else
  for a in $ARM_LIST; do
    PK=-
    for K in $TPPS; do run_rung "$a" "$K" "$PK" || { echo "$T $(date -Is) RUNG FAILED $a $K -- later rungs of $a skipped"; break; }; PK=$K; done
  done
fi
echo "$T $(date -Is) LAST TRAIN DONE (the val of the last rung is still running on card ${FIRST_GPU:-0}; the table is complete at DONE)"
[ -n "$VAL_PIDS" ] && { echo "$T $(date -Is) waiting for background vals:$VAL_PIDS"; for p in $VAL_PIDS; do wait $p 2>/dev/null; done; }
echo "$T $(date -Is) ALL RUNGS DONE"
grep -E "^\| *($(echo $ARM_LIST | tr ' ' '|')) " "$SUMMARY"
echo "$T $(date -Is) DONE"
