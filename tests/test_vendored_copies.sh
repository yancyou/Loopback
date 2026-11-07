#!/usr/bin/env bash
# The reader files are vendored twice (repository root for the pure-PyTorch model, trainer/ for the training model)
# and must be byte-identical; the fused read as well.
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; rc=0
for f in loop_linattn_delta.py loop_linattn_state.py fused_dattn.py; do
  if cmp -s "$R/$f" "$R/trainer/flame/custom_models/looped_transformer/$f"; then echo "same  $f"; else echo "DIFFER $f"; rc=1; fi
done
exit $rc
