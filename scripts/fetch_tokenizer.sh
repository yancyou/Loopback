#!/usr/bin/env bash
# The Llama 32k tokenizer every run uses, from fla's fla-hub/transformer-1.3B-100B checkpoint (tokenizer files only),
# into trainer/transformer-1.3B-100B/ -- the directory the ladder, the launcher and the evaluators use by default (TOK=).
#   bash scripts/fetch_tokenizer.sh
set -euo pipefail
DST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/trainer/transformer-1.3B-100B"
mkdir -p "$DST"
python3 - "$DST" <<'PY'
import sys
from huggingface_hub import hf_hub_download
dst = sys.argv[1]
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    hf_hub_download("fla-hub/transformer-1.3B-100B", f, local_dir=dst)
print("tokenizer files in", dst)
PY
