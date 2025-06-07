#!/bin/bash

# Ensure we stop on errors and undefined variables
set -euo pipefail

PDF_DIR=$1
MODEL=${2-}
if [ -z "$PDF_DIR" ]; then
  echo "Usage: $0 <pdf_dir> [model]"
  exit 1
fi

python data_ingest/load_pdfs.py "$PDF_DIR" --out data_ingest/dataset.jsonl
if [ -n "$MODEL" ]; then
  python training/qlora_train.py data_ingest/dataset.jsonl --out training/novel_adapter --model "$MODEL"
else
  python training/qlora_train.py data_ingest/dataset.jsonl --out training/novel_adapter
fi
python training/merge_and_quantize.py training/novel_adapter --out models

# Initialize story bible if not exists
if [ ! -f rag/story_bible.json ]; then
  echo '{"facts": [{"id": "hero", "type": "character", "text": "Hero is 17 years old", "timestamp": 0}]}' > rag/story_bible.json
fi

python -m api.server
