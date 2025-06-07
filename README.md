# NovelWriter

Offline pipeline for generating novel chapters using local language models.

## Layout
```
ai-novel-writer/
├── data_ingest/          # PDF → JSONL
├── training/             # QLoRA scripts
├── rag/                  # fact store + retriever
├── generation/           # writer + validator loop
├── api/                  # FastAPI server
├── tests/                # unit + regression tests
├── run_pipeline.sh       # one-command end-to-end
└── README.md
```

## Quick Start
1. Place your source PDFs in a folder, e.g. `pdfs/`.
2. Install the Python requirements with `pip install -r requirements.txt`.
3. Run `bash run_pipeline.sh pdfs/ [model]`.
   - The optional `model` argument can be a local path or HF repo. If omitted,
     the value from `config.yaml` is used.
4. After processing, the FastAPI server will run on `http://localhost:8000`.

When invoking `qlora_train.py` directly, the same `--model` option is
available:

```bash
python training/qlora_train.py data_ingest/dataset.jsonl --out training/novel_adapter --model /path/to/model
```

### Endpoints
- `POST /draft` `{outline: str}` → streamed chapter text.
- `POST /facts` → append new fact to story bible.
- `GET /health` → server status.

## Hardware Requirements
- 16 GB system RAM
- NVIDIA RTX 3050 6 GB (or CPU fallback)

## License Notice
This project includes the Meta Llama 3 license as `META_Llama3_LICENSE.txt`. Derivative weights may not be re-licensed.
