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
3. Authenticate with `huggingface-cli login` and accept the Llama\-3 model license.
4. Run `bash run_pipeline.sh pdfs/` (downloads the Llama\-3 base model using your HuggingFace token).
5. After processing, the FastAPI server will run on `http://localhost:8000`.

### Endpoints
- `POST /draft` `{outline: str}` → streamed chapter text.
- `POST /facts` → append new fact to story bible.
- `GET /health` → server status.

## Hardware Requirements
- 16 GB system RAM
- NVIDIA RTX 3050 6 GB (or CPU fallback)

## License Notice
This project bundles Meta Llama 3 license files. Derivative weights may not be re-licensed.
