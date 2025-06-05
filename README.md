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
2. Run `bash run_pipeline.sh pdfs/`.
3. After processing, the FastAPI server will run on `http://localhost:8000`.

### Docker
To build and run inside a container:
```bash
docker build -t novelwriter .
docker run -it -p 8000:8000 -v /path/to/pdfs:/app/pdfs novelwriter pdfs/
```

### Endpoints
- `POST /draft` `{outline: str}` → streamed chapter text.
- `POST /facts` → append new fact to story bible.
- `GET /health` → server status.

## Hardware Requirements
- 16 GB system RAM
- NVIDIA RTX 3050 6 GB (or CPU fallback)

## License Notice
This project bundles Meta Llama 3 license files. Derivative weights may not be re-licensed.
