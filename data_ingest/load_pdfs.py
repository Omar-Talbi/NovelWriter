import argparse
import json
import os
from pathlib import Path
from typing import List

from pdfminer.high_level import extract_text

CHUNK_SIZE = 2000  # words

def clean_text(text: str) -> str:
    # Basic cleaning: collapse whitespace
    return ' '.join(text.split())


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def load_pdf(path: Path) -> str:
    return extract_text(str(path))


def process_pdfs(input_dir: Path, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as fout:
        for pdf_path in sorted(input_dir.glob('*.pdf')):
            text = load_pdf(pdf_path)
            text = clean_text(text)
            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                record = {
                    'id': f"{pdf_path.stem}-{idx}",
                    'text': chunk,
                }
                fout.write(json.dumps(record) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Load PDFs and output JSONL dataset')
    parser.add_argument('pdf_dir', type=Path)
    parser.add_argument('--out', type=Path, default=Path('data_ingest/dataset.jsonl'))
    args = parser.parse_args()
    process_pdfs(args.pdf_dir, args.out)


if __name__ == '__main__':
    main()
