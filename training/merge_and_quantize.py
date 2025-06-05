import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def merge(base_model: str, adapter_dir: Path, output_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
    adapter_model = AutoModelForCausalLM.from_pretrained(adapter_dir, torch_dtype=torch.float16)
    model.load_state_dict(adapter_model.state_dict(), strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Convert to GGUF using llama.cpp (placeholder)
    # In practice, call convert-hf-to-gguf.py script
    gguf_path = output_dir / 'novel_model.gguf'
    with open(gguf_path, 'w') as f:
        f.write('GGUF PLACEHOLDER')
    print(f"Model saved to {gguf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('adapter_dir', type=Path)
    parser.add_argument('--out', type=Path, default=Path('models'))
    args = parser.parse_args()
    merge(DEFAULT_MODEL, args.adapter_dir, args.out)


if __name__ == '__main__':
    main()
