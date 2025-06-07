import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import yaml


DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CONFIG_PATH = Path("config.yaml")


def train(data_path: Path, output_dir: Path, model_name: str = DEFAULT_MODEL, epochs: int = 3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=4096), batched=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(epochs):
        for batch in dataset:
            optimizer.zero_grad()
            input_ids = torch.tensor(batch["input_ids"]).to(model.device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning")
    parser.add_argument("data", type=Path, help="Path to dataset.jsonl")
    parser.add_argument("--out", type=Path, default=Path("training/novel_adapter"))
    parser.add_argument("--model", type=str, default=None, help="Base model path or HF repo")
    args = parser.parse_args()
    model_name = args.model
    if model_name is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f) or {}
            model_name = cfg.get("model", DEFAULT_MODEL)
        else:
            model_name = DEFAULT_MODEL
    train(args.data, args.out, model_name=model_name)


if __name__ == "__main__":
    main()
