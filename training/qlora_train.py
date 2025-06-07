import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def train(data_path: Path, output_dir: Path, epochs: int = 3):
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
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
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
