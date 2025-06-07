from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

VALIDATOR_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

tokenizer = AutoTokenizer.from_pretrained(VALIDATOR_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(VALIDATOR_MODEL)
model.eval()


def validate_text(text: str) -> bool:
    """Return True if no contradiction detected."""
    prompt = f"Is the following text self-contradictory? {text} Answer yes or no."
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = logits.argmax(dim=-1).item()
        return prediction == 1  # assume label 1 means 'no contradiction'
