from pathlib import Path
from typing import List
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from .validator import validate_text
from rag.story_bible import load_story_bible

MODEL_PATH = 'models/novel_model.gguf'  # placeholder path


def generate_draft(outline: str, bible_path: Path, max_words: int = 2000) -> str:
    story_bible = load_story_bible(bible_path)
    facts = story_bible.search(outline, k=5)
    context = '\n'.join([f.text for f in facts])
    prompt = f"{context}\n\nOutline: {outline}\nDraft:"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    max_new_tokens = max_words
    for _ in range(3):
        outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if validate_text(text):
            return text
    return text
