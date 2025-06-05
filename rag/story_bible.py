import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


dataclass_schema = {
    "id": str,
    "type": str,
    "text": str,
    "timestamp": float,
}


@dataclass
class Fact:
    id: str
    type: str
    text: str
    timestamp: float


class StoryBible:
    def __init__(self, path: Path):
        self.path = path
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.index = None
        self.facts: List[Fact] = []
        if path.exists():
            self._load()

    def _load(self):
        data = json.loads(self.path.read_text())
        self.facts = [Fact(**f) for f in data['facts']]
        self.index = faiss.read_index(str(self.path.with_suffix('.index')))

    def _save(self):
        self.path.write_text(json.dumps({'facts': [asdict(f) for f in self.facts]}, indent=2))
        if self.index is not None:
            faiss.write_index(self.index, str(self.path.with_suffix('.index')))

    def add_fact(self, fact: Fact):
        self.facts.append(fact)
        self._build_index()
        self._save()

    def _build_index(self):
        embeddings = self.embedder.encode([f.text for f in self.facts], convert_to_numpy=True)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 5) -> List[Fact]:
        if not self.index:
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        scores, idx = self.index.search(q_emb, k)
        return [self.facts[i] for i in idx[0]]


def load_story_bible(path: Path) -> StoryBible:
    return StoryBible(path)
