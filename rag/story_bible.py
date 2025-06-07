import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib





@dataclass
class Fact:
    id: str
    type: str
    text: str
    timestamp: float


class StoryBible:
    def __init__(self, path: Path):
        self.path = path
        self.index_path = path.with_suffix('.pkl')
        self.embedder = TfidfVectorizer()
        self.index: NearestNeighbors | None = None
        self.facts: List[Fact] = []
        if path.exists():
            self._load()

    def _load(self):
        data = json.loads(self.path.read_text())
        self.facts = [Fact(**f) for f in data['facts']]
        if self.index_path.exists():
            self.embedder, self.index = joblib.load(self.index_path)

    def _save(self):
        self.path.write_text(json.dumps({'facts': [asdict(f) for f in self.facts]}, indent=2))
        if self.index is not None:
            joblib.dump((self.embedder, self.index), self.index_path)

    def add_fact(self, fact: Fact):
        self.facts.append(fact)
        self._build_index()
        self._save()

    def _build_index(self):
        texts = [f.text for f in self.facts]
        embeddings = self.embedder.fit_transform(texts)
        self.index = NearestNeighbors(metric='cosine')
        self.index.fit(embeddings)

    def search(self, query: str, k: int = 5) -> List[Fact]:
        if self.index is None:
            return []
        q_emb = self.embedder.transform([query])
        dists, idx = self.index.kneighbors(q_emb, n_neighbors=min(k, len(self.facts)))
        return [self.facts[i] for i in idx[0]]


def load_story_bible(path: Path) -> StoryBible:
    return StoryBible(path)
