"""Hybrid recommender: collaborative filtering + embedding cold-start (Chapter 38).

Known users get collaborative-filtering recommendations from shared interactions;
new users are profiled by embedding a free-text description against the catalog.
Explanations are generated so each pick is justified.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

from .config import get_embeddings
from .caching import _cosine
from .seed import SEED_CATALOG, SEED_INTERACTIONS


class Recommender:
    def __init__(self, *, backend: Optional[str] = None) -> None:
        self.embeddings = get_embeddings(backend=backend)
        self.catalog = {it["item_id"]: it for it in SEED_CATALOG}
        self._item_emb = {it["item_id"]: self.embeddings.embed_query(it["text"])
                          for it in SEED_CATALOG}
        self.interactions = {u: set(v) for u, v in SEED_INTERACTIONS.items()}

    def recommend(self, user_id: str, *, k: int = 3, profile_text: Optional[str] = None) -> List[dict]:
        liked = self.interactions.get(user_id, set())
        if liked:                                          # collaborative filtering
            scores: Counter = Counter()
            for other, items in self.interactions.items():
                if other == user_id:
                    continue
                overlap = len(liked & items)
                for it in items - liked:
                    if overlap:
                        scores[it] += overlap
            ranked = [i for i, _ in scores.most_common(k)]
            why = "users with similar taste also liked it"
        else:                                              # cold start via embeddings
            pv = self.embeddings.embed_query(profile_text or "")
            ranked = sorted(self._item_emb,
                            key=lambda i: _cosine(pv, self._item_emb[i]),
                            reverse=True)[:k]
            why = "it matches your stated interests"
        return [{"item_id": i, "title": self.catalog[i]["title"],
                 "explanation": f"Recommended because {why}."} for i in ranked]
