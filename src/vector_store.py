"""
A minimal local vector store for face embeddings, using FAISS instead of a
managed cloud vector DB. For a handful of students this is overkill on
raw performance (a linear scan would be instant too) — the point is to
keep the interface identical to what you'd use at 10k+ students, so
nothing else in the app has to change if the deployment grows.
"""

import os
from typing import List, Tuple

import numpy as np
import faiss

from . import config


class VectorStore:
    def __init__(self):
        self._dim = config.EMBEDDING_DIM
        if os.path.exists(config.FAISS_INDEX_PATH):
            self._index = faiss.read_index(config.FAISS_INDEX_PATH)
        else:
            flat = faiss.IndexFlatIP(self._dim)  # inner product == cosine, since embeddings are L2-normalized
            self._index = faiss.IndexIDMap(flat)

    def upsert(self, student_id: int, embedding: np.ndarray):
        """Adds or replaces a student's reference embedding."""
        ids = np.array([student_id], dtype="int64")
        # FAISS has no in-place update; remove any existing entry for this id first.
        self._index.remove_ids(ids)
        self._index.add_with_ids(embedding.reshape(1, -1), ids)
        self._save()

    def search(self, embedding: np.ndarray, top_k: int = 1) -> List[Tuple[int, float]]:
        """Returns [(student_id, cosine_similarity), ...] sorted best-first."""
        if self._index.ntotal == 0:
            return []
        scores, ids = self._index.search(embedding.reshape(1, -1), top_k)
        results = []
        for score, student_id in zip(scores[0], ids[0]):
            if student_id == -1:
                continue
            results.append((int(student_id), float(score)))
        return results

    def _save(self):
        faiss.write_index(self._index, config.FAISS_INDEX_PATH)
