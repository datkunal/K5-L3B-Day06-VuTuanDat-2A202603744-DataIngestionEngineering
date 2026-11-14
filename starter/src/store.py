"""Persistent vector store: Chroma (on-disk PersistentClient), cosine HNSW, metadata filters.

Docker/pgvector was not available on the lab machine, so Chroma replaces the PgVectorStore
skeleton; the GUIDE allows Chroma as an equivalent store.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from .models import Chunk

ALLOWED_FILTER_FIELDS = {"ip_type", "chapter", "section", "document_id"}


class ChromaStore:
    def __init__(self, path: Path, dimensions: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.dimensions = dimensions

    def collection(self, strategy: str):
        return self.client.get_or_create_collection(
            name=f"day06_strategy_{strategy}",
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,  # vectors are always supplied by our embedder
            metadata={"strategy": strategy},
        )

    def existing(self, strategy: str, document_id: str) -> dict[str, str]:
        """chunk_id -> content_sha1 currently stored for a document."""
        got = self.collection(strategy).get(where={"document_id": document_id}, include=["metadatas"])
        return {i: m.get("content_sha1", "") for i, m in zip(got["ids"], got["metadatas"])}

    def get_vectors(self, strategy: str, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        got = self.collection(strategy).get(ids=ids, include=["embeddings"])
        return {i: list(map(float, v)) for i, v in zip(got["ids"], got["embeddings"])}

    def delete(self, strategy: str, ids: list[str]) -> None:
        if ids:
            self.collection(strategy).delete(ids=ids)

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Write chunks under stable IDs (re-running replaces, never duplicates)."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError(f"vector has {len(v)} dims, expected {self.dimensions}")
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk ids in one batch")
        metadatas = [{"document_id": c.document_id, "source": c.source, "page": c.page, **c.metadata}
                     for c in chunks]
        col = self.collection(strategy)
        for i in range(0, len(chunks), 500):
            col.upsert(ids=ids[i:i + 500], embeddings=vectors[i:i + 500],
                       documents=[c.text for c in chunks[i:i + 500]], metadatas=metadatas[i:i + 500])

    def count(self, strategy: str) -> int:
        return self.collection(strategy).count()

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        where = None
        if filters:
            bad = set(filters) - ALLOWED_FILTER_FIELDS
            if bad:
                raise ValueError(f"filter fields not allowed: {bad}")
            clauses = [{k: v} for k, v in filters.items()]
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        res = self.collection(strategy).query(query_embeddings=[query_vector], n_results=top_k, where=where,
                                              include=["documents", "metadatas", "distances"])
        hits = []
        for rank, (cid, doc, md, dist) in enumerate(zip(res["ids"][0], res["documents"][0],
                                                        res["metadatas"][0], res["distances"][0]), start=1):
            hits.append({"rank": rank, "chunk_id": cid, "score": round(1.0 - float(dist), 6),
                         "cosine_distance": round(float(dist), 6), "text": doc, "metadata": md})
        return hits
