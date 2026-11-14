"""Real local embedding model (Sentence Transformers). No mock/random fallback."""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from . import config


class E5Embedder:
    """intfloat/multilingual-e5-base: 'query: ' / 'passage: ' prefixes, L2-normalized vectors."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        # Any load failure raises: we never fall back to fake vectors.
        self.model = SentenceTransformer(model_name, device=device)
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        self.max_seq_length = int(self.model.max_seq_length)
        self.device = str(self.model.device)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=config.BATCH_SIZE, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)
        if vecs.shape != (len(texts), self.dimensions):
            raise RuntimeError(f"unexpected embedding shape {vecs.shape}")
        return vecs.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([config.PASSAGE_PREFIX + t for t in texts]) if texts else []

    def embed_query(self, question: str) -> list[float]:
        return self._encode([config.QUERY_PREFIX + question])[0]

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(config.PASSAGE_PREFIX + text, add_special_tokens=True)["input_ids"])
