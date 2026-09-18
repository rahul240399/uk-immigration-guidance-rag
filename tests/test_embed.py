"""
Tests for code.index.embed — local only, no large data.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from code.index.embed import encode_query, QUERY_PREFIX


@pytest.fixture(scope="module")
def model():
    return SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")


class TestIdMapOrder:
    def test_id_map_length_and_order(self, tmp_path, model):
        """id_map length equals chunk count and preserves file order."""
        # Write 3 synthetic chunks
        chunks = [
            {"chunk_id": "c1", "text": "Immigration rules part one."},
            {"chunk_id": "c2", "text": "Skilled worker requirements."},
            {"chunk_id": "c3", "text": "Student visa conditions."},
        ]
        idx_dir = tmp_path / "para"
        idx_dir.mkdir()
        with open(idx_dir / "chunks.jsonl", "w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")

        from code.index.embed import load_chunks
        ids, texts = load_chunks(idx_dir, "para")
        assert ids == ["c1", "c2", "c3"]
        assert len(texts) == 3


class TestNormalization:
    def test_l2_norm_is_one(self, model):
        """Every embedding row has L2 norm 1 ± 1e-3."""
        texts = ["Hello world", "Test sentence", "Another one"]
        emb = model.encode(texts, normalize_embeddings=True)
        norms = np.linalg.norm(emb, axis=1)
        for i, n in enumerate(norms):
            assert abs(n - 1.0) < 1e-3, f"Row {i} norm={n}"


class TestRanking:
    def test_matching_chunk_ranks_first(self, model):
        """A query matching one chunk's text ranks that chunk first."""
        chunks = [
            "The weather in London is often rainy and cold.",
            "Skilled Worker applicants must show English language ability at B1 level.",
            "The history of the Roman Empire spans centuries.",
        ]
        emb = model.encode(chunks, normalize_embeddings=True)
        query = encode_query(model, "English language level for Skilled Worker")
        scores = emb @ query
        assert np.argmax(scores) == 1
