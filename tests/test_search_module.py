#!/usr/bin/env python3
"""
YES/NO.EVENTS - Search Module Test Suite
========================================
Tests for TF-IDF vectorizer and market search.
"""
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from search import TFIDFVectorizer


# ============================================================================
# TF-IDF VECTORIZER TESTS
# ============================================================================

class TestTFIDFVectorizer:
    """Tests for TF-IDF Vectorizer"""

    @pytest.fixture
    def sample_documents(self):
        return [
            "bitcoin cryptocurrency trading market price",
            "trump election politics voting polls",
            "sports football basketball baseball soccer",
            "bitcoin price prediction cryptocurrency market",
            "weather forecast temperature rain sunny"
        ]

    @pytest.fixture
    def trained_vectorizer(self, sample_documents):
        vec = TFIDFVectorizer()
        vec.fit(sample_documents)
        return vec

    def test_fit_builds_vocabulary(self, sample_documents):
        """Should build vocabulary from documents"""
        vec = TFIDFVectorizer()
        vec.fit(sample_documents)

        assert len(vec.vocabulary) > 0
        assert 'bitcoin' in vec.vocabulary
        assert 'trump' in vec.vocabulary

    def test_fit_computes_idf(self, sample_documents):
        """Should compute IDF values"""
        vec = TFIDFVectorizer()
        vec.fit(sample_documents)

        assert len(vec.idf) > 0
        # Rare words should have higher IDF
        assert vec.idf.get('weather', 0) >= vec.idf.get('market', 0)

    def test_tokenize_lowercase(self):
        """Should lowercase tokens"""
        vec = TFIDFVectorizer()
        tokens = vec.tokenize("Bitcoin TRUMP Weather")

        assert 'bitcoin' in tokens
        assert 'trump' in tokens
        assert 'Bitcoin' not in tokens

    def test_tokenize_removes_short_tokens(self):
        """Should filter short tokens"""
        vec = TFIDFVectorizer()
        tokens = vec.tokenize("a I be the and bitcoin")

        assert 'bitcoin' in tokens
        assert 'a' not in tokens
        assert 'I' not in tokens

    def test_tokenize_removes_special_chars(self):
        """Should remove special characters"""
        vec = TFIDFVectorizer()
        tokens = vec.tokenize("bitcoin!!! @trump #crypto $100")

        assert 'bitcoin' in tokens
        assert '!' not in ''.join(tokens)

    def test_transform_returns_sparse_vector(self, trained_vectorizer):
        """Should return sparse vector dict"""
        vec = trained_vectorizer.transform("bitcoin price market")

        assert isinstance(vec, dict)
        assert 'bitcoin' in vec
        assert 'price' in vec
        assert vec['bitcoin'] > 0

    def test_cosine_similarity_identical(self, trained_vectorizer):
        """Identical vectors should have similarity 1"""
        vec1 = trained_vectorizer.transform("bitcoin cryptocurrency")
        sim = trained_vectorizer.cosine_similarity(vec1, vec1)

        assert 0.99 <= sim <= 1.01

    def test_cosine_similarity_orthogonal(self, trained_vectorizer):
        """Completely different vectors should have low similarity"""
        vec1 = trained_vectorizer.transform("bitcoin cryptocurrency")
        vec2 = trained_vectorizer.transform("weather sunny forecast")
        sim = trained_vectorizer.cosine_similarity(vec1, vec2)

        assert sim < 0.3  # Should be very low

    def test_cosine_similarity_similar(self, trained_vectorizer):
        """Similar documents should have high similarity"""
        vec1 = trained_vectorizer.transform("bitcoin cryptocurrency price")
        vec2 = trained_vectorizer.transform("bitcoin market price")
        sim = trained_vectorizer.cosine_similarity(vec1, vec2)

        assert sim > 0.3  # Should be reasonably high

    def test_search_returns_ranked_results(self, trained_vectorizer):
        """Search should return ranked results"""
        results = trained_vectorizer.search("bitcoin cryptocurrency", top_k=3)

        assert len(results) <= 3
        assert len(results) > 0

        # Results should be sorted by score descending
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_result_relevant(self, trained_vectorizer, sample_documents):
        """Top result should be most relevant"""
        results = trained_vectorizer.search("bitcoin cryptocurrency", top_k=1)

        top_idx = results[0][0]
        top_doc = sample_documents[top_idx]

        assert 'bitcoin' in top_doc.lower()

    def test_search_respects_top_k(self, trained_vectorizer):
        """Should return at most top_k results"""
        results = trained_vectorizer.search("bitcoin", top_k=2)
        assert len(results) <= 2

        results = trained_vectorizer.search("bitcoin", top_k=10)
        assert len(results) <= 10

    def test_empty_documents(self):
        """Should handle empty document list"""
        vec = TFIDFVectorizer()
        vec.fit([])

        assert len(vec.vocabulary) == 0
        assert len(vec.doc_vectors) == 0

    def test_empty_query(self, trained_vectorizer):
        """Should handle empty query"""
        vec = trained_vectorizer.transform("")
        assert isinstance(vec, dict)

    def test_unknown_terms_in_query(self, trained_vectorizer):
        """Should handle unknown terms gracefully"""
        vec = trained_vectorizer.transform("xyznonexistent foobar")

        # Should return empty or very sparse vector
        assert isinstance(vec, dict)


# ============================================================================
# SIMILARITY SCORING TESTS
# ============================================================================

class TestSimilarityScoring:
    """Tests for similarity calculations"""

    def test_self_similarity_is_one(self):
        """Document compared to itself should score ~1"""
        docs = ["bitcoin price trading market analysis"]
        vec = TFIDFVectorizer()
        vec.fit(docs)

        results = vec.search("bitcoin price trading market analysis", top_k=1)
        assert results[0][1] > 0.9

    def test_partial_match_scores_lower(self):
        """Partial matches should score lower than exact"""
        docs = [
            "bitcoin price trading market analysis forecast",
            "weather forecast sunny cloudy temperature"
        ]
        vec = TFIDFVectorizer()
        vec.fit(docs)

        results = vec.search("bitcoin price trading", top_k=2)

        # First doc should score higher
        assert results[0][0] == 0  # Bitcoin doc
        assert results[0][1] > results[1][1]

    def test_idf_weights_rare_terms(self):
        """Rare terms should be weighted higher"""
        docs = [
            "the market is trading now",
            "unique rare special specific",
            "the market is open now"
        ]
        vec = TFIDFVectorizer()
        vec.fit(docs)

        # IDF for rare word should be higher
        assert vec.idf.get('unique', 0) > vec.idf.get('market', 0)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestSearchEdgeCases:
    """Tests for edge cases"""

    def test_single_document(self):
        """Should work with single document"""
        vec = TFIDFVectorizer()
        vec.fit(["single document here"])

        results = vec.search("document", top_k=5)
        assert len(results) == 1

    def test_duplicate_documents(self):
        """Should handle duplicate documents"""
        vec = TFIDFVectorizer()
        vec.fit(["same text", "same text", "different text"])

        results = vec.search("same text", top_k=3)
        assert len(results) == 3

    def test_very_long_document(self):
        """Should handle long documents"""
        long_doc = " ".join(["word"] * 1000)
        vec = TFIDFVectorizer()
        vec.fit([long_doc, "short doc"])

        results = vec.search("word", top_k=2)
        assert len(results) == 2

    def test_special_characters_in_search(self):
        """Should handle special chars in search"""
        vec = TFIDFVectorizer()
        vec.fit(["normal document text"])

        vec.transform("search!!! @#$% query")
        # Should not crash

    def test_numbers_in_documents(self):
        """Should handle numbers"""
        vec = TFIDFVectorizer()
        vec.fit(["price is 100 dollars", "bitcoin at 50000"])

        results = vec.search("100 dollars", top_k=2)
        assert len(results) > 0

    def test_hyphenated_words(self):
        """Should handle hyphenated words"""
        vec = TFIDFVectorizer()
        vec.fit(["high-volume trading", "low-risk investment"])

        results = vec.search("high-volume", top_k=2)
        assert len(results) > 0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestSearchPerformance:
    """Tests for performance characteristics"""

    def test_handles_moderate_corpus(self):
        """Should handle moderately sized corpus"""
        docs = [f"document number {i} with some text content" for i in range(100)]
        vec = TFIDFVectorizer()
        vec.fit(docs)

        results = vec.search("document text", top_k=10)
        assert len(results) == 10

    def test_search_speed_reasonable(self):
        """Search should complete quickly"""
        import time

        docs = [f"document {i} with various content words" for i in range(500)]
        vec = TFIDFVectorizer()
        vec.fit(docs)

        start = time.time()
        for _ in range(10):
            vec.search("document content words", top_k=10)
        elapsed = time.time() - start

        # 10 searches should complete in under 2 seconds
        assert elapsed < 2.0


# ============================================================================
# INTEGRATION WITH MARKET DATA
# ============================================================================

class TestMarketSearchIntegration:
    """Integration tests for market search"""

    def test_market_like_documents(self):
        """Should work with market-like documents"""
        docs = [
            "event jan-2026 bracket 200-220 elon tweets high volume",
            "event jan-2026 bracket 220-240 elon tweets medium volume",
            "event feb-2026 bracket 100-120 elon tweets low volume",
            "bitcoin price prediction cryptocurrency market",
        ]
        vec = TFIDFVectorizer()
        vec.fit(docs)

        # Search for high volume Elon
        results = vec.search("elon tweets high volume", top_k=3)
        assert results[0][0] == 0  # First doc should match best

        # Search for bitcoin
        results = vec.search("bitcoin crypto", top_k=3)
        assert results[0][0] == 3  # Bitcoin doc should match

    def test_bracket_search(self):
        """Should find specific brackets"""
        docs = [
            "bracket 200-220 tweets event low",
            "bracket 300-320 tweets event medium",
            "bracket 400-420 tweets event high",
        ]
        vec = TFIDFVectorizer()
        vec.fit(docs)

        results = vec.search("300-320 medium", top_k=1)
        # Should find doc with "300-320" and "medium"
        assert results[0][0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
