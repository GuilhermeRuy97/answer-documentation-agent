"""Unit tests for agent/citations.py."""

from agent.citations import build_citations


def _chunk(url: str, title: str = "Title", content: str = "Some documentation content here.") -> dict:
    return {"source_url": url, "page_title": title, "content": content}


class TestBuildCitations:
    def test_basic_markers(self):
        chunks = [_chunk("https://a.com"), _chunk("https://b.com")]
        result = build_citations("Fact one [1]. Fact two [2].", chunks)
        assert len(result["citations"]) == 2
        assert result["citations"][0]["url"] == "https://a.com"
        assert "[1]" in result["final_response"]
        assert "[2]" in result["final_response"]

    def test_dedupes_same_url(self):
        chunks = [_chunk("https://a.com"), _chunk("https://a.com"), _chunk("https://b.com")]
        result = build_citations("X [1] Y [2] Z [3]", chunks)
        # Chunks 1 and 2 share a URL -> one citation; markers collapse to [1].
        assert len(result["citations"]) == 2
        assert result["final_response"] == "X [1] Y [1] Z [2]"

    def test_drops_invalid_markers(self):
        chunks = [_chunk("https://a.com")]
        result = build_citations("Valid [1] invalid [9]", chunks)
        assert len(result["citations"]) == 1
        assert "[9]" not in result["final_response"]
        assert "[1]" in result["final_response"]

    def test_fallback_top_three_when_no_markers(self):
        chunks = [_chunk(f"https://site{i}.com") for i in range(5)]
        result = build_citations("No markers in this answer.", chunks)
        assert len(result["citations"]) == 3
        assert result["final_response"] == "No markers in this answer."

    def test_no_chunks_no_citations(self):
        result = build_citations("Answer without context.", [])
        assert result["citations"] == []
        assert result["final_response"] == "Answer without context."

    def test_snippet_truncated(self):
        chunks = [_chunk("https://a.com", content="x" * 500)]
        result = build_citations("Fact [1]", chunks)
        assert len(result["citations"][0]["snippet"]) <= 124
