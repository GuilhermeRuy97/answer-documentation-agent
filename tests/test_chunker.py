"""Unit tests for ingestion/chunker.py."""

from ingestion.chunker import _clean_markdown, chunk_pages, content_hash


def _page(markdown: str, url: str = "https://example.com/docs/xml-tags", title: str = "Use XML tags") -> dict:
    return {"url": url, "title": title, "markdown": markdown}


class TestCleanMarkdown:
    def test_removes_image_markdown(self):
        text = "Before ![alt text](https://img.example.com/x.png) after"
        assert "img.example.com" not in _clean_markdown(text)
        assert "Before" in _clean_markdown(text)

    def test_removes_boilerplate_lines(self):
        text = "Real content\nWas this page helpful?\nCopy page\nMore content"
        cleaned = _clean_markdown(text)
        assert "Was this page helpful?" not in cleaned
        assert "Copy page" not in cleaned
        assert "Real content" in cleaned
        assert "More content" in cleaned

    def test_collapses_blank_lines(self):
        text = "a\n\n\n\n\nb"
        assert _clean_markdown(text) == "a\n\nb"


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_differs_for_different_content(self):
        assert content_hash("hello") != content_hash("world")


class TestChunkPages:
    def test_chunk_metadata_and_contextual_header(self, safe_settings):
        body = "## Why use XML tags\n\n" + "XML tags help Claude parse prompts accurately. " * 10
        chunks = chunk_pages([_page(body)])

        assert len(chunks) >= 1
        first = chunks[0]
        assert first["source_url"] == "https://example.com/docs/xml-tags"
        assert first["page_title"] == "Use XML tags"
        assert first["chunk_index"] == 0
        # Contextual header: page title prepended to the content
        assert first["content"].startswith("Use XML tags")
        assert first["content_hash"] == content_hash(first["content"])

    def test_skips_chunks_below_min_chars(self, safe_settings):
        safe_settings.min_chunk_chars = 5000
        chunks = chunk_pages([_page("Short page content that is under the minimum.")])
        assert chunks == []

    def test_heading_included_in_header(self, safe_settings):
        body = "## Tagging best practices\n\n" + "Be consistent with tag names throughout your prompts. " * 8
        chunks = chunk_pages([_page(body)])
        assert any("Tagging best practices" in c["content"].splitlines()[0] for c in chunks)
