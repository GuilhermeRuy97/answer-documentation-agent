"""Unit tests for retrieval/fusion.py (Reciprocal Rank Fusion)."""

from retrieval.fusion import reciprocal_rank_fusion


def _chunk(chunk_id: str, url: str = "u", index: int = 0) -> dict:
    return {"id": chunk_id, "source_url": url, "chunk_index": index, "content": chunk_id}


class TestReciprocalRankFusion:
    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_list_preserves_order(self):
        results = [[_chunk("a"), _chunk("b"), _chunk("c")]]
        fused = reciprocal_rank_fusion(results)
        assert [c["id"] for c in fused] == ["a", "b", "c"]

    def test_chunk_in_multiple_lists_ranks_first(self):
        # "shared" is rank 2 in both lists; "a" and "b" are rank 1 in one list each.
        # RRF: shared = 2/(k+2) > a = 1/(k+1) for k >= 1 is false... with k=60:
        # shared = 2/62 = 0.0323, a = 1/61 = 0.0164 -> shared wins.
        list1 = [_chunk("a"), _chunk("shared")]
        list2 = [_chunk("b"), _chunk("shared")]
        fused = reciprocal_rank_fusion([list1, list2], rrf_k=60)
        assert fused[0]["id"] == "shared"

    def test_deduplicates_by_id(self):
        list1 = [_chunk("a"), _chunk("b")]
        list2 = [_chunk("a"), _chunk("c")]
        fused = reciprocal_rank_fusion([list1, list2])
        ids = [c["id"] for c in fused]
        assert sorted(ids) == ["a", "b", "c"]
        assert len(ids) == len(set(ids))

    def test_fusion_score_annotated_and_descending(self):
        list1 = [_chunk("a"), _chunk("b"), _chunk("c")]
        list2 = [_chunk("b")]
        fused = reciprocal_rank_fusion([list1, list2])
        scores = [c["fusion_score"] for c in fused]
        assert all(isinstance(s, float) for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_falls_back_to_url_and_index_without_id(self):
        c1 = {"source_url": "u1", "chunk_index": 0, "content": "x"}
        c2 = {"source_url": "u1", "chunk_index": 0, "content": "x"}
        fused = reciprocal_rank_fusion([[c1], [c2]])
        assert len(fused) == 1
