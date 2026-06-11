"""Unit tests for API schemas validation."""

import pytest
from pydantic import ValidationError

from api.schemas import AskRequest, FeedbackRequest


class TestAskRequest:
    def test_valid_question(self):
        req = AskRequest(question="  What are XML tags?  ")
        assert req.question == "What are XML tags?"
        assert req.session_id is None

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            AskRequest(question="   ")

    def test_overlong_question_rejected(self, safe_settings):
        safe_settings.max_question_chars = 100
        with pytest.raises(ValidationError):
            AskRequest(question="x" * 101)

    def test_overlong_session_id_rejected(self):
        with pytest.raises(ValidationError):
            AskRequest(question="q", session_id="s" * 129)


class TestFeedbackRequest:
    def test_valid_feedback(self):
        req = FeedbackRequest(trace_id="abc", score=1.0)
        assert req.score == 1.0
        assert req.comment is None

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(trace_id="abc", score=1.5)
        with pytest.raises(ValidationError):
            FeedbackRequest(trace_id="abc", score=-0.1)
