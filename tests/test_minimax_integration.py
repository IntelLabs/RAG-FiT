"""Integration tests for MiniMax executor (require MINIMAX_API_KEY)."""

import os

import pytest

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
skip_no_key = pytest.mark.skipif(
    not MINIMAX_API_KEY,
    reason="MINIMAX_API_KEY not set",
)


@skip_no_key
class TestMiniMaxExecutorIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_chat_basic(self):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        executor = MiniMaxExecutor(delay=0)
        result = executor.chat("What is 1 + 1? Reply with just the number.")
        assert result.strip(), "Expected non-empty response"

    def test_chat_with_instruction(self):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        executor = MiniMaxExecutor(delay=0)
        result = executor.chat(
            "What is RAG?",
            instruction="Reply in one short sentence.",
        )
        assert len(result) > 0

    def test_generate_interface(self):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        executor = MiniMaxExecutor(delay=0)
        result = executor.generate("Say hello in French.")
        assert result.strip(), "Expected non-empty response from generate()"
