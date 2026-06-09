"""Unit tests for MiniMaxExecutor."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestMiniMaxExecutorInit:
    """Test MiniMaxExecutor initialization."""

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_default_init(self, mock_openai):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        os.environ["MINIMAX_API_KEY"] = "test-key"
        executor = MiniMaxExecutor()

        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.minimax.io/v1",
        )
        assert executor.model == "MiniMax-M2.7"
        assert executor.delay == 1
        del os.environ["MINIMAX_API_KEY"]

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_custom_model(self, mock_openai):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        executor = MiniMaxExecutor(
            api_key="test-key", model="MiniMax-M2.7-highspeed"
        )
        assert executor.model == "MiniMax-M2.7-highspeed"

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_custom_base_url(self, mock_openai):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        executor = MiniMaxExecutor(
            api_key="test-key", base_url="https://custom.endpoint/v1"
        )
        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://custom.endpoint/v1",
        )

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_custom_chat_parameters(self, mock_openai):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        executor = MiniMaxExecutor(
            api_key="test-key",
            chat_parameters={"temperature": 0.5, "max_tokens": 500},
        )
        assert executor.chat_parameters["temperature"] == 0.5
        assert executor.chat_parameters["max_tokens"] == 500
        # Default values should be preserved for unset params
        assert executor.chat_parameters["top_p"] == 0.95

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_temperature_clamping_zero(self, mock_openai):
        """Temperature=0 should be clamped to 0.01 for MiniMax."""
        from ragfit.models.minimax_executor import MiniMaxExecutor

        executor = MiniMaxExecutor(
            api_key="test-key",
            chat_parameters={"temperature": 0.0},
        )
        assert executor.chat_parameters["temperature"] == 0.01

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_temperature_clamping_above_one(self, mock_openai):
        """Temperature > 1 should be clamped to 1.0 for MiniMax."""
        from ragfit.models.minimax_executor import MiniMaxExecutor

        executor = MiniMaxExecutor(
            api_key="test-key",
            chat_parameters={"temperature": 1.5},
        )
        assert executor.chat_parameters["temperature"] == 1.0

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_temperature_valid_range(self, mock_openai):
        """Temperature within (0, 1] should be unchanged."""
        from ragfit.models.minimax_executor import MiniMaxExecutor

        executor = MiniMaxExecutor(
            api_key="test-key",
            chat_parameters={"temperature": 0.3},
        )
        assert executor.chat_parameters["temperature"] == 0.3

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_env_api_key(self, mock_openai):
        """API key should fall back to MINIMAX_API_KEY env variable."""
        from ragfit.models.minimax_executor import MiniMaxExecutor

        os.environ["MINIMAX_API_KEY"] = "env-test-key"
        executor = MiniMaxExecutor()
        mock_openai.assert_called_once_with(
            api_key="env-test-key",
            base_url="https://api.minimax.io/v1",
        )
        del os.environ["MINIMAX_API_KEY"]


class TestMiniMaxExecutorChat:
    """Test MiniMaxExecutor chat method."""

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_chat_with_string_prompt(self, mock_openai):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        executor = MiniMaxExecutor(api_key="test-key", delay=0)
        result = executor.chat("Hello", "Be helpful")

        assert result == "Test response"
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_chat_with_message_list(self, mock_openai):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        messages = [
            {"role": "system", "content": "System msg"},
            {"role": "user", "content": "User msg"},
        ]

        executor = MiniMaxExecutor(api_key="test-key", delay=0)
        result = executor.chat(messages)

        assert result == "Response"
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["messages"] == messages

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_chat_default_instruction(self, mock_openai):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        executor = MiniMaxExecutor(api_key="test-key", delay=0)
        executor.chat("Hello")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert "AI assistant" in messages[0]["content"]

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_chat_error_returns_empty(self, mock_openai):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        executor = MiniMaxExecutor(api_key="test-key", delay=0)
        result = executor.chat("Hello")

        assert result == ""

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_chat_empty_content_returns_empty(self, mock_openai):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        executor = MiniMaxExecutor(api_key="test-key", delay=0)
        result = executor.chat("Hello")

        assert result == ""

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_chat_model_parameter(self, mock_openai):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "OK"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        executor = MiniMaxExecutor(
            api_key="test-key", model="MiniMax-M2.7-highspeed", delay=0
        )
        executor.chat("Hi")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "MiniMax-M2.7-highspeed"


class TestMiniMaxExecutorGenerate:
    """Test MiniMaxExecutor generate method (inference interface)."""

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_generate_delegates_to_chat(self, mock_openai):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Generated text"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        executor = MiniMaxExecutor(api_key="test-key", delay=0)
        result = executor.generate("prompt text")

        assert result == "Generated text"

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_generate_with_instruction(self, mock_openai):
        from ragfit.models.minimax_executor import MiniMaxExecutor

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Answer"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        executor = MiniMaxExecutor(api_key="test-key", delay=0)
        result = executor.generate("prompt text", instruction="Custom instruction")

        assert result == "Answer"
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["content"] == "Custom instruction"


class TestMiniMaxChatStep:
    """Test MiniMaxChat processing step."""

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_process_item(self, mock_openai, tmp_path):
        from ragfit.processing.local_steps.api.minimax import MiniMaxChat

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Generated answer"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        # Create a temporary instruction file
        instruction_file = tmp_path / "instruction.txt"
        instruction_file.write_text("You are a helpful assistant.")

        step = MiniMaxChat(
            model={"api_key": "test-key", "delay": 0},
            instruction=str(instruction_file),
            prompt_key="prompt",
            answer_key="answer",
        )

        item = {"prompt": "What is RAG?"}
        result = step.process_item(item, 0, {})

        assert result["answer"] == "Generated answer"

    @patch("ragfit.models.minimax_executor.OpenAI")
    def test_process_item_preserves_other_keys(self, mock_openai, tmp_path):
        from ragfit.processing.local_steps.api.minimax import MiniMaxChat

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Answer"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        instruction_file = tmp_path / "instruction.txt"
        instruction_file.write_text("Instruction")

        step = MiniMaxChat(
            model={"api_key": "test-key", "delay": 0},
            instruction=str(instruction_file),
            prompt_key="prompt",
            answer_key="answer",
        )

        item = {"prompt": "Q?", "context": "Some context", "id": 42}
        result = step.process_item(item, 0, {})

        assert result["answer"] == "Answer"
        assert result["context"] == "Some context"
        assert result["id"] == 42
