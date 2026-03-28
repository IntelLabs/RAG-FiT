import logging
import os
import time
from typing import List, Union

from openai import OpenAI


class MiniMaxExecutor:
    """
    Class representing an interface to the MiniMax API.

    MiniMax provides an OpenAI-compatible API at https://api.minimax.io/v1.
    Supported models include MiniMax-M2.7 and MiniMax-M2.7-highspeed.
    """

    MINIMAX_BASE_URL = "https://api.minimax.io/v1"

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "MiniMax-M2.7",
        chat_parameters: dict = None,
        delay: int = 1,
    ):
        """
        Initialize the MiniMaxExecutor.

        Args:
            api_key (str): The API key. Can also be read from MINIMAX_API_KEY env variable.
            base_url (str): Override the base URL. Defaults to MiniMax's OpenAI-compatible endpoint.
            model (str): The model to use. Defaults to "MiniMax-M2.7".
            chat_parameters (dict): The chat parameters.
            delay (int): Delay between calls in seconds.
        """
        self.delay = delay
        self.model = model
        self.chat_parameters = dict(
            temperature=0.7,
            max_tokens=200,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        if chat_parameters:
            self.chat_parameters.update(chat_parameters)

        # Clamp temperature to MiniMax's accepted range (0, 1]
        temp = self.chat_parameters.get("temperature", 0.7)
        if temp is not None:
            self.chat_parameters["temperature"] = max(0.01, min(1.0, temp))

        self.client = OpenAI(
            api_key=api_key or os.getenv("MINIMAX_API_KEY"),
            base_url=base_url or self.MINIMAX_BASE_URL,
        )

    def chat(self, prompt: Union[List, str], instruction: str = None) -> str:
        """
        Chat with the MiniMax API.

        Args:
            prompt (Union[List, str]): The prompt to chat.
            instruction (str): The instruction to use.

        Returns:
            str: The response. Empty string if error.
        """
        if isinstance(prompt, str):
            prompt = [
                {
                    "role": "system",
                    "content": (
                        instruction
                        or "You are an AI assistant that helps people find information."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

        if self.delay:
            time.sleep(self.delay)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                **self.chat_parameters,
            )
            message_obj = completion.choices[0].message

            if hasattr(message_obj, "content"):
                answer = message_obj.content
                return answer or ""
            else:
                return ""

        except Exception as e:
            logging.info(f"MiniMax error:\n{e}")
            return ""

    def generate(self, prompt: str, instruction: str = None) -> str:
        """
        Generate text based on the given prompt.

        This method provides a ``generate``-style interface consistent with
        HFInference and VLLMInference, so ``MiniMaxExecutor`` can be used
        directly in the inference config (``inference.py``).

        Args:
            prompt (str): The input prompt.
            instruction (str): Optional system instruction.

        Returns:
            str: The generated text.
        """
        return self.chat(prompt, instruction)
