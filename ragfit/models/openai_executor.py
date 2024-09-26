import logging
import os
import time
from typing import List, Union

from openai import AzureOpenAI


class OpenAIExecutor:
    """
    Class representing an interface to the Azure OpenAI API.
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str = None,
        api_version: str = "2024-02-15-preview",
        model: str = "GPT-4-32k-Bot",
        chat_parameters: dict = None,
        delay: int = 1,
    ):
        """
        Initialize the OpenAIExecutor.

        Args:
            azure_endpoint (str): The Azure endpoint.
            api_key (str): The API key, can also read of ENV variable.
            api_version (str): The API version.
            model (str): The model to use, sometimes called deployment or engine.
            chat_parameters (dict): The chat parameters.
            delay (int): delay between calls.
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

        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=api_version,
        )

    def chat(self, prompt: Union[List, str], instruction: str = None) -> str:
        """
        Chat with the OpenAI API.

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
            logging.info(f"OPENAI error:\n{e}")
            return ""
