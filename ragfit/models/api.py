"""LiteLLM model for inference."""

import logging

from openai import AzureOpenAI


class AzureLLM:
    def __init__(
        self,
        instruction,
        template=None,
        template_map=None,
        extra_keys=None,
        examples=None,
        max_tokens=100,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.instruction = open(instruction).read()
        self.template = open(template).read() if template else None
        self.template_map = template_map
        self.extra_keys = extra_keys
        self.examples = open(examples).read() if examples else None
        self.max_tokens = max_tokens
        self.model = AzureOpenAI(
            timeout=600,
            max_retries=10,
            **kwargs,
        )

    def generate(self, example: dict) -> str:
        """
        Given an input, generate a response.
        """

        # We move task instruction to system instruction which requires template filling.
        instruction = self.instruction.format(
            **{k: v for k, v in self.extra_keys.items()} if self.extra_keys else {},
        )

        prompt = self.template.format(
            **{k: example[v] for k, v in self.template_map.items()},
            **{k: v for k, v in self.extra_keys.items()} if self.extra_keys else {},
        )

        # prompt = (
        #     self.template.format(**{k: example[v] for k, v in self.template_map.items()})
        #     if (self.template and self.template_map)
        #     else example["prompt"]
        # )

        if self.examples:
            examples = self.examples.split("####")
            assert len(examples) % 2 == 0, "Examples must be in pairs"
            fewshot = []
            for i in range(0, len(examples), 2):
                fewshot.append({"role": "user", "content": examples[i].strip()})
                fewshot.append({"role": "assistant", "content": examples[i + 1].strip()})

        messages = [
            {"role": "system", "content": instruction},
        ]

        if self.examples:
            messages.extend(fewshot)

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.model.chat.completions.create(
                model=self.kwargs["azure_deployment"],
                messages=messages,
                max_tokens=self.max_tokens,
            )

            message_obj = response.choices[0].message

            if hasattr(message_obj, "content"):
                answer = message_obj.content
                return answer or ""
            else:
                return ""

        except Exception as e:
            logging.info(f"API error:\n{e}")
            return "No response."
