import logging
from pathlib import Path
from typing import Dict

from transformers import AutoConfig, AutoTokenizer

from ragfoundry.utils import check_package_installed

logger = logging.getLogger(__name__)


class VLLMInference:
    """
    Initializes a vLLM-based inference engine.

    Args:
        model_name_or_path (str): The name or path of the model.
        instruction (Path): path to the instruction file.
        instruct_in_prompt (bool): whether to include the instruction in the prompt for models without system role.
        template (Path): path to a prompt template file if tokenizer does not include chat template. Optional.
        num_gpus (int, optional): The number of GPUs to use. Defaults to 1.
        llm_params (Dict, optional): Additional parameters for the LLM model. Supports all parameters define by vLLM LLM engine. Defaults to an empty dictionary.
        generation (Dict, optional): Additional parameters for text generation. Supports all the keywords of `SamplingParams` of vLLM. Defaults to an empty dictionary.
    """

    def __init__(
        self,
        model_name_or_path: str,
        instruction: Path,
        instruct_in_prompt: False,
        template: Path = None,
        num_gpus: int = 1,
        llm_params: Dict = {},
        generation: Dict = {},
    ):
        check_package_installed(
            "vllm",
            "please refer to vLLM website for installation instructions, or run: pip install vllm",
        )
        from vllm import LLM, SamplingParams

        logger.info(f"Using the following instruction: {self.instruction}")

        self.instruct_in_prompt = instruct_in_prompt
        self.template = open(template).read() if template else None
        self.instruction = open(instruction).read()

        self.sampling_params = SamplingParams(**generation)
        self.llm = LLM(
            model=model_name_or_path, tensor_parallel_size=num_gpus, **llm_params
        )
        if self.instruct_in_prompt:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.config = AutoConfig.from_pretrained(self.model_name)

    def generate(self, prompt: str) -> str:
        """
        Generates text based on the given prompt.
        """
        if self.template:
            prompt = self.template.format(instruction=self.instruction, query=prompt)
        elif self.instruct_in_prompt:
            prompt = self.instruction + "\n" + prompt
            messages = [
                {"role": "system", "content": self.instruction},
                {"role": "user", "content": prompt},
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                truncation=True,
                max_length=(
                    self.config.max_position_embeddings - self.sampling_param.max_tokens
                ),
            )

        output = self.llm.generate(prompt, self.sampling_params)
        return output[0].outputs[0].text
