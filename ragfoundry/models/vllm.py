import logging
from typing import Dict
from vllm import LLM, SamplingParams

from ragfoundry.utils import check_package_installed


logger = logging.getLogger(__name__)


class VLLMInference:
    """
    Initializes a vLLM-based inference engine.

    Args:
        model_name_or_path (str): The name or path of the model.
        num_gpus (int, optional): The number of GPUs to use. Defaults to 1.
        llm_params (Dict, optional): Additional parameters for the LLM model. Supports all parameters define by vLLM LLM engine. Defaults to an empty dictionary.
        generation (Dict, optional): Additional parameters for text generation. Supports all the keywords of `SamplingParams` of vLLM. Defaults to an empty dictionary.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_gpus: int = 1,
        llm_params: Dict = {},
        generation: Dict = {},
    ):
        check_package_installed("vllm")
        self.sampling_params = SamplingParams(**generation)
        self.llm = LLM(model=model_name_or_path, tensor_parallel_size=num_gpus, **llm_params)

    def generate(self, prompt: str) -> str:
        """
        Generates text based on the given prompt.
        """
        output = self.llm.generate(prompt, self.sampling_params)
        return output[0].outputs[0].text
