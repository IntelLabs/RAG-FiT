"""Module for inference steps, which can use LLM output to augment the data."""

from ragfit.models.vllm import VLLMInference

from ..step import LocalStep


class HFStep(LocalStep):
    """
    Class for running inference with a Hugging Face model based on the vLLM engine.
    """

    def __init__(self, input_key, output_key, model_kwargs, **kwargs):
        """
        Initialize the HFStep class.

        Args:
                input_key (str): The key for the input text to be served as the prompt.
                output_key (str): The key for for saving the generated text.
                model_kwargs (dict): The keyword arguments to pass to the vLLM model.
                **kwargs: Additional keyword arguments to pass to the LocalStep.
        """
        super().__init__(**kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.model_kwargs = model_kwargs
        self.model = VLLMInference(**model_kwargs)

    def process_item(self, item, index, datasets, **kwargs):
        prompt = item[self.input_key]
        response = self.model.generate(prompt)
        item[self.output_key] = response
        return item
