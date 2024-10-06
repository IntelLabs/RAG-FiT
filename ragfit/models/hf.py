import logging
from pathlib import Path

from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class HFInference:
    """
    Class for running HF model inference locally.
    """

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype,
        device_map,
        instruction: Path,
        instruct_in_prompt: False,
        template: Path = None,
        lora_path=None,
        generation=None,
        task="text-generation",
        **kwargs,
    ):
        """
        Initialize a HF model, with optional LORA adapter.

        Args:
            model_name_or_path (str): HF model name or path.
            torch_dtype (str): torch dtype for the model.
            device_map: device map for the model.
            instruction (Path): path to the instruction file.
            instruct_in_prompt (bool): whether to include the instruction in the prompt for models without system role.
            template (Path): path to a prompt template file if tokenizer does not include chat template. Optional.
            lora_path (Path): path to the LORA adapter.
            generation (dict): generation kwargs.
            task (str): task for the pipeline.
        """

        self.model_name = model_name_or_path
        self.generation_kwargs = generation
        self.instruction = open(instruction).read()
        logger.info(f"Using the following instruction: {self.instruction}")

        self.instruct_in_prompt = instruct_in_prompt
        self.template = open(template).read() if template else None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)

        self.config = AutoConfig.from_pretrained(self.model_name, **kwargs)
        self.config.torch_dtype = torch_dtype or "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, config=self.config, device_map=device_map, **kwargs
        )
        if lora_path:
            logger.info(f"Loading LORA: {lora_path}")
            self.model.load_adapter(lora_path)

        self.pipe = pipeline(
            task=task,
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate(self, prompt: str) -> str:
        """
        Given an input, generate a response.
        """

        if self.template:
            prompt = self.template.format(instruction=self.instruction, query=prompt)

        else:
            if self.instruct_in_prompt:
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
                    self.config.max_position_embeddings
                    - self.generation_kwargs["max_new_tokens"]
                ),
            )

        output = self.pipe(prompt, **self.generation_kwargs)
        return output[0]["generated_text"]


class HFTrain:
    """
    Class for training HF models locally.
    """

    def __init__(
        self,
        model_name_or_path,
        torch_dtype,
        device_map,
        lora: LoraConfig = None,
        generation=None,
        completion_start: str = "",
        instruction_in_prompt=None,
        max_sequence_len=None,
        **kwargs,
    ):
        """
        Args:
            model_name_or_path: str - HF model name or path.
            torch_dtype: str - torch dtype for the model.
            device_map: dict - device map for the model.
            lora: dict - LoRA adapter config.
            generation: dict - generation kwargs.
            completion_start: str - used to find the start of the completion in the prompt.
            instruction_in_prompt: bool - whether to include the instruction in the prompt for models without system role.
        """
        self.model_name = model_name_or_path
        self.complete_start = completion_start
        self.instructions_in_prompt = instruction_in_prompt
        self.generation_kwargs = generation

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = AutoConfig.from_pretrained(self.model_name, **kwargs)
        self.config.torch_dtype = torch_dtype or "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            device_map=device_map,
            **kwargs,
        )

        self.model.config.use_cache = False
        logger.info(f"Loaded model: {self.model}")

        logger.info(f"Initializing LORA based on {lora}")
        self.model = get_peft_model(self.model, LoraConfig(**lora))
