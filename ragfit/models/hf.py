import logging
from pathlib import Path

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class HFInference:
    """
    Class for running HF model inference locally.
    """

    def __init__(
        self,
        model_name_or_path: str,
        instruction: Path,
        torch_dtype=None,
        device_map="auto",
        instruct_in_prompt=False,
        template: Path = None,
        lora_path=None,
        lora_path_2nd=None,
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.config = AutoConfig.from_pretrained(self.model_name, **kwargs)
        self.config.torch_dtype = torch_dtype or "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, config=self.config, device_map=device_map, **kwargs
        )
        if lora_path_2nd:
            assert lora_path, "2nd LORA requires 1st LORA"
            logger.info(f"Loading LORA: {lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model, lora_path, device_map=device_map, **kwargs
            )
            self.model = self.model.merge_and_unload(safe_merge=True)
            delattr(self.model, "peft_config")
            logger.info(f"Loading 2nd LORA: {lora_path_2nd}")
            self.model.load_adapter(lora_path_2nd)

        elif lora_path:
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

        if not isinstance(prompt, str):
            prompt = prompt["prompt"]

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


class HFInferencePrompt(HFInference):
    """
    Class for running HF model inference locally with a prompt.
    """

    def __init__(
        self,
        model_name_or_path: str,
        instruction: Path,
        torch_dtype=None,
        device_map="auto",
        instruct_in_prompt=False,
        template: Path = None,
        template_map: dict = None,
        extra_keys: dict = None,
        examples: Path = None,
        lora_path=None,
        lora_path_2nd=None,
        generation=None,
        task="text-generation",
        **kwargs,
    ):
        self.template_map = template_map
        self.extra_keys = extra_keys
        self.examples = open(examples).read() if examples else None
        super().__init__(
            model_name_or_path,
            instruction,
            torch_dtype,
            device_map,
            instruct_in_prompt,
            template,
            lora_path,
            lora_path_2nd,
            generation,
            task,
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
        trained_adapter=None,
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
            trained_adapter: str - path to a trained adapter. It will be loaded and merged.
            generation: dict - generation kwargs.
            completion_start: str - used to find the start of the completion in the prompt.
            instruction_in_prompt: bool - whether to include the instruction in the prompt for models without system role.
        """
        self.model_name = model_name_or_path
        self.complete_start = completion_start
        self.instructions_in_prompt = instruction_in_prompt
        self.generation_kwargs = generation
        self.max_sequence_len = max_sequence_len
        self.trained_adapter = trained_adapter

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

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

        if self.trained_adapter:
            logger.info(f"Loading and merging trained adapter: {self.trained_adapter}")
            self.model = PeftModel.from_pretrained(self.model, self.trained_adapter)
            self.model = self.model.merge_and_unload(safe_merge=True)
            delattr(self.model, "peft_config")

        logger.info(f"Initializing LORA based on {lora}")
        self.model = get_peft_model(self.model, LoraConfig(**lora))
