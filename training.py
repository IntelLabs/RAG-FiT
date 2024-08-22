import logging
import os
from pathlib import Path

import hydra
import wandb
from datasets import load_dataset
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

logger = logging.getLogger(__name__)


def setup_wandb(args: dict):
    """
    WANDB integration for tracking training runs.
    """
    env = {key: os.getenv(key) for key in os.environ}
    run = wandb.init(
        job_type="train",
        project=args["experiment"],
        entity=args["wandb_entity"],
        config={**args, **env},
        tags=["train"],
    )
    return run


@hydra.main(version_base=None, config_path="./configs", config_name="training")
def main(args):
    logger.info(OmegaConf.to_yaml(args))
    OmegaConf.set_struct(args, False)

    logger.info(f"Experiment name: {args.experiment}")
    logger.info(f"Output path: {args.train.output_dir}")

    if args.use_wandb:
        run = setup_wandb(OmegaConf.to_container(args))

    logger.info(f"Loading dataset: {args.data_file}")
    dataset = load_dataset(
        "json", data_files=to_absolute_path(args.data_file), split="train"
    )

    logger.info(f"Loading instruction from file {args.instruction}...")
    instruction = open(args.instruction).read()
    logger.info(f"Loaded instruction: {instruction}")

    if args.shuffle:
        dataset = dataset.shuffle(seed=args.shuffle)

    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    model_class = hydra.utils.instantiate(args.model, _convert_="object")
    logger.info("Model was loaded.")

    def format_answer(example):
        query = example[args.input_key]
        if args.model.instruction_in_prompt:
            query = instruction + "\n" + query

        output = (
            out[0] if isinstance(out := example[args.output_key], list) else out
        ) or ""

        if args.template:
            return open(args.template).read().format(query=query, output=output)
        else:
            messages = [
                {
                    "role": "system",
                    "content": instruction,
                },
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "content": output,
                },
            ]

            return dict(messages=messages)

    dataset = dataset.map(format_answer)

    collator = DataCollatorForCompletionOnlyLM(
        model_class.tokenizer.encode(
            args.model.completion_start, add_special_tokens=False
        ),
        tokenizer=model_class.tokenizer,
    )

    logger.info("Initializing training arguments...")
    training_args = TrainingArguments(**args.train)

    logger.info("Starting to train...")
    trainer = SFTTrainer(
        model=model_class.model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        dataset_batch_size=1,
        packing=False,
        max_seq_length=args.model.max_sequence_len,
        dataset_kwargs=dict(add_special_tokens=False),
    )
    trainer.train(resume_from_checkpoint=args.resume_checkpoint)

    logger.info(
        f"Finished training; saving model to {args.train.output_dir}/checkpoint..."
    )

    trainer.model.save_pretrained(Path(args.train.output_dir) / "checkpoint/")

    if args.hfhub_tag:
        trainer.model.push_to_hub(args.hfhub_tag, private=True)


if __name__ == "__main__":
    main()
