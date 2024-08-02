import logging
import os
from collections import defaultdict

import hydra
import torch
import yaml
from datasets import load_dataset
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_wandb(args: dict):
    """
    WANDB integration for tracking evaluations.
    """
    from wandb.wandb_run import Run

    import wandb

    env = {key: os.getenv(key) for key in os.environ}
    run: Run = wandb.init(
        job_type="eval",
        project=args["experiment"],
        entity=args["wandb_entity"],
        config={**args, **env},
        tags=["eval"],
    )
    return run


@hydra.main(version_base=None, config_path="./configs", config_name="evaluation")
def main(args):
    logger.info(OmegaConf.to_yaml(args))

    if args.use_wandb:
        run = setup_wandb(OmegaConf.to_container(args))

    logger.info(f"Loading dataset: {args.data_file}")
    data = load_dataset(
        "json", data_files=to_absolute_path(args.data_file), split="train"
    )

    generated_data = load_dataset("json", data_files=args.generated_file, split="train")
    logging.info(f"Loaded {len(generated_data)} examples from {args.generated_file}")

    if args.limit:
        data = data.select(range(args.limit))

    if args.answer_processor:
        answer_processor = hydra.utils.instantiate(
            args.answer_processor, _convert_="object"
        )
    else:

        def answer_processor(x):
            return x

    def map_load(example, idx):
        example[args.key_names["generated"]] = answer_processor(
            generated_data[idx][args.key_names["generated"]]
        )
        return example

    data = data.map(map_load, with_indices=True)
    size = len(data)

    results = {"local": defaultdict(list), "global": {}}
    for metric in args.metrics:
        obj = hydra.utils.instantiate(
            metric, key_names=args.key_names, _convert_="object"
        )
        if obj.local:
            for example in tqdm(data):
                calculation = obj.measure(example)
                for key, val in calculation.items():
                    results["local"][key].append(val)
        else:
            calculation = obj.measure(data)
            for key, val in calculation.items():
                results["global"][key] = val
        del obj
        torch.cuda.empty_cache()

    logging.info(f"Normalizing by size {size}")
    for key in results["local"].keys():
        results["local"][key] = float(sum(results["local"][key]) / size)
    results["local"] = dict(results["local"])

    logging.info(f"Results: {results}")
    if args.use_wandb:
        run.log(results, step=0)

    if args.results_file:
        with open(args.results_file, "w") as f:
            yaml.dump(results, f, sort_keys=True)
        logging.info(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
