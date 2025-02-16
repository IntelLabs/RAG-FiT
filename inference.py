import json
import logging
from functools import partial
from pathlib import Path

import hydra
from datasets import load_dataset
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="./configs", config_name="inference")
def main(args):
    logging.info(OmegaConf.to_yaml(args))

    logging.info(f"Loading data file: {args.data_file}")
    data = load_dataset(
        "json", data_files=to_absolute_path(args.data_file), split="train"
    )

    model = hydra.utils.instantiate(args.model, _convert_="object")
    logging.info(f"Loaded model: {model}")

    logging.info(f"Generated (opt. cache) file: {args.generated_file}")

    if Path(args.generated_file).exists():
        saved_data = load_dataset("json", data_files=args.generated_file, split="train")
    else:
        saved_data = []

    if args.limit:
        data = data.select(range(args.limit))

    def map_generate(model, example, idx):
        if idx >= len(saved_data):
            out = model.generate(example)
            example[args.generation_key] = out

            with open(args.generated_file, "a") as f:
                f.write(
                    json.dumps({"text": out, "target": example[args.target_key]}) + "\n"
                )

        else:
            example[args.generation_key] = saved_data[idx]["text"]

        return example

    data = data.map(partial(map_generate, model), with_indices=True)


if __name__ == "__main__":
    main()
