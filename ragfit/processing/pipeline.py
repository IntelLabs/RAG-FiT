import logging
import math
import os
from typing import List

import hydra
from datasets import load_dataset
from tqdm import tqdm

from .step import BaseStep


class DataPipeline:
    """Class for creating a data pipeline.

    The pipeline holds the list of steps and run them one after the other. The
    datasets are stored in a global dictionary, where datasets are referred by a
    key name, as indicated in the `inputs` parameter for each step. The pipeline
    manages the cache lookup and creation.

    Args:
        name (str): Name of the pipeline.
        output_path (str, optional): Path to store the cache files. Defaults to ".".
        cache (bool, optional): Whether to cache the datasets. Defaults to True.
        steps (List[BaseStep], optional): List of steps in the pipeline. Defaults to [].
        inputs (str, optional): Name of the main dataset. Defaults to "main_dataset".

    """

    def __init__(
        self,
        name,
        output_path=".",
        cache=True,
        steps: List[BaseStep] = [],
        inputs: str = "main_dataset",
        **kwargs,
    ) -> None:
        self.name = name
        self.output_path = output_path
        self.cache = cache
        logging.info(f"Caching state: {self.cache}")
        self.last_update = math.inf

        self.steps = [
            hydra.utils.instantiate(step, _convert_="object") for step in steps
        ]  # TODO: do it lazily to prevent OOM

        self.inputs = inputs if isinstance(inputs, list) else [inputs]
        self.datasets = {}

    def gen_cache_fn(self, step, index, dataset_name):
        """
        Create a unique cache filename for  a given dataset, at a given step, in a given index.
        Uses the step name, inputs, hash and pipeline's path and name and dataset name.

        Returns a string.
        """
        return (
            f"{self.output_path}/cache"
            f"_{self.name}_{index}"
            f"_{type(step).__name__}"
            f"_{dataset_name}_{step.get_hash()}.json"
        )

    def get_cache_mapping(self, step: BaseStep, index: int):
        """
        Returns a mapping between input datasets and cache filenames, for a given step.
        """
        if self.cache:
            datasets_caches = {
                dataset_name: self.gen_cache_fn(step, index, dataset_name)
                for dataset_name in step.inputs
            }
            return datasets_caches

        return None

    def cache_step(self, step, step_index):
        """
        Write to cache-files the current state of the global datasets dictionary for the given inputs.
        """
        if self.cache:
            for dataset_name in step.inputs:
                dataset = self.datasets[dataset_name]
                saved_path = self.gen_cache_fn(step, step_index, dataset_name)
                dataset.to_json(saved_path, lines=True)

    def load_from_cache(self, caches_map):
        """
        Load datasets from cache using a cache_map.
        Updates the global datasets dictionary.

        Internal function, shouldn't be used by the user.
        """
        logging.info(f"Loading dataset from checkpoints {caches_map}")
        for dataset_name, saved_path in caches_map.items():
            self.datasets[dataset_name] = load_dataset(
                "json", data_files=[saved_path], split="train"
            )

    def delete_cache(self):
        """
        Removing cache files for all steps, cleaning the pipeline.
        """
        logging.info("Removing cache files for entire pipeline.")
        if self.cache:
            for i, step in enumerate(self.steps):
                cache_map = self.get_cache_mapping(step, i)
                if cache_map is not None:
                    for dataset_name, cache_path in cache_map.items():
                        if os.path.exists(cache_path):
                            os.remove(cache_path)

    def process(self):
        """
        Run pipeline, step after step.

        Caching is handled here. A step is calculated either if there was a change in the pipeline at a previous
        step OR the current step has no cache file.

        When a step is calculated, it is cached and self.last_update is updated to the current step index.
        """
        for i, step in tqdm(enumerate(self.steps)):
            logging.info(f"Processing step {i}")

            cache_map = self.get_cache_mapping(step, i)
            if (
                (cache_map is not None)
                and (all(os.path.exists(v) for v in cache_map.values()))
                and (i < self.last_update)
            ):
                logging.info(f"Loading cached datasets for {type(step).__name__}")
                self.load_from_cache(cache_map)
            else:
                step(self.datasets)
                if step.cache_step:
                    self.cache_step(step, i)
                    self.last_update = i
