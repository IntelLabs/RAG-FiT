import random

from ..step import GlobalStep


class ShuffleSelect(GlobalStep):
    """
    Class to optionally shuffle and select a subset of the dataset.

    Based on the `shuffle` and `select` methods of HF Dataset.
    """

    def __init__(self, shuffle=None, limit=None, **kwargs):
        """
        Args:
            shuffle (int): Seed for shuffling the dataset.
            limit (int): Number of items to select from the dataset.
        """
        super().__init__(**kwargs)
        self.shuffle = shuffle
        self.limit = limit

    def process_all(self, dataset, datasets, **kwargs):
        if self.shuffle:
            dataset = dataset.shuffle(seed=self.shuffle)
        if self.limit:
            dataset = dataset.select(range(self.limit))
        return dataset


class Sampler(GlobalStep):
    """
    Class to augment a dataset with sampled examples from the same or another dataset.

    Full examples can be collected, as well as an individual example keys like `query`, `documents`, etc.

    The step can be used to collect negative documents, negative queries and collect fewshot examples.
    For fewshot examples, use the dedicated `FewShot` class.
    """

    def __init__(
        self, k, input_key=None, output_key="fewshot", input_dataset=None, **kwargs
    ):
        """
        Args:
            k (int): Number of examples to collect.
            input_key (str): a key to collect from the collected examples, or None to take entire example.
            output_key (str): output key to use for the examples.
            input_dataset (str): Name of the dataset to take the examples from. To use the same dataset, use None.
        """
        super().__init__(**kwargs)
        self.k = k
        self.input_key = input_key
        self.input_dataset = input_dataset
        self.output_key = output_key

    def process(self, dataset_name, datasets, **kwargs):
        input_dataset = datasets[self.input_dataset or dataset_name]

        def find_examples(item, idx):
            ids = []
            while len(ids) < self.k:
                rand_idx = random.randint(0, len(input_dataset) - 1)
                if self.input_dataset is None and rand_idx == idx:
                    continue
                if rand_idx in ids:
                    continue
                ids.append(rand_idx)
            examples = [
                (
                    input_dataset[id_]
                    if self.input_key is None
                    else input_dataset[id_][self.input_key]
                )
                for id_ in ids
            ]
            item[self.output_key] = examples if self.k > 1 else examples[0]
            return item

        datasets[dataset_name] = datasets[dataset_name].map(
            lambda item, index: find_examples(item, index),
            with_indices=True,
            load_from_cache_file=False,
        )


class FewShot(Sampler):
    """
    Class to collect fewshot examples from the same or another dataset.
    """

    def __init__(self, k, output_key="fewshot", input_dataset=None, **kwargs):
        """
        Args:
            k (int): Number of examples to collect.
            output_key (str): output key to use for the collected examples.
            input_dataset (str): Name of the dataset to take the examples from. To use the same dataset, use None.
        """
        super().__init__(
            k=k,
            output_key=output_key,
            input_key=None,
            input_dataset=input_dataset,
            **kwargs
        )
