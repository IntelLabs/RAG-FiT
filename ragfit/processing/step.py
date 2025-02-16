import logging

from .utils import dict_hash, is_jsonable


class BaseStep:
    """
    Class representing a step in a processing pipeline.
    Entry point is `__call__`.
    Users would inherit either LocalStep or GlobalStep.

    Step can be cached (on by default: `cache_step=True`) to prevent re-computation.

    Individual steps can disable caching if and only if they do not manipulate the dataset, as
    re-computation of later steps is conditioned on the necessity of caching.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.inputs: list[str] = kwargs.get("inputs", ["main_dataset"])
        self.step_hash = None
        self.cache_step = True

        if isinstance(self.inputs, str):
            self.inputs = [self.inputs]

        assert not isinstance(self.inputs, str) and len(self.inputs) > 0, (
            f"`inputs` should be a list, got {type(self.inputs)}"
        )

    def calc_hash(self):
        """
        Calculate hash for a step based on its properties.
        Updates the `step_hash` property.
        """
        args_to_hash = {}
        for property, value in vars(self).items():
            if is_jsonable(value):
                args_to_hash[property] = value
        self.step_hash = dict_hash(args_to_hash)

    def get_hash(self):
        """
        Step hash getter. If hash is not calculated, it calculates it first.
        """
        if self.step_hash is None:
            self.calc_hash()
        return self.step_hash

    def __call__(self, datasets, **kwargs):
        """
        Pipeline is running these steps using `__call__`.
        """
        logging.info(f"Running processing step: {type(self).__name__}")
        self.process_inputs(datasets, **kwargs)

    def process_inputs(self, datasets, **kwargs):
        """
        Run the step `process` function for each dataset in `inputs`.
        """
        for dataset_name in self.inputs:
            self.process(dataset_name, datasets, **kwargs)

    def process(self, dataset_name, datasets, **kwargs):
        """
        General processing of `dataset_name` in `datasets`, in place.
        """
        pass


class LocalStep(BaseStep):
    """
    Class representing a step in a processing pipeline, processing individual examples.

    The function to overwrite is `process_item`; the function accepts an item, index, and all the other datasets, if needed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, dataset_name, datasets, **kwargs):
        datasets[dataset_name] = datasets[dataset_name].map(
            lambda item, index: self.process_item(item, index, datasets, **kwargs),
            with_indices=True,
            load_from_cache_file=False,
        )

    def process_item(self, item, index, datasets, **kwargs):
        return item


class GlobalStep(BaseStep):
    """
    Class representing a step in a processing pipeline, processing the entire dataset.

    The function to overwrite is `process_all`; the function accepts the dataset and all the other datasets, if needed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, dataset_name, datasets, **kwargs):
        datasets[dataset_name] = self.process_all(
            datasets[dataset_name], datasets, **kwargs
        )

    def process_all(self, dataset, datasets, **kwargs):
        return dataset
