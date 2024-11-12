from datasets import concatenate_datasets

from ..step import GlobalStep
from .filters import filters


class FilterDataset(GlobalStep):
    """
    Step for filtering a dataset.
    """

    def __init__(self, filter_fn, **kwargs):
        """
        Args:
            filter_fn (function): Function to filter the dataset.
        """
        super().__init__(**kwargs)
        self.filter_fn = filters[filter_fn]

    def process(self, dataset_name, datasets, **kwargs):
        datasets[dataset_name] = datasets[dataset_name].filter(self.filter_fn)


class SelectColumns(GlobalStep):
    """
    Step for selecting specified columns in a dataset.
    """

    def __init__(self, columns: list[str], **kwargs):
        """
        Args:
            columns (list): List of keys to keep in the dataset.
        """
        super().__init__(**kwargs)
        assert isinstance(columns, list), "columns should be a list of strings."
        self.columns = columns

    def process(self, dataset_name, datasets, **kwargs):
        datasets[dataset_name] = datasets[dataset_name].select_columns(self.columns)


class MergeDatasets(GlobalStep):
    """
    Step for merging datasets.

    Merge is done using concatenation. Optional shuffling by providing a seed.
    """

    def __init__(self, output, shuffle=None, **kwargs):
        """
        Args:
            output (str): Name of the output dataset. Should be unique.
            shuffle (int, optional): seed for shuffling. Default is None.
        """
        super().__init__(**kwargs)
        self.output = output
        self.shuffle = shuffle
        self.completed = False
        self.cache_step = False

    def process(self, dataset_name, datasets, **kwargs):
        if not self.completed:
            data = concatenate_datasets([datasets[name] for name in self.inputs])
            if self.shuffle:
                data = data.shuffle(self.shuffle)
            datasets[self.output] = data
            self.completed = True


class DatasetTagger(GlobalStep):
    """
    Class to tag each example with the dataset name. Useful when running aggregations.
    """

    def __init__(self, keyword="source", **kwargs):
        """
        Args:
            keyword (str): The key to use for tagging. Default is "source".
        """
        super().__init__(**kwargs)
        self.keyword = keyword

    def tag(self, item, dataset_name):
        item[self.keyword] = dataset_name
        return item

    def process(self, dataset_name, datasets, **kwargs):
        datasets[dataset_name] = datasets[dataset_name].map(
            lambda item: self.tag(item, dataset_name),
            load_from_cache_file=False,
        )
