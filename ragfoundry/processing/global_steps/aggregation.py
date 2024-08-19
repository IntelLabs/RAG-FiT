from datasets import concatenate_datasets

from ..step import GlobalStep


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
