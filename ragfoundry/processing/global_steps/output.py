import os

from ..step import GlobalStep


class OutputData(GlobalStep):
    """
    Simple class to output the dataset to a jsonl file.
    """

    def __init__(self, prefix, filename=None, directory=None, **kwargs):
        """
        Args:
            prefix (str): Prefix for the output.
            filename (str, optional): Name of the output file. If not provided, the output file name will be generated based on the prefix and dataset name.
            directory (str, optional): Directory to save the output file. If not provided, the output file will be saved in the current directory.

        The output name is `{prefix}-{dataset_keyname/filename}.jsonl` if `filename` is not provided.
        """
        super().__init__(**kwargs)
        self.prefix = prefix
        self.filename = filename
        self.dir = directory

    def process(self, dataset_name, datasets, **kwargs):
        if self.filename:
            name = self.filename
        else:
            name = dataset_name
        fname = f"{self.prefix}-{name}.jsonl"
        if self.dir is not None:
            fname = os.path.join(self.dir, fname) if self.dir else fname
        datasets[dataset_name].to_json(fname, lines=True)


class HFHubOutput(GlobalStep):
    """
    Simple class to output the dataset to Hugging Face Hub.
    """

    def __init__(self, hfhub_tag, private=True, **kwargs):
        """
        Args:
            hfhub_tag (str): Tag for the Hugging Face Hub.
            private (bool): Whether the dataset should be private or not. Default is True.
        """
        super().__init__(**kwargs)
        self.hfhub_tag = hfhub_tag
        self.private = private

    def process(self, dataset_name, datasets, **kwargs):
        datasets[dataset_name].push_to_hub(self.hfhub_tag, private=self.private)
