from ..step import GlobalStep


class OutputData(GlobalStep):
    """
    Simple class to output the dataset to a jsonl file.
    """

    def __init__(self, prefix, **kwargs):
        """
        Args:
            prefix (str): Prefix for the output.

        The output name is `{prefix}-{dataset_keyname}.jsonl`.
        """
        super().__init__(**kwargs)
        self.prefix = prefix

    def process(self, dataset_name, datasets, **kwargs):
        fname = f"{self.prefix}-{dataset_name}.jsonl"
        datasets[dataset_name].to_json(fname, lines=True)
