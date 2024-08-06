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
