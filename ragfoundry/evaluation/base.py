class MetricBase:
    """
    Base class for metrics.

    Metrics can be local or global; local means score are calculated per example.
    Global means score is calculated by looking at the entire dataset, e.g. fluency.
    """

    def __init__(self, key_names, **kwargs):
        self.key_names = key_names
        self.kwargs = kwargs
        self.field = self.key_names["generated"]
        self.target = self.key_names["label"]

    def measure(self, example: dict) -> dict[str, float]:
        """
        Measure the performance of the model on a given example.

        Parameters:
            example (dict): The example to evaluate the model on.

        Returns:
            dict[str, float]: A dictionary containing the performance metrics.

        """
        pass
