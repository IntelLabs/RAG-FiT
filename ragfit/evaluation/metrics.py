import re
import string
from collections import Counter, defaultdict

from .base import MetricBase


class HFEvaluate(MetricBase):
    """
    Wrapper class around `evaluate` metrics; easy to use, only need metric names.
    """

    def __init__(self, key_names, metric_names: list[str], **kwargs):
        """
        Args:
            key_names (dict): A dictionary containing the field names.
            metric_names (list[str]): A list of metric names.
        """
        import evaluate

        super().__init__(key_names, **kwargs)
        self.metric_names = metric_names
        self.metric = evaluate.combine(metric_names)
        self.local = True

    def measure(self, example):
        """
        Measure the performance of the model on a given example.

        Args:
            example (dict): The example containing input and target values.

        Returns:
            dict: The performance metric(s) computed for the example.
        """
        input = example[self.field]
        target = example[self.target]

        if isinstance(target, list):
            results = defaultdict(int)
            for tar in target:
                results = {
                    k: max(v, results[k])
                    for k, v in self.metric.compute(
                        predictions=[input], references=[tar]
                    ).items()
                }
            return results
        else:
            return self.metric.compute(predictions=[input], references=[target])


class Classification(MetricBase):
    """
    Metrics for classification answers: accuracy, precision, recall, F1; macro-averaged.

    mapping: dict - mapping of labels to integers.
        Example: {"true": 1, "false": 0, "maybe": 2}
    else_value: int - value to assign to labels not in the mapping.
    """

    def __init__(
        self, key_names: dict, mapping: dict, else_value: int = 2, **kwargs
    ) -> None:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        super().__init__(key_names, **kwargs)
        self.local = False
        self.mapping = mapping
        self.else_value = else_value
        self.precision_recall_fn = precision_recall_fscore_support
        self.accuracy_fn = accuracy_score

    def measure(self, example: dict):
        inputs = example[self.field]
        targets = example[self.target]

        if isinstance(targets[0], list):
            targets = [t[0] for t in targets]

        inputs = [
            self.mapping.get(normalize_text(i).strip(), self.else_value) for i in inputs
        ]

        targets = [
            self.mapping.get(normalize_text(t).strip(), self.else_value) for t in targets
        ]

        precision, recall, f1, _ = self.precision_recall_fn(
            targets, inputs, average="macro"
        )
        accuracy = self.accuracy_fn(targets, inputs)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }


def normalize_text(s):
    """
    Normalize the given text by lowercasing it, removing punctuation, articles, and extra whitespace.

    Args:
        s (str): The text to be normalized.

    Returns:
        str: The normalized text.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class F1(MetricBase):
    """
    Implementing F1 based on code from Kilt.
    """

    def __init__(self, key_names, **kwargs) -> None:
        """Initialize the Metrics class.

        Args:
            key_names (dict): A dictionary containing the field names.
        """
        super().__init__(key_names, **kwargs)
        self.local = True

    @staticmethod
    def _f1(prediction, ground_truth):
        prediction_tokens = normalize_text(prediction).split()
        ground_truth_tokens = normalize_text(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def measure(self, example: dict):
        input = example[self.field]
        target = example[self.target]

        assert isinstance(input, str), f"Generated text should be a string: {input}"
        if not isinstance(target, list):
            target = [target]

        scores = [self._f1(input, t) for t in target]
        return {"F1": max(scores)}


class EM(MetricBase):
    """
    Implementing Exact Match based on code from Kilt.
    """

    def __init__(self, key_names, **kwargs) -> None:
        """Initialize the Metrics class.

        Args:
            key_names (dict): A dictionary containing the field names.
        """
        super().__init__(key_names, **kwargs)
        self.local = True

    def measure(self, example: dict):
        input = example[self.field]
        target = example[self.target]

        assert isinstance(input, str), f"Generated text should be a string: {input}"
        if not isinstance(target, list):
            target = [target]

        scores = [normalize_text(input) == normalize_text(t) for t in target]
        return {"EM": int(max(scores))}


class StringEM(MetricBase):
    """
    Implementing String Exact Match.

    Used in ASQA to evaluate whether the annoated short answers appear in the
    generated answer as sub-strings.
    """

    def __init__(self, key_names: dict, **kwargs) -> None:
        """
        Initialize the Metrics class.

        Args:
            key_names (dict): A dictionary containing the field names.
        """
        super().__init__(key_names, **kwargs)
        self.local = True

    def measure(self, example: dict):
        input = example[self.field]
        target = example[self.target]

        assert isinstance(input, str), f"Generated text should be a string: {input}"
        assert isinstance(target[0], list), f"Target should be a list of lists: {target}"

        input = normalize_text(input)
        scores = [any(cand in input for cand in item) for item in target]

        return {"StringEM": sum(scores) / len(scores)}


class BERTScore(MetricBase):
    """
    BERTScore metric, based on the BERTScore library.
    """

    def __init__(self, key_names: dict, model="microsoft/deberta-large-mnli", **kwargs):
        """Initialize the Metrics class.

        Args:
            key_names (dict): A dictionary containing the field names.
            model (str, optional): The name of the BERT model to use. Defaults to "microsoft/deberta-large-mnli".
        """
        super().__init__(key_names, **kwargs)
        from bert_score import BERTScorer

        self.scorer = BERTScorer(model, lang="en", rescale_with_baseline=True)
        self.local = True

    def measure(self, example):
        input = example[self.field]
        target = example[self.target]

        if not isinstance(target, list):
            target = [target]

        scores = [self.scorer.score([input], [t])[2].item() for t in target]

        return {"BERTScore-F1": max(scores)}


class Semantic(MetricBase):
    """
    Semantic similarity between label and answer using a cross-encoder.
    """

    def __init__(
        self,
        key_names: dict,
        model: str = "vectara/hallucination_evaluation_model",
        **kwargs,
    ) -> None:
        """
        Initializes an instance of the class.

        Args:
            key_names (dict): A dictionary containing the field names.
            model (str, optional): The name of the BERT model to use.
        """
        super().__init__(key_names, **kwargs)

        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)
        self.local = True

    def measure(self, example):
        input = example[self.field]
        target = example[self.target]
        if not isinstance(target, list):
            target = [target]

        scores = self.model.predict([[input, t] for t in target])

        return {"Semantic": max(scores)}
