import logging
import math

from deepeval.test_case import LLMTestCase
from langchain_openai import AzureChatOpenAI

from .base import MetricBase


class DeepEvalBase(MetricBase):
    """
    Base class for DeepEval metrics.

    Here we use AzureChatOpenAI interface; replace if needed.
    """

    def __init__(
        self,
        key_names: dict,
        api_version,
        azure_endpoint,
        azure_deployment,
        **kwargs,
    ):
        super().__init__(key_names, **kwargs)
        self.local = True
        self.query = self.key_names["query"]
        self.context = self.key_names["context"]

        self.model = AzureChatOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            request_timeout=600,
            max_retries=10,
        )


class Faithfulness(DeepEvalBase):
    """
    Faithfulness metric from DeepEval, based on RAGAS.

    Measures faithfulness of generated text by comparing it to the target text.
    """

    def __init__(self, key_names: dict, threshold=0.3, **kwargs):
        super().__init__(key_names, **kwargs)
        from deepeval.metrics.ragas import RAGASFaithfulnessMetric

        self.metric = RAGASFaithfulnessMetric(threshold=threshold, model=self.model)

    def measure(self, example):
        query = example[self.query]
        output = example[self.field]
        context = example[self.context]

        test_case = LLMTestCase(
            input=query,
            actual_output=output or "No answer.",
            retrieval_context=[context] if isinstance(context, str) else context,
        )
        try:
            self.metric.measure(test_case)
            score = self.metric.score
        except Exception as e:
            logging.error(f"OpenAI exception: {e}")
            score = 0

        return {"faithfulness": score if not math.isnan(score) else 0}


class Relevancy(DeepEvalBase):
    """
    Answer relevancy metric from DeepEval, based on RAGAS.

    Measures relevancy of generated text by comparing it to the retrieved documents.
    """

    def __init__(self, key_names: dict, embeddings, threshold=0.3, **kwargs):
        super().__init__(key_names, **kwargs)
        from deepeval.metrics.ragas import RAGASAnswerRelevancyMetric
        from ragas.embeddings import HuggingfaceEmbeddings

        self.metric = RAGASAnswerRelevancyMetric(
            threshold=threshold,
            embeddings=HuggingfaceEmbeddings(model_name=embeddings),
            model=self.model,
        )

    def measure(self, example):
        query = example[self.query]
        output = example[self.field]
        context = example[self.context]

        test_case = LLMTestCase(
            input=query,
            actual_output=output or "No answer.",
            retrieval_context=[context] if isinstance(context, str) else context,
        )
        try:
            self.metric.measure(test_case)
            score = self.metric.score
        except Exception as e:
            logging.error(f"OpenAI exception: {e}")
            score = 0

        return {"relevancy": score}


class Hallucination(DeepEvalBase):
    """
    Hallucination metric from DeepEval.

    Measures hallucination of generated text by comparing it to the retrieved documents.
    """

    def __init__(self, key_names: dict, threshold=0.5, **kwargs):
        super().__init__(key_names, **kwargs)
        from deepeval.metrics import HallucinationMetric

        self.metric = HallucinationMetric(
            threshold=threshold, include_reason=False, model=self.model
        )

    def measure(self, example):
        output = example[self.field]
        context = example[self.context]

        test_case = LLMTestCase(
            input="",
            actual_output=output,
            context=[context] if isinstance(context, str) else context,
        )

        try:
            self.metric.measure(test_case)
            score = self.metric.score
        except Exception as e:
            logging.error(f"OpenAI exception: {e}")
            score = 0

        return {"hallucination": score}
