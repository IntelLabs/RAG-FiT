from ragfit.evaluation.metrics import normalize_text

from ..step import LocalStep


class ASQA(LocalStep):
    """
    Normalizes ASQA dataset.

    It has long answer, to be measured with ROUGE-L and multiple short answers, to be
    measured with string-EM. Long answer is saved in the `answers` field, while the
    short answers (list of lists) are saved in the `answer-short` field.
    """

    def process_item(self, item, index, datasets, **kwargs):
        item["answer-long"] = [ann["long_answer"] for ann in item["annotations"]]
        short = []
        for qa_pair in item["qa_pairs"]:
            normalize = [normalize_text(ans) for ans in qa_pair["short_answers"]]
            short.append(normalize)
        item["answer-short"] = short

        item["answers"] = item["answer-long"]
        item["query"] = item["ambiguous_question"]

        return item


class HotPot(LocalStep):
    """
    Normalizes NotPotQA dataset to look like NQ, TQA
    """

    def process_item(self, item, index, datasets, **kwargs):
        item["answers"] = [item["answer"]]
        item["query"] = item["question"]

        # Contexts converted into a list of relevant documents (Dict with title + text)
        titles = item["context"]["title"]

        sentences = item["context"]["sentences"]
        sentences = ["".join(lines) for lines in sentences]

        docs = [f"{title}: {text}" for title, text in zip(titles, sentences)]
        item["positive_passages"] = docs

        return item


class ARCC(LocalStep):
    """
    Prepare dataset for RAG augmentation.
    """

    def process_item(self, item, index, datasets, **kwargs):
        item["query"] = item["question"]
        item["options"] = item["choices"]["text"]
        item["answers"] = item["answerKey"]

        return item


class PubMed(LocalStep):
    """
    Prepare dataset for RAG augmentation.
    """

    def process_item(self, item, index, datasets, **kwargs):
        item["query"] = item["QUESTION"]
        item["answers"] = [item["final_decision"]]
        docs = item["CONTEXTS"]
        item["positive_passages"] = docs

        return item
