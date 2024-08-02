import random

from ..step import LocalStep


class RAFTStep(LocalStep):
    """
    Implementation of RAFT: Adapting Language Model to Domain Specific RAG.

    This class compiles a list of negative documents with probability `raft_p`,
    and a combination of positive and negative documents with probability 1 - `raft_p`.

    Zhang, Tianjun, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia,
    Ion Stoica, and Joseph E. Gonzalez. 2024. “RAFT: Adapting Language Model
    to Domain Specific RAG.” arXiv. http://arxiv.org/abs/2403.10131.
    """

    def __init__(
        self,
        k: int = 5,
        raft_p=0.5,
        neg_docs_num=5,
        positive_key="positive_passages",
        negative_key="negative_passages",
        output_key="docs",
        **kwargs,
    ):
        """
        Args:
            k (int): The number of positive passages to consider.
            raft_p (float, optional): The probability of using positive passages. Defaults to 0.5.
            neg_docs_num (int, optional): The number of negative passages to consider. Defaults to 2.
            positive_key (str, optional): The key containing the positive passages. Defaults to "positive_passages".
            negative_key (str, optional): The key containing the negative passages. Defaults to "negative_passages".
            output_key (str, optional): The key to store the output. Defaults to "docs".
        """
        super().__init__(**kwargs)
        self.k = k
        self.raft_p = raft_p
        self.neg_docs_num = neg_docs_num
        self.positive_key = positive_key
        self.negative_key = negative_key
        self.output_key = output_key

    def process_item(self, item: dict, index, datasets, **kwargs):
        docs_pos = item[self.positive_key]
        docs_neg = item.get(self.negative_key, [])

        p = random.random()  # nosec
        oracle = 0

        if p > self.raft_p:
            docs = docs_pos[: self.k] + docs_neg[: self.neg_docs_num]
        else:
            docs = docs_neg[: self.neg_docs_num]

        item[self.output_key] = docs

        return item
