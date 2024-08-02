from ..step import LocalStep


class ContextHandler(LocalStep):
    """
    Example class for processing retrieved documents.

    In this simple example, the text is combined with the title.
    """

    def __init__(self, docs_key, title_key="title", text_key="content", **kwargs):
        """
        Args:
            docs_key (str): Key to the documents in the item.
            title_key (str): Key to the title in the document.
            text_key (str): Key to the text in the document.
        """
        super().__init__(**kwargs)
        self.docs_key = docs_key
        self.title_key = title_key
        self.text_key = text_key

    def process_item(self, item, index, datasets, **kwargs):
        docs = item[self.docs_key]
        docs = [f"{doc[self.title_key]}: {doc[self.text_key]}" for doc in docs]
        item[self.docs_key] = docs
        return item


class DocumentsJoiner(LocalStep):
    """
    Class to select top-K and join the documents into a string.
    """

    def __init__(self, docs_key, k=None, join_string="\n", **kwargs):
        """
        Args:
            docs_key (str): Key to the documents in the item.
            k (int, optional): Number of documents to select or take all. Defaults to None.
            join_string (str): String to join the documents. Defaults to "\n".
        """
        super().__init__(**kwargs)
        self.docs_key = docs_key
        self.k = k
        self.join_string = join_string

    def process_item(self, item, index, datasets, **kwargs):
        docs = item[self.docs_key]
        if self.k is not None:
            docs = docs[: self.k]
        docs = self.join_string.join(docs)
        item[self.docs_key] = docs
        return item
