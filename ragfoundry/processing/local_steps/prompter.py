from ..step import LocalStep


class TextPrompter(LocalStep):
    """
    Class for creating prompts. The input is a prompt file with placeholders, a mapping of the placeholders to the item keys, and the key to store the result.

    Example: the prompt file is "prompt.txt" with the content "{query}?\n{answer}".
    The mapping is {"query": "question", "answer": "solution"}.
    The result key is "prompt".
    """

    def __init__(self, prompt_file: str, mapping: dict, output_key, **kwargs):
        super().__init__(**kwargs)
        self.prompt = open(prompt_file).read()
        self.mapping = mapping
        self.output_key = output_key

    def process_item(self, item, index, datasets, **kwargs):
        prompt = self.prompt.format(
            **{k: item.get(v, "") for k, v in self.mapping.items()}
        )
        item[self.output_key] = prompt
        return item


class FewshotPrompter(LocalStep):
    """
    Class for formatting fewshot examples into a string, to be used in a prompt.

    The prompt template contains a placeholder for the fewshot examples; this
    class is used to format the examples into a string.
    """

    def __init__(
        self, prompt_file: str, fewshot_key: str, mapping: dict, output_key: str, **kwargs
    ):
        """
        Args:
            prompt_file (str): Path to the prompt file for the individual fewshot examples.
            fewshot_key (str): Key to the fewshot examples in the item.
            mapping (dict): Mapping of the placeholders in the prompt to the item keys.
            output_key (str): Key to store the formatted fewshot examples.
        """
        super().__init__(**kwargs)
        self.prompt = open(prompt_file).read()
        self.fewshot_key = fewshot_key
        self.mapping = mapping
        self.output_key = output_key

    def process_item(self, item, index, datasets, **kwargs):
        texts = []
        for ex in item[self.fewshot_key]:
            text = self.prompt.format(
                **{k: ex.get(v, "") for k, v in self.mapping.items()}
            )
            texts.append(text)

        item[self.output_key] = "\n\n".join(texts)
        return item
