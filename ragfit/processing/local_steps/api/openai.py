from ragfit.models.openai_executor import OpenAIExecutor

from ...step import LocalStep


class OpenAIChat(LocalStep):
    """
    Interaction with OpenAI service.

    Model is represented by the `OpenAIExecutor`.

    This step is a wrapper, extracting the prompt from the item, interact with the API, and
    saves the response to the `answer` key in the item.
    """

    def __init__(self, model, instruction, prompt_key, answer_key, **kwargs):
        """
        Args:
            model (dict): Configuration for the OpenAIExecutor.
            instruction (str): Path to the system instruction file.
            prompt_key (str): Key to the prompt in the item.
            answer_key (str): Key to store the response.
        """
        super().__init__(**kwargs)

        self.model = OpenAIExecutor(**model)
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.instruction = open(instruction).read()

    def process_item(self, item, index, datasets, **kwargs):
        answer = self.model.chat(item[self.prompt_key], self.instruction)
        item[self.answer_key] = answer
        return item
