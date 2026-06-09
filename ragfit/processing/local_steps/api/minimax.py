from ragfit.models.minimax_executor import MiniMaxExecutor

from ...step import LocalStep


class MiniMaxChat(LocalStep):
    """
    Interaction with MiniMax service.

    Model is represented by the ``MiniMaxExecutor``, which uses MiniMax's
    OpenAI-compatible API (https://api.minimax.io/v1).

    This step is a wrapper, extracting the prompt from the item, interacting
    with the API, and saving the response to the ``answer`` key in the item.
    """

    def __init__(self, model, instruction, prompt_key, answer_key, **kwargs):
        """
        Args:
            model (dict): Configuration for the MiniMaxExecutor.
            instruction (str): Path to the system instruction file.
            prompt_key (str): Key to the prompt in the item.
            answer_key (str): Key to store the response.
        """
        super().__init__(**kwargs)

        self.model = MiniMaxExecutor(**model)
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.instruction = open(instruction).read()

    def process_item(self, item, index, datasets, **kwargs):
        answer = self.model.chat(item[self.prompt_key], self.instruction)
        item[self.answer_key] = answer
        return item
