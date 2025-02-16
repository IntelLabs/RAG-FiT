import re


class RegexAnswer:
    """
    Extract answers from the text using regular expressions.

    Pattern is the regular expression used to extract the answer.
    Stopping pattern is a string used to split the answer.

    Example:
    `r = RegexAnswer("<ANSWER>: (.*)", "[,.;]")`
    """

    def __init__(self, capture_pattern=None, stopping_pattern=None):
        self.capture_pattern = capture_pattern
        self.stopping_pattern = stopping_pattern

    def __call__(self, text: str):
        """
        Extract the answer from the text.
        """
        if (capture := self.capture_pattern) and capture != "":
            match = re.search(capture, text, re.MULTILINE | re.DOTALL)
            if match:
                text = match.group(1)
            else:
                return ""

        if (stopping := self.stopping_pattern) and stopping != "":
            text = re.split(stopping, text)[0]

        return text


class SimpleRegex:
    """
    Extract answers from the text using regular expressions.
    Find the last index where the pattern is found in the text.

    Example:

    r = SimpleRegex("answer is: ")
    """

    def __init__(self, pattern=None, **kwargs):
        self.pattern = pattern

    def __call__(self, text: str) -> str:
        """
        Extract the answer from the text by returning everything after the last match of pattern.
        """
        if (pattern := self.pattern) and pattern != "":
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            matches = list(matches)
            if matches:
                return text[matches[-1].end() :]
            else:
                return ""
        else:
            return text


if __name__ == "__main__":
    text = """\
        The answer is: 42.
        The answer is: 43.
        The answer is: 44."""

    r = SimpleRegex("Answer is: ")
    z = SimpleRegex("answer: ")
    x = SimpleRegex()
    y = SimpleRegex("")

    print("r", r(text))
    print("z", z(text))
    print("x", x(text))
    print("y", y(text))
