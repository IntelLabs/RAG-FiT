import re
import string


def check_package_installed(package_name: str, optional_msg: str = ""):
    """
    Check if a package is installed.
    """

    import importlib.util

    if importlib.util.find_spec(package_name) is None:
        raise ImportError(f"{package_name} package is not installed; {optional_msg}")


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
