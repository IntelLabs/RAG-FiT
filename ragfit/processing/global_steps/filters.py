"""Module containing filters."""


def msmarco_positive_filter(x):
    """
    Identify the positive passages in MSMARCO dataset.
    """
    return 1 in x["passages"]["is_selected"]


filters = {"MSMARCO": msmarco_positive_filter}
