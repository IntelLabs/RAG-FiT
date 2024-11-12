"""Module containing filters"""


def msmarco_positive_filter(x):
    return 1 in x["passages"]["is_selected"]


filters = {"MSMARCO": msmarco_positive_filter}
