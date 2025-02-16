import glob
import os
import re
from pathlib import Path

import pandas as pd
import yaml


def read_yaml_file(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def get_score(dic: dict) -> dict:
    return dic["local"] if dic["local"] else {"acc": 10}


def extract(x):
    data: dict = get_score(read_yaml_file(x))
    x = os.path.basename(x)
    match = re.match(
        r"(?P<method>.*)-(?P<data>(asqa|hotpotqa|triviaqa))-(?P<misc>.*?)-?test-(?P<model>.*?)-?generated-(?P<extra>[^-]*)-?results.yaml",
        x,
    )
    return {
        "original": x,
        "data": match.group("data"),
        "Method": match.group("method")
        + ("-" + extra if (extra := match.group("extra")) else "")
        + ("-" + misc if (misc := match.group("misc")) else ""),
        "Model": model if (model := match.group("model")) else "",
        **data,
    }


prefix = (
    "baseline",
    "cot",
)


data = [
    extract(f)
    for f in glob.glob("../data/results/*.yaml")
    if any(Path(f).stem.startswith(m) for m in prefix)
]


ds = pd.DataFrame(data)


ds.to_csv("all-runs.csv", index=None)
