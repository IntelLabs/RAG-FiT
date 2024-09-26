<div align="center">
    <img src="assets/rag_fit.png" width="400"/>
</div>

----------

[RAG Foundry: A Framework for Enhancing LLMs for Retrieval Augmented Generation](https://arxiv.org/abs/2408.02545)

**RAG Foundry** is a library designed to improve LLMs ability to use external information by fine-tuning models on
specially created RAG-augmented datasets. The library helps create the data for training, given a RAG technique, helps
easily train models using parameter-efficient finetuning (PEFT), and finally can help users measure the improved
performance using various, RAG-specific metrics. The library is modular, workflows are customizable using configuration
files.

Comments, suggestions, issues and pull-requests are welcomed! ❤️

### Installation
Clone and run:

```sh
pip install -e .
```

Optional packages can be installed:
```sh
pip install -e .[haystack]
pip install -e .[deepeval]
```

### Quick Start

For a simple, end-to-end example, see the [PubmedQA Tutorial](pubmed.md).

## Overview

The RAG Foundry framework facilitates fast prototyping and experimentation with various RAG settings and configurations,
including data selection and filtering, processing, retrieval, ranking, query manipulation, prompt generation, training,
inference, output processing and evaluation. The library is comprised of 4 modules: dataset creation, training,
inference and evaluation.

* **Dataset Creation**: The processing module creates datasets, persisting RAG interactions, to be used for RAG training
and inference. RAG interactions include dataset loading, columns normalization, data aggregation (fewshot creation),
information retrieval using external tools and frameworks, API integration, template-based prompt creation and any other
form of pre-processing. The data is saved in a consistent, model-independent, input-output format, along with all other
fields and metadata. See [Processing](processing.md).

* **Training**: using PEFT for efficient training and TRL (e.g. supervised FT) users can train any model on the augmented
datasets. Training is done on the completions. Models can be pushed to HF Hub. See [Training](training.md).

* **Inference**: generating predictions using the augmented datasets with trained or untrained LLMs. See [Inference](inference.md).

* **Evaluation**: running evaluation on the generated output from the inference module. Users can provide a list of
metrics to run; custom metrics can be implemented easily. Current metrics include EM, F1, ROUGE, BERTScore, Deepeval,
RAGAS, HF `evaluate` and classification. Metrics can be *local*—run on each example, or *global*—run on the entire
dataset, e.g. recall. Metrics can utilize any feature in the dataset, like retrieval results, reasoning,
citations and attributions, not just the input and output texts. See [Evaluation](evaluation.md).


## Running
The 4 modules are represented as scripts: `processing.py`, `training.py`, `inference.py` and `evaluation.py` at the top
level. Every call has the form `python SCRIPT options...`.

The library utilizes the [Hydra](https://hydra.cc/docs/intro/) configuration tool; it enables the use of hierarchical
configurations, easily overridden of values in the CLI and the ability to run multiple jobs remotely (e.g. integrations with
SLURM and Ray). It represents a *configuration-as-code* approach, as it can instantiate python classes according to
configuration (the `_target_` keyword indicates the python class to use in a given context).

There are default configurations for each module in the [configs](./configs/) folder. A configuration file can be
overridden like so:

```sh
python processing -cp configs/paper -cn processing-asqa-retrieval
```

Individual keywords can be overridden as well:
```sh
python processing -cp configs/paper -cn processing-asqa-retrieval   \
       output_path=/store/data/here                                 \
       cache=true
```

For a complete set of configurations, **reproducing the experimentation in the paper with the ASQA dataset**, see the
configurations in the [Paper](./configs/paper) folder.

## Citation

Please cite our paper if it helps your research:

```BibTex
@article{fleischerRAGFoundryFramework2024,
  title =        {{RAG} {Foundry}: {A} {Framework} for {Enhancing} {LLMs} for {Retrieval} {Augmented} {Generation}},
  author =       {Fleischer, Daniel and Berchansky, Moshe and Wasserblat, Moshe and Izsak, Peter},
  year =         2024,
  note =         {arXiv:2408.02545 [cs]},
  annote =       {Comment: 10 pages},
  url =          {http://arxiv.org/abs/2408.02545},
  publisher =    {arXiv},
}
```

## License

The code is licensed under the [Apache 2.0 License](LICENSE).

## Disclaimer

This is not an official Intel product.