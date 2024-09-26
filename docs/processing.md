# Data Augmentation

To demonstrate the usage of RAG Foundry data augmentation, we will follow the experimentation presented in the paper.
Choosing the ASQA Q&A dataset and the Phi-3 model. We compare a baseline configuration with 4 other configurations:

1. Retrieval augmentation using a corpus and inserting the documents in the prompt after the question.
2. Similar to (1) but having the model fine-tune on the completions.
3. Similar to (1) and adding a Chain-of-Thought instruction for the model to explain its reasoning and format its
answer.
4. Similar to (3) but having the model fine-tune on the completions while implementing a technique from RAFT where
distracting documents are used.

The [ASQA dataset](https://huggingface.co/datasets/din0s/asqa) has two types of answer: a long answer and lists of short
answers (actually list of lists). Additionally, it has some minimal amount of context in the data, so we augment it
using a corpus, stored as a vector DB; we use [Qdrant](https://qdrant.tech/).

In order to train configuration (4), we need to have CoT well-reasoned responses as labels, so we use OpenAI GPT4 model to augment a
dataset with these synthetic labels.

**Notice**: all the configurations mentioned here, implementing the experiments done in the paper, are saved in
`configs/paper/`. They don't run by default, they need to be specified by running:

```sh
python module-name.py -cp configs/paper -cn config-name-without-extension
```

## Retrieval

The first step would be to augment the entire dataset (train, dev) with relevant documents, based on the questions, see
[processing-asqa-retrieval.yaml](../configs/paper/processing-asqa-retrieval.yaml). Let's focus on the different steps:

```yaml
- _target_: ragfit.processing.dataset_loaders.loaders.HFLoader
  inputs: train
  dataset_config:
        path: din0s/asqa
        split: train

- _target_: ragfit.processing.dataset_loaders.loaders.HFLoader
  inputs: dev
  dataset_config:
        path: din0s/asqa
        split: dev
```

We load the train and dev splits, to be used in the pipeline; they will be referred using the `inputs` keyword used in this
step.

```yaml
- _target_: ragfit.processing.local_steps.common_datasets.ASQA
  inputs: [train, dev]
```
We do some minimal processing, related to ASQA, namely column renaming, collecting the short and long answers and
having a consistent scheme, for example: `query`, `answers`, `positive_passages`, etc. Feel free to add your own types
of pre-processing.

Notice the `inputs` keyword can accept a list of strings, meaning the step will run over the datasets specified.

```yaml
- _target_:
        ragfit.processing.local_steps.retrievers.haystack.HaystackRetriever
  inputs: [train, dev]
  pipeline_or_yaml_path: ./configs/external/haystack/qdrant.yaml
  docs_key: positive_passages
  query_key: query
```
This is the retrieval step. We use the [Haystack](https://haystack.deepset.ai/) framework for building RAG pipelines; in
this example, the Haystack pipeline is comprised of an embedder and a retriever, connecting the Qdrant using a
Qdrant-Haystack integration (all defined in the requirements file). The Haystack pipeline is initialized from the
[Qdrant.yaml](../configs/external/haystack/qdrant.yaml) configuration. One can use other frameworks for retrieval, like
LangChain, LlamaIndex, or others.

The retrieval step will store the most relevant documents (k=5) in the `docs_key` and the query will be defined by the
`query_key`.

```yaml
- _target_: ragfit.processing.local_steps.context.ContextHandler
  inputs: [train, dev]
  docs_key: positive_passages
```
In this simple step, the documents retrieved are processed; they have a title and content fields and this step combine
these into a single string for every document. This step may be unnecessary, depending on the retrieval mechanism and
format.

```yaml
- _target_: ragfit.processing.global_steps.sampling.Sampler
  inputs: [train, dev]
  k: 1
  input_key: positive_passages
  output_key: negative_passages
```
The `Sampler` class deals with sampling examples from the same dataset or others. In order to train the RAFT-based
model on a combination of relevant and distracting documents, we need to collect these distracting documents. Here we
chose to collect positive documents from other examples, to be used as negative documents. The `Sampler` is then ran
with k=1, it collects only the `positive_passages` from the examples it samples and store them in a new keyword, called
`negative_passages`.

```yaml
- _target_: ragfoundry.processing.global_steps.output.OutputData
  inputs: [train, dev]
  prefix: asqa
```
Finally we write the two resulting dataset to disk. They represent the retrieval-augmented datasets, ready to be
processed for the different tasks.

To run this process:
```sh
python processing.py -cp configs/paper -cn processing-asqa-retrieval
```

## Baseline Configuration

For the baseline, there is not going to be context, only the question presented to the model. We use
instruction-following models that have a chat template built-in. The framework populates the chat template using the
inputs and outputs we generate, so we don't need to worry about roles and special tokens. Additionally, the system
instruction is specified only during training and inference: it needn't be part of the dataset so these next steps mainly
deal with the prompt generation.

These are the interesting steps:

```yaml
- _target_: ragfit.processing.dataset_loaders.loaders.LocalLoader
  inputs: dev
  filename: asqa-dev.jsonl

- _target_: ragfit.processing.local_steps.prompter.TextPrompter
  inputs: dev
  prompt_file: ragfit/processing/prompts/qa-short.txt
  output_key: prompt
  mapping:
        query: query
```

We load the locally retrieval-augmented files we generated in the previous section.

The `TextPrompter` populates a template file containing placeholders in python format, see the [short
template](../ragfit/processing/prompts/qa-short.txt). The step replace the placeholders with variables using a provided
mapping. The result is a string, saved in a keyword called `outputs_key`.

To run this process:
```sh
python processing.py -cp configs/paper -cn processing-asqa-baseline
```

## Context

Preparing for configurations (1) and (2), we want to augment the examples with the top 5 documents we collected in the
first step.

```yaml
- _target_: ragfit.processing.local_steps.context.DocumentsJoiner
  inputs: [train, dev]
  docs_key: positive_passages
  k: 5

- _target_: ragfit.processing.local_steps.prompter.TextPrompter
  inputs: [train, dev]
  prompt_file: ragfit/processing/prompts/qa.txt
  output_key: prompt
  mapping:
        question: query
        context: positive_passages
```
The `DocumentJoiner` joins a list of strings and is needed before the `TextPrompter` we've seen from the previous
section. We prepare a dev file—for testing the model with retrieved documents—and also a training file, in order
to run fine-tuning. Both configurations will be evaluated on the dev dataset.

To run this process:
```sh
python processing.py -cp configs/paper -cn processing-asqa-context
```

## Chain-of-Thought

We prepare a dev set with CoT reasoning prompt. The configuration will be similar to the *Context* configuration,
however here we use a different prompt template:

```yaml
- _target_: ragfit.processing.local_steps.prompter.TextPrompter
  inputs: dev
  prompt_file: ragfit/processing/prompts/cot.txt
  output_key: prompt
  mapping:
        question: query
        context: positive_passages
```

To run this process:
```sh
python processing.py -cp configs/paper -cn processing-asqa-cot-dev
```

## Chain-of-Thought Training Dataset

In order to train a model on a CoT-based prompt, we need to collect well-reasoned responses; we use GPT4 for that.
Additionally, we implement a technique from RAFT where some percentage of the examples have purely distractor documents,
in order for the model ability to filter noise. Here are the relevant steps:

```yaml
- _target_: ragfit.processing.local_steps.raft.RAFTStep
  inputs: train
  k: 5
  raft_p: 0.5
  neg_docs_num: 2
  output_key: raft_docs
```
The `RAFTStep` implements the logic presented in the paper; the percentage of purely-distractor documents is defined by
`raft_p`. The list of documents, some relevant, some distracting, are saved in a keyword called `output_key`.

```yaml
- _target_: ragfit.processing.local_steps.context.DocumentsJoiner
  inputs: train
  docs_key: raft_docs
  k:

- _target_: ragfit.processing.local_steps.prompter.TextPrompter
  inputs: train
  prompt_file: ragfit/processing/prompts/cot.txt
  output_key: prompt
  mapping:
        question: query
        context: raft_docs
```
The documents are joined into strings; when `k:` all documents are used. The prompt used is the same as when building the dev dataset.

Next is interacting with OpeanAI; we implemented an [OpenAI class](../ragfit/models/openai_executor.py) using Azure,
one can implement using other abstractions. The step itself needs the `prompt_key`, instruction file and the results are
saved in the `answer_key`.
```yaml
- _target_: ragfit.processing.local_steps.api.openai.OpenAIChat
  inputs: train
  prompt_key: prompt
  answer_key: generated_answer
  instruction: ragfit/processing/prompts/prompt_instructions/qa.txt
  model:
        azure_endpoint: azure.endpoint.com
        api_version: 2024-05-01-preview
        model: GPT-4-32k-Bot
```

To run this process:
```sh
python processing.py -cp configs/paper -cn processing-asqa-cot-train
```
