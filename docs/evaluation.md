# Evaluations

The evaluation module takes the produced inference file and the original processed dataset and runs a list of
evaluations, producing a final results file, in a YAML format. The evaluations are represented as metric classes.

We implement several metrics including: a wrapper for HuggingFace `evaluate` class, which can accept a list of metrics,
EM, F1, classification (accuracy, precision, recall, F1), BERTScore, Semantic similarity (using a customizable
cross-encoder). The module can also run metrics from [DeepEval](https://docs.confident-ai.com/docs/getting-started),
which offers a large collection of LLM evaluations.

Metrics can be either local or global; a local metric runs over each example individually, scores are collected and
averaged. A global metric runs on the entire dataset at once, for example: classification F1.


The configuration contains the following section:

```yaml
answer_processor:
  _target_: ragfit.processing.answer_processors.regex.RegexAnswer
  capture_pattern:          # "<ANSWER>: (.*)"
  stopping_pattern:         # "[,.;]"
```
The evaluation module introduces the concept of an Answer Processor. This class can run post-processing on the
generated text, preparing it for evaluations or the specific format some metrics require.

There is a default processor, called `RegexAnswer`; it can filter text, based on a python regex capture pattern. It can
also split text using a stopping pattern. For example, in the Chain-of-Thought reasoning we used in the paper, the model
is instruction to explain its answer, cite if needed and finally print the final results in the following format
`<ANSWER>: ...`. We can use this format as a capture pattern; thus models that learn to answer using this pattern (obey the
instruction) will score higher.

Next is a list of metrics; each one is a python class:
```yaml
metrics:
  - _target_: ragfit.evaluation.metrics.HFEvaluate
    metric_names: [rouge]
  - _target_: ragfit.evaluation.metrics.EM
  - _target_: ragfit.evaluation.metrics.F1
  - _target_: ragfit.evaluation.metrics.BERTScore
    model: microsoft/deberta-large-mnli
```

Some metrics require additional parameters, for example HuggingFace `evaluate` requires the metrics' names, BERTScore
requires an embedding model.

```yaml
key_names:
  generated: generated
  label: answer
  query: query
  context: context
```
A mapping of keys and values: the values should represent the names of the corresponding fields in the processed
dataset.

Finally:
```yaml
results_file: my-evaluation.yaml
generated_file: inference.jsonl
data_file: my-processed-data.jsonl
limit:
```

One needs to provide the generated inference file, the processed dataset and a filename for the results summary. A limit
number of rows can be provided for debugging purposes.

## Running Evaluations on ASQA

As the final part of the demonstration of the framework with the ASQA dataset and Phi-3 models, we will evaluate the
different RAG configurations, with and without the use of fine-tuning.

As a reminder, ASQA has 2 types of answers: long answer and short answers. We will evaluate the generated answers using
the long answer with RAGAS metrics (faithfulness and relevancy) and use the short answers with ASQA defined STR-EM.

### Short
Starting with the short answers, the label keyword is `answer-short` (recall the processing) and a representative
configuration looks like this:

```yaml
answer_processor:
    _target_: ragfit.processing.answer_processors.regex.RegexAnswer
    capture_pattern: "<ANSWER>: (.*)"
    stopping_pattern:

metrics:
    - _target_: ragfit.evaluation.metrics.StringEM

key_names:
    generated: text
    label: answer-short
    query: query

results_file: evaluation-asqa-baseline.yaml
generated_file: asqa-baseline-dev-generated.jsonl
data_file: asqa-baseline-dev.jsonl
```

Here are the calls to evaluate the different configurations:

**Baseline**:
```sh
python evaluation.py -cp configs/paper -cn evaluation-asqa-short
```

**Context**:
```sh
python evaluation.py -cp configs/paper -cn evaluation-asqa-short    \
       results_file=asqa-context-dev-generated-results.yaml         \
       data_file=asqa-context-dev.jsonl                             \
       generated_file=asqa-context-dev-generated.jsonl
```

**Context with fine-tuned model**:
```sh
python evaluation.py -cp configs/paper -cn evaluation-asqa-short    \
       results_file=asqa-context-ft-dev-generated-results.yaml      \
       data_file=asqa-context-dev.jsonl                             \
       generated_file=asqa-context-ft-dev-generated.jsonl
```

**Chain-of-Thought**:
```sh
python evaluation.py -cp configs/paper -cn evaluation-asqa-short    \
       results_file=asqa-cot-dev-generated-results.yaml             \
       data_file=asqa-cot-dev.jsonl                                 \
       generated_file=asqa-cot-dev-generated.jsonl
```

**Chain-of-Thought with fine-tuned model**:
```sh
python evaluation.py -cp configs/paper -cn evaluation-asqa-short    \
       results_file=asqa-cot-ft-dev-generated-results.yaml          \
       data_file=asqa-cot-dev.jsonl                                 \
       generated_file=asqa-cot-ft-dev-generated.jsonl
```


### Long
Evaluation the generated output with respect to the full answer, we use two RAGAS metrics, namely faithfulness and
relevancy. The RAGAS metrics require a context for the critic to make a judgment, so these are not relevant for the
baseline configuration.

The different in configuration is in the list of metrics and keywords:

```yaml
metrics:
    - _target_: ragfit.evaluation.deep.Faithfulness
      azure_endpoint: azure.endpoint.com
      azure_deployment: GPT-4-32k-Bot
      api_version: 2024-05-01-preview
    - _target_: ragfit.evaluation.deep.Relevancy
      azure_endpoint: azure.endpoint.com
      azure_deployment: GPT-4-32k-Bot
      api_version: 2024-05-01-preview
      embeddings: BAAI/bge-small-en-v1.5

key_names:
    generated: text
    label: answers
    query: query
    context: positive_passages
```

The relevancy metrics an embedder—it generates probable questions based on the generated answer (and the context) and
then measures semantic similarity to the original question.

**Context**:
```sh
python evaluation.py -cp configs/paper -cn evaluation-asqa-long     \
       results_file=asqa-context-dev-generated-results-ragas.yaml   \
       data_file=asqa-context-dev.jsonl                             \
       generated_file=asqa-context-dev-generated.jsonl
```

**Context with fine-tuned model**:
```sh
python evaluation.py -cp configs/paper -cn evaluation-asqa-long         \
       results_file=asqa-context-ft-dev-generated-results-ragas.yaml    \
       data_file=asqa-context-dev.jsonl                                 \
       generated_file=asqa-context-ft-dev-generated.jsonl
```

**Chain-of-Thought**:
```sh
python evaluation.py -cp configs/paper -cn evaluation-asqa-long \
       results_file=asqa-cot-dev-generated-results-ragas.yaml   \
       data_file=asqa-cot-dev.jsonl                             \
       generated_file=asqa-cot-dev-generated.jsonl
```

**Chain-of-Thought with fine-tuned model**:
```sh
python evaluation.py -cp configs/paper -cn evaluation-asqa-long     \
       results_file=asqa-cot-ft-dev-generated-results-ragas.yaml    \
       data_file=asqa-cot-dev.jsonl                                 \
       generated_file=asqa-cot-ft-dev-generated.jsonl
```




