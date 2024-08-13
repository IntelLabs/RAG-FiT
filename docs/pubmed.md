# Fine-tuning Phi-3 for PubmedQA

We will demonstrate the RAG Foundry framework by creating a RAG augmented dataset, fine-tuning a model and running an evaluation on the [PubmedQA](https://huggingface.co/datasets/bigbio/pubmed_qa) dataset. We will follow the experimentation in the paper, implementing the **RAG-sft** configuration, which comprised of creating prompts with relevant context and fine-tuning a model on the completions.

The [PubmedQA](https://huggingface.co/datasets/bigbio/pubmed_qa) dataset contains relevant context for each question, so there's no need for retrieval—for an example with a retrieval step, see the ASQA processing [tutorial](./processing.md).

**Notice**: all the configurations mentioned in this guide, implementing the experiments done in the paper, are saved in
`configs/paper/`. They don't run by default, they need to be specified by running:

```sh
python module-name.py -cp configs/paper -cn config-name-without-extension
```


## RAG Dataset Creation

We use the 1st module, called `processing.py` to generate the RAG-augmented dataset. To run it:

```sh
python processing.py -cp configs/paper -cn processing-pubmed-rag
```

Let's analyze the [configuration file](../configs/paper/processing-pubmed-context.yaml) used for the dataset creation:

```yaml
name: pubmed_rag
cache: true
output_path: .
```

Start by defining a pipeline name, turning caching on, and specifying the current folder for the output files.

```yaml
steps:
    - _target_: ragfoundry.processing.dataset_loaders.loaders.HFLoader
      inputs: train
      dataset_config:
            path: bigbio/pubmed_qa
            split: train

    - _target_: ragfoundry.processing.dataset_loaders.loaders.HFLoader
      inputs: test
      dataset_config:
            path: bigbio/pubmed_qa
            name: pubmed_qa_labeled_fold0_source
            split: test
```

Next we load a training and test sets from the Hugging Face hub. The `inputs` keyword is used to denote the datasets to be used on the subsequent steps.

```yaml
    - _target_: ragfoundry.processing.global_steps.sampling.ShuffleSelect
      inputs: train
      limit: 50000

    - _target_: ragfoundry.processing.local_steps.common_datasets.PubMed
      inputs: [train, test]

    - _target_: ragfoundry.processing.local_steps.context.DocumentsJoiner
      inputs: [train, test]
      docs_key: positive_passages
      k: 5
```

Next are 3 technical steps: we limit the size of the training dataset to 50k examples (optional). We do minimal processing of features: namely creating a `query`, `answers` and `positive_passages` features. Finally, we combine `k=5` relevant documents for each example into a string, to be used later in a prompt.

```yaml
    - _target_: ragfoundry.processing.local_steps.prompter.TextPrompter
      inputs: [train, test]
      prompt_file: ragfoundry/processing/prompts/qa.txt
      output_key: prompt
      mapping:
            question: query
            context: positive_passages
```

Next is the prompt generation step; we used a QA prompt with `question` and `context` placeholders. We map the values using the `mapping` keyword.

> [!IMPORTANT]
> There is no model-dependency in the prompt building. For models/tokenizers supporting a chat format, the prompt is going to be uttered by the *user* role, where the chat, including a system instruction, is constructed only in the training and inference stages. For models/tokenizers not supporting a chat format, a template can be provided by the users, to be used in the training and inference stages.

Finally we write the results to files.

## Training

Training is done on the generated files. The training configuration has 3 parts: model, training arguments and data.

```yaml
model:
    _target_: ragfoundry.models.hf.HFTrain
    model_name_or_path: microsoft/Phi-3-mini-128k-instruct
    load_in_4bit: false
    load_in_8bit: true
    lora:
        lora_alpha: 16
        lora_dropout: 0.1
        peft_type: LORA
        r: 16
        target_modules:
            - qkv_proj
        task_type: CAUSAL_LM
    completion_start: <|assistant|>
    instruction_in_prompt:
    max_sequence_len: 2000
```

Model loading is implemented using the `HFTrain` class, which loads models from HuggingFace hub and uses PEFT adapters. Other classes can be implemented. The important keys here are: `completion_start` which indicates the beginning of the text where loss is to be calculated. This is model/tokenizer specific. Additionally, there is the `instruction_in_prompt` key, which if set to *True*, inserts the system instruction in the prompt, for models which do not support a dedicated system role.

```yaml
train:
    output_dir: ./trained_model/
    gradient_accumulation_steps: 2
    learning_rate: 1e-4
    logging_steps: 10
    lr_scheduler_type: cosine
    num_train_epochs: 1
    per_device_train_batch_size: 1
    optim: paged_adamw_8bit
    warmup_ratio: 0.03
    weight_decay: 0.001
```

Training is done using the `SFTTrainer` in `TRL`. Training arguments are based on HuggingFace `Trainer`.

```yaml
instruction: ragfoundry/processing/prompts/prompt_instructions/qa-yes-no.txt
template:
data_file: pubmed-rag-train.jsonl
input_key: prompt
output_key: answers
limit:
shuffle:
hfhub_tag:
```

Here are they important keys:

- The instruction file to use for training (should later be used for inference as well). In the case of PubmedQA, the answers are either Yes or No, so we specify this in the system instruction.
- If the model/tokenizer do not support a chat template, the user needs to provided a custom template; they placeholders to fill are `query` and `output`.
- Data file is the processed file to train on.
- Input key is the prompt.
- Output key is completion text to learn.
- Limit and shuffle can be used to filter the dataset for debugging purposes.
- The framework can push the trained model to `hfhub_tab`.

We create a training job by running:

```sh
python training.py -cp configs/paper -cn training-pubmed
```

## Inference

In the inference stage, we take the processed dataset and LLM and make predictions. The LLM can be fine-tuned. The processed data encapsulates the RAG interactions: pre-processing, retrieval, ranking, prompt-creation, and possibly other types of transformations. So this step deals with producing the predictions to be evaluated.

It is simple in nature, described by the following configuration:

```yaml
model:
    _target_: ragfoundry.models.hf.HFInference
    model_name_or_path: microsoft/Phi-3-mini-128k-instruct
    load_in_4bit: false
    load_in_8bit: true
    device_map: auto
    trust_remote_code: true
    instruction: ragfoundry/processing/prompts/prompt_instructions/qa-yes-no.txt
    lora_path: ./trained_model/checkpoint
    generation:
        do_sample: false
        max_new_tokens: 50
        return_full_text: false

data_file: pubmed-rag-test.jsonl
generated_file: pubmed-rag-test-generated.jsonl
input_key: prompt
generation_key: output
target_key: answers
limit:
```

The model section deals with details regarding the model loading and generation options. System instruction can be provided, as we mentioned previously: the datasets are model independent, and all model details (system instruction, custom chat template) are needed only during training and inference. Similarly, `instruct_in_prompt` inserts the system instruction inside the prompt, for models which don't support a *system* role.

Other parameters:

- Data file is the processed file.
- Generated file is the file that will be created with the completions (and labels, for easy debugging).
- Target key is the label keyword.
- Limit: to a number of examples, for debugging.

In order to run inference:

```sh
python inference.py -cp configs/paper -cn inference-pubmed
```

## Evaluations

The evaluation module takes the produced inference file and the original processed dataset and runs a list of evaluations, producing a final results file, in a YAML format. The evaluations are represented as metric classes.

We implement several metrics including: a wrapper for HuggingFace `evaluate` class, which can accept a list of metrics, EM, F1, classification (accuracy, precision, recall, F1), BERTScore, Semantic similarity (using a customizable cross-encoder). The module can also run metrics from [DeepEval](https://docs.confident-ai.com/docs/getting-started), which offers a large collection of LLM evaluations.

The configuration for the evaluation looks like this:

```yaml
answer_processor:
    _target_: ragfoundry.processing.answer_processors.regex.RegexAnswer
    capture_pattern:
    stopping_pattern:

metrics:
    - _target_: ragfoundry.evaluation.metrics.Classification
      mapping:
        "yes": 1
        "no": 0
        "maybe": 2
      else_value: 2

key_names:
    generated: text
    label: answers
    query: query

results_file: evaluation-pubmed-rag.yaml
generated_file: pubmed-rag-test-generated.jsonl
data_file: pubmed-rag-test.jsonl
limit:
```

The evaluation module introduces the concept of an **Answer Processor**. This class can run post-processing on the generated text, including: aligning text with the expect output, implement evaluation-specific formatting, extracting the specific sections, processing meta-data like citations, etc.

The default processor is called `RegexAnswer`; it can filter text, based on a python regex capture pattern. It can also split text using a stopping pattern. For example, in the Chain-of-Thought reasoning we used in the paper, the model is instruction to explain its answer, cite if needed and finally print the final results in the following format `<ANSWER>: ...`. We can use this format as a capture pattern; thus models that learn to answer using this pattern (obey the instruction) will score higher.

For PubmedQA we use a **classification metric**; we provide a mapping of keys and a default key, since the PubmedQA expert annotated test set can contain Yes, No or Maybe, as answers.

The rest of the arguments are straightforward:

- Keyword names for input, output and target.
- Name of inference file, name of the processed data.
- Name for the results summary report.
- Limit, for debugging purposes.

Running the evaluation:

```sh
python evaluation.py -cp configs/paper -cn evaluation-pubmed
```

## Summary

In this tutorial, we enhanced an LLM to better perform Q&A on the PubmedQA task, by generating a training dataset containing relevant context, fine-tuning and evaluating the model on the testset. By modifying the configurations presented here, one can run an evaluation on an untrained model and see the benefit of RAG. One can implement other RAG techniques; for example, see the ASQA tutorial for a more advanced usecase (as well as more thorough explanations), including external retrieval, OpenAI integration and Chain-of-thought prompting: [data creation](./processing.md), [training](./training.md), [inference](./inference.md) and [evaluation](./evaluation.md).
