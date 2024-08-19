# Inference

In the inference stage, we take the processed dataset and LLM and make predictions. The LLM can be fine-tuned. The
processed data encapsulates the RAG interactions: pre-processing, retrieval, ranking, prompt-creation, and possibly
other types of transformations. So this step deals with producing the predictions to be evaluated.

It is simple in nature, described by the following configuration:

```yaml
model:
    _target_: ragfoundry.models.hf.HFInference
    model_name_or_path: microsoft/Phi-3-mini-128k-instruct
    load_in_4bit: false
    load_in_8bit: true
    device_map: auto
    torch_dtype:
    trust_remote_code: true
    instruction: ragfoundry/processing/prompts/prompt_instructions/qa.txt
    instruct_in_prompt: false
    lora_path:
    generation:
        do_sample: false
        max_new_tokens: 50
        max_length:
        temperature:
        top_k:
        top_p:
        return_full_text: false

data_file: asqa-baseline-dev.jsonl
generated_file: asqa-baseline-dev-generated.jsonl
input_key: prompt
generation_key: output
target_key: answers
limit:
```

The model section deals with details regarding the model loading and generation options. System instruction can be
provided, as we mentioned previously: the datasets are model independent, and all model details (system instruction,
custom chat template) are needed only during training and inference. Similarly, `instruct_in_prompt` inserts the system
instruction inside the prompt, for models which don't support a system role.

Other parameters:
- Data file is the processed file.
- Generated file is the file that will be created with the completions (and labels, for easy debugging).
- Target key is the label keyword.
- Limit: to a number of examples, for debugging.

## Running Inference
In order to run evaluations for ASQA, like in the paper, there are 5 configurations to run: baseline, context, context
with fine-tuned model, CoT reasoning, and CoT reasoning with a model that was fine-tuned with distractor documents.

The baseline inference uses the configuration as is; the other calls, use the configuration and just override the value
of the processed data to use and optionally LORA path for the model.


**Baseline**:
```sh
python inference.py -cp configs/paper -cn inference-asqa
```

**Context**:
```sh
python inference.py -cp configs/paper -cn inference-asqa    \
       data_file=asqa-context-dev.jsonl                     \
       generated_file=asqa-context-dev-generated.jsonl
```

**Context with fine-tuned model**:
```sh
python inference.py -cp configs/paper -cn inference-asqa    \
       data_file=asqa-context-dev.jsonl                     \
       generated_file=asqa-context-ft-dev-generated.jsonl   \
       model.lora_path=./path/to/lora/checkpoint
```

**Chain-of-Thought**:
```sh
python inference.py -cp configs/paper -cn inference-asqa    \
       data_file=asqa-cot-dev.jsonl                         \
       generated_file=asqa-cot-ft-dev-generated.jsonl
```

**Chain-of-Thought with fine-tuned model**:
```sh
python inference.py -cp configs/paper -cn inference-asqa    \
       data_file=asqa-cot-dev.jsonl                         \
       generated_file=asqa-cot-ft-dev-generated.jsonl       \
       model.lora_path=./path/to/lora/checkpoint
```

## Running Inference with vLLM Backend

To achieve potentially faster inference speeds, you can run inference using the vLLM backend. The functionality of the inference process remains similar to the previously defined process, with the addition of extra arguments that can be used with the vLLM engine.

Here is an example of an inference configuration using the vLLM engine:

```yaml
model:
    _target_: ragfoundry.models.vllm.VLLMInference
    model_name_or_path: "facebook/opt-125m"
    llm_params:
        dtype: auto
    generation:
        temperature: 0.5
        top_p: 0.95
        seed: 1911
    num_gpus: 1

data_file: my-processed-data.jsnol
generated_file: model-predictions.jsonl
input_key: prompt
generation_key: output
target_key: answers
limit:
```

The main differences in this configuration are as follows:

- `ragfoundry.models.vllm.VLLMInference`: This class is used to utilize the vLLM-based engine.
- `llm_params`: These are optional vLLM arguments that can be passed to the LLM class.
- `generation`: These are optional arguments that define the generation policy. The supported arguments are compatible with vLLM's `SamplingParams`.
- `num_gpus`: This specifies the number of GPUs to use during inference.
