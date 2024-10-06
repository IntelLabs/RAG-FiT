# Training

Training is done on the processed files. The training configuration has 3 parts: model, training arguments and data.

```yaml
model:
    _target_: ragfit.models.hf.HFTrain
    model_name_or_path: microsoft/Phi-3-mini-128k-instruct
    load_in_4bit: false
    load_in_8bit: true
    lora:
        bias: none
        fan_in_fan_out: false
        lora_alpha: 16
        lora_dropout: 0.1
        peft_type: LORA
        r: 16
        target_modules:
            - qkv_proj
        task_type: CAUSAL_LM
        use_rslora: true
    completion_start: <|assistant|>
    instruction_in_prompt:
    max_sequence_len: 4000
```
Model loading is done in the `HFTrain` class, which loads models from HuggingFace hub and uses PEFT adapters. Other
classes can be implemented. The important keys here are: `completion_start` which indicates the beginning of the text
where loss is to be calculated. This is model/tokenizer specific. Additionally, there is the `instruction_in_prompt`
key, which if set to *True*, inserts the system instruction in the prompt, for models which do not support a dedicated
system role.

Next is the training arguments:
```yaml
train:
    output_dir: ./trained_models/
    bf16: false
    fp16: false
    gradient_accumulation_steps: 2
    group_by_length:
    learning_rate: 1e-4
    logging_steps: 10
    lr_scheduler_type: cosine
    max_steps: -1
    num_train_epochs: 1
    per_device_train_batch_size: 1
    optim: paged_adamw_8bit
    remove_unused_columns: true
    save_steps: 20000
    save_total_limit: 1
    warmup_ratio: 0.03
    weight_decay: 0.001
    report_to:
```

Training is done using the `SFTTrainer` in `TRL`. Training arguments are based on HuggingFace `Trainer`.

Finally, data and other options:
```yaml
instruction: ragfit/processing/prompts/prompt_instructions/qa.txt
template:
data_file:
input_key: prompt
output_key:
resume_checkpoint:
limit:
shuffle:
hfhub_tag:
use_wandb:
experiment:
wandb_entity:
```

Here are they important keys:

- The instruction file to use for training (should later be used for inference as well).
- If the model/tokenizer do not support a chat template, the user needs to provided a custom template; they placeholders to
fill are `query` and `output`.
- Data file is the processed file to train on.
- Input key is the prompt.
- Output key is completion text to learn.
- Limit and shuffle can be used to filter the dataset for debugging purposes.
- The framework can push the trained model to `hfhub_tab`.
- The last three keys related to experiment tracking using WANDB. Other services can be used by modifying the
`report_to` key.

## Sending Runs

As we mentioned in the Data Augmentation page, we demonstrate the framework functionality using the ASQA dataset and the
Phi-3 model, experimenting with 5 different configurations. Only 2 configurations require fine-tuning. One can send the
training job like this:

```sh
python training.py -cp configs/paper -cn training-asqa  \
       data_file=asqa-prefix-train.jsonl                \
       output_key=answers                               \
       train.output_dir=./trained_models_context/
```

The `-cp` and `-cn` are overrides for the default configuration, which is `./configs/training.yaml`. Then there are
overrides for the processed data file to use, the name of the label key and where to save the trained model. Overrides
are based on the [Hydra](https://hydra.cc/) vocabulary.

For the CoT model with RAFT contexts, we run:
```sh
python training.py -cp configs/paper -cn training-asqa  \
       data_file=asqa-raft-cot-train.jsonl              \
       output_key=generated_answer                      \
       train.output_dir=./trained_models_cot/
```