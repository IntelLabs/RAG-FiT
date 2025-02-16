# CoT local runs
python inference.py -cn cot/inference-asqa-cot-rag                                                          \
       -m model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-asqa-cot-rag                                                      \
       -m +model.examples=ragfit/processing/prompts/fewshots/cot-rag.txt                                \
       generated_file=cot-fewshot-asqa-test-generated.jsonl                                             \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-triviaqa-cot-rag                                                      \
       -m model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-triviaqa-cot-rag                                                  \
       -m +model.examples=ragfit/processing/prompts/fewshots/cot-rag.txt                                \
       generated_file=cot-fewshot-triviaqa-test-generated.jsonl                                         \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-hotpotqa-cot-rag                                                      \
       -m model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-hotpotqa-cot-rag                                                  \
       -m +model.examples=ragfit/processing/prompts/fewshots/cot-rag.txt                                \
       generated_file=cot-fewshot-hotpotqa-test-generated.jsonl                                         \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct


# RaR local runs
python inference.py -cn cot/inference-asqa-rar-rag -m                                                   \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-asqa-rar-rag -m               \
       model.examples=ragfit/processing/prompts/fewshots/rar-rag.txt \
       generated_file=cot1-fewshot-asqa-test-generated.jsonl \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-triviaqa-rar-rag -m                                               \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-triviaqa-rar-rag -m               \
       model.examples=ragfit/processing/prompts/fewshots/rar-rag.txt \
       generated_file=cot1-fewshot-triviaqa-test-generated.jsonl \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-hotpotqa-rar-rag -m                                               \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-hotpotqa-rar-rag -m               \
       model.examples=ragfit/processing/prompts/fewshots/rar-rag.txt \
       generated_file=cot1-fewshot-hotpotqa-test-generated.jsonl \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct


# SQuARE, aka rarN runs, local models
python inference.py -cn cot/inference-asqa-rarN-rag -m      \
       generated_file=cot3-fewshot-asqa-test-generated.jsonl \
       model.extra_keys.N="three" \
       model.examples=ragfit/processing/prompts/fewshots/3-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-hotpotqa-rarN-rag -m      \
       generated_file=cot3-fewshot-hotpotqa-test-generated.jsonl \
       model.extra_keys.N="three" \
       model.examples=ragfit/processing/prompts/fewshots/3-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-triviaqa-rarN-rag -m      \
       generated_file=cot3-fewshot-triviaqa-test-generated.jsonl \
       model.extra_keys.N="three" \
       model.examples=ragfit/processing/prompts/fewshots/3-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-asqa-rarN-rag -m        \
       generated_file=cot3-fewshot-asqa-test-generated.jsonl \
       model.extra_keys.N="three" \
       model.examples=ragfit/processing/prompts/fewshots/3-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-hotpotqa-rarN-rag -m        \
       generated_file=cot3-fewshot-hotpotqa-test-generated.jsonl \
       model.extra_keys.N="three" \
       model.examples=ragfit/processing/prompts/fewshots/3-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-triviaqa-rarN-rag -m        \
       generated_file=cot3-fewshot-triviaqa-test-generated.jsonl \
       model.extra_keys.N="three" \
       model.examples=ragfit/processing/prompts/fewshots/3-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-asqa-rarN-rag -m      \
       generated_file=cot5-fewshot-asqa-test-generated.jsonl \
       model.extra_keys.N="five" \
       model.examples=ragfit/processing/prompts/fewshots/5-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-hotpotqa-rarN-rag -m      \
       generated_file=cot5-fewshot-hotpotqa-test-generated.jsonl \
       model.extra_keys.N="five" \
       model.examples=ragfit/processing/prompts/fewshots/5-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-triviaqa-rarN-rag -m      \
       generated_file=cot5-fewshot-triviaqa-test-generated.jsonl \
       model.extra_keys.N="five" \
       model.examples=ragfit/processing/prompts/fewshots/5-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-asqa-rarN-rag -m        \
       generated_file=cot5-fewshot-asqa-test-generated.jsonl \
       model.extra_keys.N="five" \
       model.examples=ragfit/processing/prompts/fewshots/5-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-hotpotqa-rarN-rag -m        \
       generated_file=cot5-fewshot-hotpotqa-test-generated.jsonl \
       model.extra_keys.N="five" \
       model.examples=ragfit/processing/prompts/fewshots/5-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-triviaqa-rarN-rag -m        \
       generated_file=cot5-fewshot-triviaqa-test-generated.jsonl \
       model.extra_keys.N="five" \
       model.examples=ragfit/processing/prompts/fewshots/5-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-asqa-rarN-rag -m        \
       generated_file=cot10-fewshot-asqa-test-generated.jsonl \
       model.extra_keys.N="ten" \
       model.examples=ragfit/processing/prompts/fewshots/10-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-hotpotqa-rarN-rag -m        \
       generated_file=cot10-fewshot-hotpotqa-test-generated.jsonl \
       model.extra_keys.N="ten" \
       model.examples=ragfit/processing/prompts/fewshots/10-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-triviaqa-rarN-rag -m        \
       generated_file=cot10-fewshot-triviaqa-test-generated.jsonl \
       model.extra_keys.N="ten" \
       model.examples=ragfit/processing/prompts/fewshots/10-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-asqa-rarN-rag -m        \
       generated_file=cot10-fewshot-asqa-test-generated.jsonl \
       model.extra_keys.N="ten" \
       model.examples=ragfit/processing/prompts/fewshots/10-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-hotpotqa-rarN-rag -m        \
       generated_file=cot10-fewshot-hotpotqa-test-generated.jsonl \
       model.extra_keys.N="ten" \
       model.examples=ragfit/processing/prompts/fewshots/10-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct
python inference.py -cn cot/inference-triviaqa-rarN-rag -m        \
       generated_file=cot10-fewshot-triviaqa-test-generated.jsonl \
       model.extra_keys.N="ten" \
       model.examples=ragfit/processing/prompts/fewshots/10-rag.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct



# Inference with 2 aggregation methods: Summary and Vote
python inference.py -cn cot/inference-asqa-rarN-rag -m         \
       generated_file=cot3-fewshot-summary-asqa-test-generated.jsonl \
       model.extra_keys.N="three" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-summary.txt \
       model.examples=ragfit/processing/prompts/fewshots/3-rag-summary.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-asqa-rarN-rag -m         \
       generated_file=cot3-fewshot-voting-asqa-test-generated.jsonl \
       model.extra_keys.N="three" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-voting.txt \
       model.examples=ragfit/processing/prompts/fewshots/3-rag-voting.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-hotpotqa-rarN-rag -m             \
       generated_file=cot3-fewshot-summary-hotpotqa-test-generated.jsonl \
       model.extra_keys.N="three" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-summary.txt \
       model.examples=ragfit/processing/prompts/fewshots/3-rag-summary.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-hotpotqa-rarN-rag -m             \
       generated_file=cot3-fewshot-voting-hotpotqa-test-generated.jsonl \
       model.extra_keys.N="three" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-voting.txt \
       model.examples=ragfit/processing/prompts/fewshots/3-rag-voting.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-triviaqa-rarN-rag -m             \
       generated_file=cot3-fewshot-summary-triviaqa-test-generated.jsonl \
       model.extra_keys.N="three" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-summary.txt \
       model.examples=ragfit/processing/prompts/fewshots/3-rag-summary.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-triviaqa-rarN-rag -m             \
       generated_file=cot3-fewshot-voting-triviaqa-test-generated.jsonl \
       model.extra_keys.N="three" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-voting.txt \
       model.examples=ragfit/processing/prompts/fewshots/3-rag-voting.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-asqa-rarN-rag -m             \
       generated_file=cot5-fewshot-summary-asqa-test-generated.jsonl \
       model.extra_keys.N="five" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-summary.txt \
       model.examples=ragfit/processing/prompts/fewshots/5-rag-summary.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-asqa-rarN-rag -m             \
       generated_file=cot5-fewshot-voting-asqa-test-generated.jsonl \
       model.extra_keys.N="five" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-voting.txt \
       model.examples=ragfit/processing/prompts/fewshots/5-rag-voting.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-hotpotqa-rarN-rag -m             \
       generated_file=cot5-fewshot-summary-hotpotqa-test-generated.jsonl \
       model.extra_keys.N="five" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-summary.txt \
       model.examples=ragfit/processing/prompts/fewshots/5-rag-summary.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-hotpotqa-rarN-rag -m             \
       generated_file=cot5-fewshot-voting-hotpotqa-test-generated.jsonl \
       model.extra_keys.N="five" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-voting.txt \
       model.examples=ragfit/processing/prompts/fewshots/5-rag-voting.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-triviaqa-rarN-rag -m             \
       generated_file=cot5-fewshot-summary-triviaqa-test-generated.jsonl \
       model.extra_keys.N="five" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-summary.txt \
       model.examples=ragfit/processing/prompts/fewshots/5-rag-summary.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-triviaqa-rarN-rag -m             \
       generated_file=cot5-fewshot-voting-triviaqa-test-generated.jsonl \
       model.extra_keys.N="five" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-voting.txt \
       model.examples=ragfit/processing/prompts/fewshots/5-rag-voting.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-asqa-rarN-rag -m             \
       generated_file=cot10-fewshot-summary-asqa-test-generated.jsonl \
       model.extra_keys.N="ten" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-summary.txt \
       model.examples=ragfit/processing/prompts/fewshots/10-rag-summary.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-asqa-rarN-rag -m             \
       generated_file=cot10-fewshot-voting-asqa-test-generated.jsonl \
       model.extra_keys.N="ten" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-voting.txt \
       model.examples=ragfit/processing/prompts/fewshots/10-rag-voting.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-hotpotqa-rarN-rag -m             \
       generated_file=cot10-fewshot-summary-hotpotqa-test-generated.jsonl \
       model.extra_keys.N="ten" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-summary.txt \
       model.examples=ragfit/processing/prompts/fewshots/10-rag-summary.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-hotpotqa-rarN-rag -m             \
       generated_file=cot10-fewshot-voting-hotpotqa-test-generated.jsonl \
       model.extra_keys.N="ten" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-voting.txt \
       model.examples=ragfit/processing/prompts/fewshots/10-rag-voting.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-triviaqa-rarN-rag -m             \
       generated_file=cot10-fewshot-summary-triviaqa-test-generated.jsonl \
       model.extra_keys.N="ten" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-summary.txt \
       model.examples=ragfit/processing/prompts/fewshots/10-rag-summary.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
python inference.py -cn cot/inference-triviaqa-rarN-rag -m             \
       generated_file=cot10-fewshot-voting-triviaqa-test-generated.jsonl \
       model.extra_keys.N="ten" \
       model.instruction=ragfit/processing/prompts/prompt_instructions/qa-rephrase-N-rag-voting.txt \
       model.examples=ragfit/processing/prompts/fewshots/10-rag-voting.txt \
       model.model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct




# GPT-4o Inference
python inference.py -cn cot/inference-asqa-baseline-no-rag-gpt
python inference.py -cn cot/inference-asqa-baseline-gpt
python inference.py -cn cot/inference-asqa-cot-rag-gpt
python inference.py -cn cot/inference-asqa-cot-rag-gpt \
       model.examples=ragfit/processing/prompts/fewshots/cot-rag.txt \
       generated_file=cot-fewshot-asqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-asqa-rar-rag-gpt
python inference.py -cn cot/inference-asqa-rar-rag-gpt \
       model.examples=ragfit/processing/prompts/fewshots/rar-rag.txt \
       generated_file=cot1-fewshot-asqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-asqa-rarN-rag-gpt \
       model.extra_keys.N=three \
       generated_file=cot3-asqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-asqa-rarN-rag-gpt \
       model.extra_keys.N=three \
       model.examples=ragfit/processing/prompts/fewshots/3-rag.txt \
       generated_file=cot3-fewshot-asqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-asqa-rarN-rag-gpt \
       model.extra_keys.N=five \
       generated_file=cot5-asqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-asqa-rarN-rag-gpt \
       model.extra_keys.N=five \
       model.examples=ragfit/processing/prompts/fewshots/5-rag.txt \
       generated_file=cot5-fewshot-asqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-asqa-rarN-rag-gpt \
       model.extra_keys.N=ten \
       generated_file=cot10-asqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-asqa-rarN-rag-gpt \
       model.extra_keys.N=ten \
       model.examples=ragfit/processing/prompts/fewshots/10-rag.txt \
       generated_file=cot10-fewshot-asqa-test-gpt4-generated.jsonl

python inference.py -cn cot/inference-hotpotqa-baseline-no-rag-gpt
python inference.py -cn cot/inference-hotpotqa-baseline-gpt
python inference.py -cn cot/inference-hotpotqa-cot-rag-gpt
python inference.py -cn cot/inference-hotpotqa-cot-rag-gpt \
       model.examples=ragfit/processing/prompts/fewshots/cot-rag.txt \
       generated_file=cot-fewshot-hotpotqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-hotpotqa-rar-rag-gpt
python inference.py -cn cot/inference-hotpotqa-rar-rag-gpt \
       model.examples=ragfit/processing/prompts/fewshots/rar-rag.txt \
       generated_file=cot1-fewshot-hotpotqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-hotpotqa-rarN-rag-gpt \
       model.extra_keys.N=three \
       generated_file=cot3-hotpotqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-hotpotqa-rarN-rag-gpt \
       model.extra_keys.N=three \
       model.examples=ragfit/processing/prompts/fewshots/3-rag.txt \
       generated_file=cot3-fewshot-hotpotqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-hotpotqa-rarN-rag-gpt \
       model.extra_keys.N=five \
       generated_file=cot5-hotpotqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-hotpotqa-rarN-rag-gpt \
       model.extra_keys.N=five \
       model.examples=ragfit/processing/prompts/fewshots/5-rag.txt \
       generated_file=cot5-fewshot-hotpotqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-hotpotqa-rarN-rag-gpt \
       model.extra_keys.N=ten \
       generated_file=cot10-hotpotqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-hotpotqa-rarN-rag-gpt \
       model.extra_keys.N=ten \
       model.examples=ragfit/processing/prompts/fewshots/10-rag.txt \
       generated_file=cot10-fewshot-hotpotqa-test-gpt4-generated.jsonl

python inference.py -cn cot/inference-triviaqa-baseline-no-rag-gpt
python inference.py -cn cot/inference-triviaqa-baseline-gpt
python inference.py -cn cot/inference-triviaqa-cot-rag-gpt
python inference.py -cn cot/inference-triviaqa-cot-rag-gpt \
       model.examples=ragfit/processing/prompts/fewshots/cot-rag.txt \
       generated_file=cot-fewshot-triviaqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-triviaqa-rar-rag-gpt
python inference.py -cn cot/inference-triviaqa-rar-rag-gpt \
       model.examples=ragfit/processing/prompts/fewshots/rar-rag.txt \
       generated_file=cot1-fewshot-triviaqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-triviaqa-rarN-rag-gpt \
       model.extra_keys.N=three \
       generated_file=cot3-triviaqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-triviaqa-rarN-rag-gpt \
       model.extra_keys.N=three \
       model.examples=ragfit/processing/prompts/fewshots/3-rag.txt \
       generated_file=cot3-fewshot-triviaqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-triviaqa-rarN-rag-gpt \
       model.extra_keys.N=five \
       generated_file=cot5-triviaqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-triviaqa-rarN-rag-gpt \
       model.extra_keys.N=five \
       model.examples=ragfit/processing/prompts/fewshots/5-rag.txt \
       generated_file=cot5-fewshot-triviaqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-triviaqa-rarN-rag-gpt \
       model.extra_keys.N=ten \
       generated_file=cot10-triviaqa-test-gpt4-generated.jsonl
python inference.py -cn cot/inference-triviaqa-rarN-rag-gpt \
       model.extra_keys.N=ten \
       model.examples=ragfit/processing/prompts/fewshots/10-rag.txt \
       generated_file=cot10-fewshot-triviaqa-test-gpt4-generated.jsonl



