python evaluation.py -cn align/evaluation-asqa -m \
       generated_file=baseline-no-rag-asqa-test-gpt4-generated.jsonl,baseline-no-rag-asqa-test-llama-3B-generated.jsonl,baseline-no-rag-asqa-test-llama-8B-generated.jsonl,baseline-asqa-test-gpt4-generated.jsonl,baseline-asqa-test-llama-3B-generated.jsonl,baseline-asqa-test-llama-8B-generated.jsonl

python evaluation.py -cn cot/evaluation-asqa-format -m \
       feature=format                                                    \
       generated_file=cot-asqa-test-gpt4-generated.jsonl,cot-asqa-test-llama-3B-generated.jsonl,cot-asqa-test-llama-8B-generated.jsonl,cot-fewshot-asqa-test-gpt4-generated.jsonl,cot-fewshot-asqa-test-llama-3B-generated.jsonl,cot-fewshot-asqa-test-llama-8B-generated.jsonl,cot1-asqa-test-generated-lora.jsonl,cot1-asqa-test-generated.jsonl,cot1-asqa-test-gpt4-generated.jsonl,cot1-asqa-test-llama-3B-generated.jsonl,cot1-asqa-test-llama-8B-generated.jsonl,cot1-fewshot-asqa-test-gpt4-generated.jsonl,cot1-fewshot-asqa-test-llama-3B-generated.jsonl,cot1-fewshot-asqa-test-llama-8B-generated.jsonl,cot3-asqa-test-gpt4-generated.jsonl,cot3-asqa-test-llama-3B-generated.jsonl,cot3-asqa-test-llama-8B-generated.jsonl,cot3-fewshot-asqa-test-gpt4-generated.jsonl,cot3-fewshot-asqa-test-llama-3B-generated.jsonl,cot3-fewshot-asqa-test-llama-8B-generated.jsonl,cot3-fewshot-summary-asqa-test-llama-8B-generated.jsonl,cot3-fewshot-voting-asqa-test-llama-8B-generated.jsonl,cot3-formatted-asqa-test-llama-3B-generated.jsonl,cot3-formatted-asqa-test-llama-8B-generated.jsonl,cot5-asqa-test-gpt4-generated.jsonl,cot5-asqa-test-llama-3B-generated.jsonl,cot5-asqa-test-llama-8B-generated.jsonl,cot5-fewshot-asqa-test-gpt4-generated.jsonl,cot5-fewshot-asqa-test-llama-3B-generated.jsonl,cot5-fewshot-asqa-test-llama-8B-generated.jsonl,cot5-fewshot-summary-asqa-test-llama-8B-generated.jsonl,cot5-fewshot-voting-asqa-test-llama-8B-generated.jsonl,cot10-asqa-test-gpt4-generated.jsonl,cot10-asqa-test-llama-3B-generated.jsonl,cot10-asqa-test-llama-8B-generated.jsonl,cot10-fewshot-asqa-test-gpt4-generated.jsonl,cot10-fewshot-asqa-test-llama-3B-generated.jsonl,cot10-fewshot-asqa-test-llama-8B-generated.jsonl,cot10-fewshot-summary-asqa-test-llama-8B-generated.jsonl,cot10-fewshot-voting-asqa-test-llama-8B-generated.jsonl

python evaluation.py -cn align/evaluation-hotpotqa -m \
       generated_file=baseline-no-rag-hotpotqa-test-gpt4-generated.jsonl,baseline-no-rag-hotpotqa-test-llama-3B-generated.jsonl,baseline-no-rag-hotpotqa-test-llama-8B-generated.jsonl,baseline-hotpotqa-test-gpt4-generated.jsonl,baseline-hotpotqa-test-llama-3B-generated.jsonl,baseline-hotpotqa-test-llama-8B-generated.jsonl


python evaluation.py -cn cot/evaluation-hotpotqa-format -m \
       feature=format                                                    \
       generated_file=cot-hotpotqa-test-gpt4-generated.jsonl,cot-hotpotqa-test-llama-3B-generated.jsonl,cot-hotpotqa-test-llama-8B-generated.jsonl,cot-fewshot-hotpotqa-test-gpt4-generated.jsonl,cot-fewshot-hotpotqa-test-llama-3B-generated.jsonl,cot-fewshot-hotpotqa-test-llama-8B-generated.jsonl,cot1-hotpotqa-test-generated-lora.jsonl,cot1-hotpotqa-test-generated.jsonl,cot1-hotpotqa-test-gpt4-generated.jsonl,cot1-hotpotqa-test-llama-3B-generated.jsonl,cot1-hotpotqa-test-llama-8B-generated.jsonl,cot1-fewshot-hotpotqa-test-gpt4-generated.jsonl,cot1-fewshot-hotpotqa-test-llama-3B-generated.jsonl,cot1-fewshot-hotpotqa-test-llama-8B-generated.jsonl,cot3-hotpotqa-test-gpt4-generated.jsonl,cot3-hotpotqa-test-llama-3B-generated.jsonl,cot3-hotpotqa-test-llama-8B-generated.jsonl,cot3-fewshot-hotpotqa-test-gpt4-generated.jsonl,cot3-fewshot-hotpotqa-test-llama-3B-generated.jsonl,cot3-fewshot-hotpotqa-test-llama-8B-generated.jsonl,cot3-fewshot-summary-hotpotqa-test-llama-8B-generated.jsonl,cot3-fewshot-voting-hotpotqa-test-llama-8B-generated.jsonl,cot3-formatted-hotpotqa-test-llama-3B-generated.jsonl,cot3-formatted-hotpotqa-test-llama-8B-generated.jsonl,cot5-hotpotqa-test-gpt4-generated.jsonl,cot5-hotpotqa-test-llama-3B-generated.jsonl,cot5-hotpotqa-test-llama-8B-generated.jsonl,cot5-fewshot-hotpotqa-test-gpt4-generated.jsonl,cot5-fewshot-hotpotqa-test-llama-3B-generated.jsonl,cot5-fewshot-hotpotqa-test-llama-8B-generated.jsonl,cot5-fewshot-summary-hotpotqa-test-llama-8B-generated.jsonl,cot5-fewshot-voting-hotpotqa-test-llama-8B-generated.jsonl,cot10-hotpotqa-test-gpt4-generated.jsonl,cot10-hotpotqa-test-llama-3B-generated.jsonl,cot10-hotpotqa-test-llama-8B-generated.jsonl,cot10-fewshot-hotpotqa-test-gpt4-generated.jsonl,cot10-fewshot-hotpotqa-test-llama-3B-generated.jsonl,cot10-fewshot-hotpotqa-test-llama-8B-generated.jsonl,cot10-fewshot-summary-hotpotqa-test-llama-8B-generated.jsonl,cot10-fewshot-voting-hotpotqa-test-llama-8B-generated.jsonl

python evaluation.py -cn align/evaluation-triviaqa -m \
       generated_file=baseline-no-rag-triviaqa-test-gpt4-generated.jsonl,baseline-no-rag-triviaqa-test-llama-3B-generated.jsonl,baseline-no-rag-triviaqa-test-llama-8B-generated.jsonl,baseline-triviaqa-test-gpt4-generated.jsonl,baseline-triviaqa-test-llama-3B-generated.jsonl,baseline-triviaqa-test-llama-8B-generated.jsonl

python evaluation.py -cn cot/evaluation-triviaqa-format -m \
       feature=format                                                    \
       generated_file=cot-triviaqa-test-gpt4-generated.jsonl,cot-triviaqa-test-llama-3B-generated.jsonl,cot-triviaqa-test-llama-8B-generated.jsonl,cot-fewshot-triviaqa-test-gpt4-generated.jsonl,cot-fewshot-triviaqa-test-llama-3B-generated.jsonl,cot-fewshot-triviaqa-test-llama-8B-generated.jsonl,cot1-triviaqa-test-generated-lora.jsonl,cot1-triviaqa-test-generated.jsonl,cot1-triviaqa-test-gpt4-generated.jsonl,cot1-triviaqa-test-llama-3B-generated.jsonl,cot1-triviaqa-test-llama-8B-generated.jsonl,cot1-fewshot-triviaqa-test-gpt4-generated.jsonl,cot1-fewshot-triviaqa-test-llama-3B-generated.jsonl,cot1-fewshot-triviaqa-test-llama-8B-generated.jsonl,cot3-triviaqa-test-gpt4-generated.jsonl,cot3-triviaqa-test-llama-3B-generated.jsonl,cot3-triviaqa-test-llama-8B-generated.jsonl,cot3-fewshot-triviaqa-test-gpt4-generated.jsonl,cot3-fewshot-triviaqa-test-llama-3B-generated.jsonl,cot3-fewshot-triviaqa-test-llama-8B-generated.jsonl,cot3-fewshot-summary-triviaqa-test-llama-8B-generated.jsonl,cot3-fewshot-voting-triviaqa-test-llama-8B-generated.jsonl,cot3-formatted-triviaqa-test-llama-3B-generated.jsonl,cot3-formatted-triviaqa-test-llama-8B-generated.jsonl,cot5-triviaqa-test-gpt4-generated.jsonl,cot5-triviaqa-test-llama-3B-generated.jsonl,cot5-triviaqa-test-llama-8B-generated.jsonl,cot5-fewshot-triviaqa-test-gpt4-generated.jsonl,cot5-fewshot-triviaqa-test-llama-3B-generated.jsonl,cot5-fewshot-triviaqa-test-llama-8B-generated.jsonl,cot5-fewshot-summary-triviaqa-test-llama-8B-generated.jsonl,cot5-fewshot-voting-triviaqa-test-llama-8B-generated.jsonl,cot10-triviaqa-test-gpt4-generated.jsonl,cot10-triviaqa-test-llama-3B-generated.jsonl,cot10-triviaqa-test-llama-8B-generated.jsonl,cot10-fewshot-triviaqa-test-gpt4-generated.jsonl,cot10-fewshot-triviaqa-test-llama-3B-generated.jsonl,cot10-fewshot-triviaqa-test-llama-8B-generated.jsonl,cot10-fewshot-summary-triviaqa-test-llama-8B-generated.jsonl,cot10-fewshot-voting-triviaqa-test-llama-8B-generated.jsonl
