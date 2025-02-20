# Slurm-based Execution Environment

## Example (`pubmed-sample`)

```sh
# Baselines
make baselines DATASET=pubmed-sample

# OpenAI models (via LiteLLM)
make summarization ENGINE=litellm GPUS=0 DATASET=pubmed-sample PROMPT_NAME=generic MODEL=gpt-4o-2024-08-06
make summarization ENGINE=litellm GPUS=0 DATASET=pubmed-sample PROMPT_NAME=generic MODEL=gpt-4o-mini-2024-07-18

# Local models (via vLLM)
make summarization DATASET=pubmed-sample PROMPT_NAME=generic MODEL=allenai/OLMo-7B-Instruct-hf
make summarization DATASET=pubmed-sample PROMPT_NAME=generic MODEL=allenai/OLMo-7B-0724-Instruct-hf
make summarization DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Llama-2-7b-chat-hf
make summarization DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Llama-2-13b-chat-hf GPUS=2
make summarization DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Llama-2-70b-chat-hf GPUS=2 DURATION=02:30:00
make summarization DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Meta-Llama-3-8B-Instruct
make summarization DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Meta-Llama-3-70B-Instruct GPUS=2 DURATION=01:00:00
make summarization DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
make summarization DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct GPUS=4 DURATION=01:00:00
make summarization DATASET=pubmed-sample PROMPT_NAME=generic MODEL=mistralai/Mistral-7B-Instruct-v0.3 GPUS=1
make summarization DATASET=pubmed-sample PROMPT_NAME=generic MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1 GPUS=2
```

<details>
<summary>Temperature sweep</summary>

Adjust the duration as necessary.

```sh
make temperature_sweep DATASET=pubmed-sample PROMPT_NAME=generic MODEL=allenai/OLMo-7B-Instruct-hf DURATION=06:00:00
make temperature_sweep DATASET=pubmed-sample PROMPT_NAME=generic MODEL=allenai/OLMo-7B-0724-Instruct-hf DURATION=06:00:00
make temperature_sweep DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Llama-2-7b-chat-hf DURATION=06:00:00
make temperature_sweep DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Llama-2-13b-chat-hf GPUS=2 DURATION=06:00:00
make temperature_sweep DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Llama-2-70b-chat-hf GPUS=2 DURATION=06:00:00
make temperature_sweep DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Meta-Llama-3-8B-Instruct DURATION=06:00:00
make temperature_sweep DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Meta-Llama-3-70B-Instruct GPUS=2 DURATION=06:00:00
make temperature_sweep DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct DURATION=06:00:00
make temperature_sweep DATASET=pubmed-sample PROMPT_NAME=generic MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct GPUS=4 DURATION=06:00:00
make temperature_sweep DATASET=pubmed-sample PROMPT_NAME=generic MODEL=mistralai/Mistral-7B-Instruct-v0.3 GPUS=1 DURATION=06:00:00
make temperature_sweep DATASET=pubmed-sample PROMPT_NAME=generic MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1 GPUS=2 DURATION=06:00:00
```

</details>


Claim entailment

```sh
make claim_extraction DATASET=pubmed-sample
./scripts/claim_entailment_task_list.sh pubmed-sample output/pubmed-sample/facts.json
```

Discord questions

```sh
# Get reference answers and answer claims
sbatch scripts/qa.sh \
--documents_json data/processed/pubmed-sample/documents.json \
--questions_json output/pubmed-sample/discord_questions.json \
--answers_json output/pubmed-sample/discord_answers.json \
--answer_facts_json output/pubmed-sample/discord_facts.json

# Entailment of reference answer claims
./scripts/claim_entailment_task_list.sh pubmed-sample output/pubmed-sample/discord_facts.json
```

## LLM Introspection

```sh
sbatch --time=01:00:00 scripts/introspection.sh --engine litellm --model gpt-4o-2024-08-06
sbatch --time=01:00:00 scripts/introspection.sh --engine litellm --model gpt-4o-mini-2024-07-18
sbatch --time=01:00:00 --gres=gpu:a100_80gb:1 scripts/introspection.sh --engine vllm --model meta-llama/Llama-2-7b-chat-hf
sbatch --time=01:00:00 --gres=gpu:a100_80gb:2 scripts/introspection.sh --engine vllm --model meta-llama/Llama-2-13b-chat-hf
sbatch --time=03:00:00 --gres=gpu:a100_80gb:2 scripts/introspection.sh --engine vllm --model meta-llama/Llama-2-70b-chat-hf
sbatch --time=01:00:00 --gres=gpu:a100_80gb:1 scripts/introspection.sh --engine vllm --model meta-llama/Meta-Llama-3-8B-Instruct
sbatch --time=03:00:00 --gres=gpu:a100_80gb:2 scripts/introspection.sh --engine vllm --model meta-llama/Meta-Llama-3-70B-Instruct
sbatch --time=01:00:00 --gres=gpu:a100_80gb:1 scripts/introspection.sh --engine vllm --model meta-llama/Meta-Llama-3.1-8B-Instruct
sbatch --time=03:00:00 --gres=gpu:a100_80gb:2 scripts/introspection.sh --engine vllm --model meta-llama/Meta-Llama-3.1-70B-Instruct
sbatch --time=01:00:00 --gres=gpu:a100_80gb:1 scripts/introspection.sh --engine vllm --model mistralai/Mistral-7B-Instruct-v0.3
sbatch --time=01:00:00 --gres=gpu:a100_80gb:2 scripts/introspection.sh --engine vllm --model mistralai/Mixtral-8x7B-Instruct-v0.1
sbatch --time=01:00:00 --gres=gpu:a100_80gb:1 scripts/introspection.sh --engine vllm --model allenai/OLMo-7B-Instruct-hf
sbatch --time=02:00:00 --gres=gpu:a100_80gb:1 scripts/introspection.sh --engine vllm --model allenai/OLMo-7B-0724-Instruct-hf
```
