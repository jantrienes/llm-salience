DATASET ?= qmsum-generic
PROMPT_NAME ?= qmsum-generic
MODEL ?= meta-llama/Meta-Llama-3.1-8B-Instruct
ENGINE ?= vllm
TEMPERATURE ?= 0.3
N_SAMPLES ?= 5
GPUS ?= 1
DURATION ?= 01:00:00

baselines:
	sbatch --time=$(DURATION) scripts/summarization_baselines.sh \
	--documents_json data/processed/$(DATASET)/documents.json \
	--output_path output/$(DATASET)/

summarization:
	sbatch --gres=gpu:a100_80gb:$(GPUS) --time=$(DURATION) scripts/summarization.sh \
	--input_json data/processed/$(DATASET)/documents.json \
	--output_path output/$(DATASET)/$(shell basename $(MODEL))/summaries/ \
	--model $(MODEL) \
	--prompt_name $(PROMPT_NAME) \
	--engine $(ENGINE) \
	--temperature $(TEMPERATURE) \
	--n_samples $(N_SAMPLES)

temperature_sweep:
	sbatch --array=0-20%5 --gres=gpu:a100_80gb:$(GPUS) --time=$(DURATION) scripts/summarization.sh \
	--input_json data/processed/$(DATASET)/documents.json \
	--output_path output/$(DATASET)/$(shell basename $(MODEL))/summaries/ \
	--model $(MODEL) \
	--prompt_name $(PROMPT_NAME) \
	--engine vllm \
	--n_samples $(N_SAMPLES)

claim_extraction:
	sbatch --gres=gpu:a100_80gb:$(GPUS) --time=$(DURATION) scripts/claim_extraction.sh \
    --input_json data/processed/$(DATASET)/documents.json \
    --output_json output/$(DATASET)/facts.json

CODE=src
TESTS=tests

export OPENAI_API_KEY=''

format:
	black ${CODE} ${TESTS}
	isort ${CODE} ${TESTS}

test:
	pytest --cov-report html --cov=${CODE} ${CODE} ${TESTS}

lint:
	pylint --recursive=y --disable=R,C ${CODE} ${TESTS}
	black --check ${CODE} ${TESTS}
	isort --check-only ${CODE} ${TESTS}

lintci:
	pylint --recursive=y --disable=W,R,C ${CODE} ${TESTS}
	black --check ${CODE} ${TESTS}
	isort --check-only ${CODE} ${TESTS}


.PHONY: baselines openai vllm temperature_sweep
