import matplotlib.pyplot as plt

MODEL_MAP = {
    # Baselines
    "greedy": "Greedy",
    "random": "Random",
    "lead_1": "Lead 1",
    "lead_n": "Lead N",
    "textrank": "TextRank",

    # OLMo
    "OLMo-7B-Instruct-hf": "OLMo (7B)",
    "OLMo-7B-0724-Instruct-hf": "OLMo 0724 (7B)",

    # Mistral
    "Mistral-7B-Instruct-v0.3": "Mistral (7B)",
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral (8x7B)",

    # Llama 2
    "Llama-2-7b-chat-hf": "Llama 2 (7B)",
    "Llama-2-13b-chat-hf": "Llama 2 (13B)",
    "Llama-2-70b-chat-hf": "Llama 2 (70B)",

    # Llama 3
    "Meta-Llama-3-8B-Instruct": "Llama 3 (8B)",
    "Meta-Llama-3-70B-Instruct": "Llama 3 (70B)",

    # Llama 3.1
    "Meta-Llama-3.1-8B-Instruct": "Llama 3.1 (8B)",
    "Meta-Llama-3.1-70B-Instruct": "Llama 3.1 (70B)",

    # Llama 3.1 (quantized)
    "Meta-Llama-3.1-8B-Instruct-quantized.w8a8": "Llama 3.1 (8B_Q8)",
    "Meta-Llama-3.1-70B-Instruct-quantized.w8a8": "Llama 3.1 (70B_Q8)",

    # GPT-4o
    # "gpt-4o-2024-05-13": "GPT-4o (05/24)",
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
    "gpt-4o-2024-08-06": "GPT-4o",
}
