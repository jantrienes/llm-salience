import json
import warnings
from collections import defaultdict
from functools import partial
from json.decoder import JSONDecodeError
from pathlib import Path
from pprint import pprint

import click
import outlines
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from json_repair import repair_json
from pydantic import BaseModel

from info_salience.llm import LitellmGenerator, VLLMGenerator


class SummarizationOutput(BaseModel):
    summary: str


@outlines.prompt
def prompt_generic(text, length_target):
    """
    ## Document
    {{ text }}

    ## Instruction
    Please summarize the above document. Use up to {{ length_target }} words. Respond exactly in following JSON format:

    {
        "summary": "(the {{ length_target }} words summary)"
    }
    """


@outlines.prompt
def prompt_qmsum_generic(text, length_target):
    """
    ## Meeting Transcript
    {{ text }}

    ## Instruction
    Please summarize the above meeting transcript. Use up to {{ length_target }} words. Respond exactly in following JSON format:

    {
        "summary": "(the {{ length_target }} words summary)"
    }
    """


def build_messages(df, prompt_name, length_target):
    if prompt_name == "generic":
        prompt = prompt_generic
    elif prompt_name == "qmsum-generic":
        prompt = prompt_qmsum_generic
    else:
        raise ValueError(f"Invalid prompt name {prompt_name}.")

    messages = []
    for _, row in df.iterrows():
        content = prompt(row["text"], length_target)
        message = [{"role": "user", "content": content}]
        messages.append(message)

    return messages


def parse_response(response, key):
    try:
        response_fixed = repair_json(response)
    except RecursionError:
        print("Recursion error")
        print(response)
        return None
    try:
        summary = json.loads(response_fixed).get(key, None)
    except (JSONDecodeError, AttributeError):
        print("=" * 10, "failed to parse", "=" * 10)
        print(response)
        summary = None
    return summary


def stats(data):
    stats = []
    for target_length, summaries in data.items():
        lens = []
        for summary in summaries:
            if not summary:
                summary = ""
            summary = summary.strip()
            lens.append(len(summary.split()))
        lens = pd.Series(lens)
        lens.name = target_length
        stats.append(lens)
    stats = pd.concat(stats, axis=1)
    empty = (stats == 0).sum(axis=0)
    stats = stats.describe().round(1)
    stats.loc["empty"] = empty
    with pd.option_context("display.max_columns", None):
        print(stats)


@click.command()
@click.option(
    "--input_json",
    default="data/processed/qmsum-generic/documents.json",
    help="Path to the input JSON file.",
)
@click.option(
    "--output_path",
    default="output/qmsum-generic/Meta-Llama-3.1-8B-Instruct/",
    help="Path to output directory.",
)
@click.option(
    "--model",
    default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    help="Model name to be used.",
)
@click.option(
    "--engine", default="vllm", help="Inference engine to use (vllm|litellm)."
)
@click.option("--temperature", default=0.3, type=float, help="Sampling temperature.")
@click.option("--n_samples", default=5, type=int, help="Number of output samples.")
@click.option("--prompt_name", default="generic", help="Name of the prompt to use.")
@click.option(
    "--debug",
    is_flag=True,
    show_default=True,
    default=False,
    help="Debug mode. Only process 5 documents.",
)
def main(
    input_json, output_path, model, engine, temperature, n_samples, prompt_name, debug
):
    pprint(locals())

    if temperature == 0 and n_samples > 0:
        n_samples = 1
        warnings.warn(
            "When temperature == 0, n_samples should be 1. Setting n_samples=1 now."
        )

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    out_files = [
        output_path / f"temperature{temperature}-{i}.json" for i in range(n_samples)
    ]
    if all(f.exists() for f in out_files):
        print(f"All generations already exist. Skip.\n{out_files}")
        return

    dataset = Path(input_json).parent.name
    if engine == "vllm":
        import torch

        if (dataset == "astro-ph") and (
            model == "meta-llama/Meta-Llama-3.1-70B-Instruct"
        ):
            # To fix the following Llama 3.1 70B (128k context window). ValueError: The model's max seq len (131072) is larger than the maximum number of tokens that can be stored in KV cache (24480). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
            max_model_len = 32768
        else:
            max_model_len = None

        llm = VLLMGenerator(
            model,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=max_model_len,
        )
        llm_generate = partial(llm.generate, schema=SummarizationOutput)
    else:
        llm = LitellmGenerator(
            model,
            caching=True,
            report_costs=True,
            disk_cache_dir=f".litellm_cache/summarization-{dataset}-{model}",
        )
        llm_generate = llm.generate

    df = pd.read_json(input_json)
    if debug:
        df = df.head(5)
    lengths = [10, 20, 50, 100, 200]

    all_summaries = [defaultdict(list) for _ in range(n_samples)]
    for length_target in lengths:
        print(f"Generate summaries for length: {length_target}", flush=True)
        messages = build_messages(df, prompt_name, length_target)
        responses = llm_generate(
            messages=messages, temperature=temperature, max_tokens=1024, n=n_samples
        )

        for response in responses:
            for i in range(n_samples):
                try:
                    summary = parse_response(response[i], key="summary")
                except IndexError:
                    # Occurs when the prompt is too long for context window of LLM, in which case there is only an empty list with one empty string ['']. We set all summaries to None for this generation.
                    summary = None
                all_summaries[i][f"summary_{length_target}w"].append(summary)

    for i, summaries in enumerate(all_summaries):
        print(f"Sample: {i}")
        stats(summaries)

        df_out = df.copy()
        for length_target, summaries_for_length in summaries.items():
            df_out[length_target] = summaries_for_length
        df_out.to_json(out_files[i], orient="records")


if __name__ == "__main__":
    # override=True because litellm sets OPEN_API_KEY='' on import, and dotenv does not touch variables which already have a value
    load_dotenv(find_dotenv(), override=True)
    main()
