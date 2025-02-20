import json
from pathlib import Path
import traceback
from typing import List

import click
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import outlines
from json_repair import repair_json

from info_salience.llm import LitellmGenerator, VLLMGenerator


@outlines.prompt
def ranking_prompt(task: str, questions: List[str], length_constraint: str):
    """
    ## Task
    {{task}} The summary length is constrained, requiring you to think about what content to prioritize. Ask yourself: what are some key questions you want the summary to answer? Your task is to rate the relative importance of a list of questions that the summary could answer.

    ## Questions
    Here is the list of questions you should evaluate.

    {% for question in questions %}
    {{ loop.index }}. {{ question }}
    {% endfor %}

    ## Rating
    Please use the following scale, going from least important to most important.

    1 - Least important; I would exclude this information from a summary.
    2 - Low importance; I would include this information if there is room.
    3 - Medium importance; I would probably include this information.
    4 - High importance; I would definitely include this information.
    5 - Most important; One of the first questions to be answered in the summary.

    For each rating, please provide a brief (1-sentence) rationale explaining your decision or highlighting any considerations or uncertainties you had.

    Important considerations:
    - Use the full scale (1-5) to express relative importance.
    - {{ length_constraint }}
    - Make sure to rate all given questions.

    Please respond as a valid JSON list with following format:

    [
        {
            "id": "[the question number]",
            "question": "[repeat the exact question]",
            "rationale": "[your one-sentence rationale for the rating]",
            "rating": "[your numeric rating, 1-5]"
        }
    ]
    """


@outlines.prompt
def ranking_prompt_olmo(task: str, questions: List[str], length_constraint: str):
    """
    ## Task
    {{task}} The summary length is constrained, requiring you to think about what content to prioritize. Ask yourself: what are some key questions you want the summary to answer? Your task is to rate the relative importance of a list of questions that the summary could answer.

    ## Questions
    Here is the list of questions you should evaluate.

    {% for question in questions %}
    {{ loop.index }}. {{ question }}
    {% endfor %}

    ## Rating
    Please use the following scale, going from least important to most important.

    1 - Least important; I would exclude this information from a summary.
    2 - Low importance; I would include this information if there is room.
    3 - Medium importance; I would probably include this information.
    4 - High importance; I would definitely include this information.
    5 - Most important; One of the first questions to be answered in the summary.

    Important considerations:
    - Use the full scale (1-5) to express relative importance.
    - {{ length_constraint }}
    - Make sure to rate all given questions.

    Please respond as a valid JSON list with following format:

    [
        {
            "id": "[the question number]",
            "question": "[repeat the exact question]",
            "rating": "[your numeric rating, 1-5]"
        }
    ]
    """


TASKS = {
    "astro-ph": "You are a research expert in astrophysics. Imagine you are asked to summarize the discussion section of an astrophysics paper for a typical reader in this field. The summary should provide enough context to stand alone, since the reader will only see your summary and no other parts of the paper.",
    "cs-cl": "You are a research expert in natural language processing (NLP). Imagine you are asked to summarize the related work section of an NLP paper for a typical reader in this field. The summary should provide enough context to stand alone, since the reader will only see your summary and no other parts of the paper.",
    "pubmed-sample": "You are a research expert in randomized controlled trials (RCTs). Imagine you are asked to summarize a paper describing the results of an RCT for a typical reader in this field. The summary should provide enough context to stand alone, since the reader will only see your summary and no other parts of the paper.",
    "qmsum-generic": "You are an expert in communications and meetings. Imagine you are asked to summarize a meeting transcript (e.g., research group meetings) for a typical reader of these texts. The summary should provide enough context to stand alone, since the reader will only see your summary and not the full meeting transcript.",
}

LENGTH_CONSTRAINTS = {
    "200w": "Remember that space in the summary is limited to exactly 200 words, so not everything can be included, and you CANNOT rate all questions as 5.",
    "100w": "Remember that space in the summary is limited to exactly 100 words, so not everything can be included, and you CANNOT rate all questions as 5.",
    "50w": "Remember that space in the summary is limited to exactly 50 words, so not everything can be included, and you CANNOT rate all questions as 5.",
    "20w": "Remember that space in the summary is limited to exactly 20 words, so not everything can be included, and you CANNOT rate all questions as 5.",
    "10w": "Remember that space in the summary is limited to exactly 10 words, so not everything can be included, and you CANNOT rate all questions as 5.",
    "generic": "Remember that space in the summary is limited, so not everything can be included, and you CANNOT rate all questions as 5.",
}


def load_questions(dataset):
    df = pd.read_json(f"output/{dataset}/discord_questions.json")
    df = df.rename({"centroid": "question"}, axis=1)
    df = df.set_index("cluster_id")
    return df[["question"]]


def rate_questions(llm, df_questions, task, length_constraint, n=5, max_retries=25):
    df = df_questions.copy()

    i = 0
    retries = 0
    while i < n and retries < max_retries:
        print(f"rating {i + 1}/{n} (retry: {retries}/{max_retries})")

        df = df.sample(frac=1)
        questions = df["question"].values

        if llm.model == "allenai/OLMo-7B-Instruct-hf":
            # shorter prompt without rationale to fit within 2k context restriction
            prompt = ranking_prompt_olmo(task, questions, length_constraint)
        else:
            prompt = ranking_prompt(task, questions, length_constraint)

        messages = [{"role": "user", "content": prompt}]
        response = None
        try:
            if llm.engine == "litellm":
                response = llm.generate(
                    [messages], temperature=0.3, max_tokens=2048, timeout=60
                )
            else:
                response = llm.generate([messages], temperature=0.3, max_tokens=2048)

            response = response[0][0]
            rated_questions = repair_json(response, return_objects=True)
            rated_questions = list(rated_questions)
            assert len(rated_questions) == len(questions)

            # some LLMs sort the questions from most important to least important, so we order them by ID
            rated_questions = sorted(
                rated_questions, key=lambda question: int(question["id"])
            )
            ratings = [question["rating"] for question in rated_questions]
            df[f"rating{i}"] = ratings
            df[f"rating{i}"] = df[f"rating{i}"].astype(
                str
            )  # ensure all columns are string
            df[f"rationale{i}"] = [question.get('rationale', '') for question in rated_questions]
            i += 1
        except AssertionError:
            print(f'Length mismatch. Expected: {len(questions)}, Generated: {len(rated_questions)}')
            retries += 1
        except Exception:
            print(response)
            traceback.print_exc()
            retries += 1

    for i in range(n):
        if f"rating{i}" not in df.columns:
            df[f"rating{i}"] = None

    return df.sort_index(), retries


@click.command()
@click.option(
    "--model",
    default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    help="Model name to be used.",
)
@click.option(
    "--engine", default="vllm", help="Inference engine to use (vllm|litellm)."
)
def main(model, engine):
    print(f"Running introspection\nModel: {model}\nEngine: {engine}")

    if engine == "vllm":
        import torch

        if model == "meta-llama/Meta-Llama-3.1-70B-Instruct":
            # To fix the following Llama 3.1 70B (128k context window). ValueError: The model's max seq len (131072) is larger than the maximum number of tokens that can be stored in KV cache (24480). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
            max_model_len = 14000
        else:
            max_model_len = None

        llm = VLLMGenerator(
            model,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=max_model_len,
        )
    else:
        llm = LitellmGenerator(
            model,
            caching=False,
            report_costs=True,
        )

    for dataset in ["pubmed-sample", "astro-ph", "cs-cl", "qmsum-generic"]:
        for length_key, length_constraint in LENGTH_CONSTRAINTS.items():
            # Split model name. Example: meta-llama/Meta-Llama-3.1-8B-Instruct --> Meta-Llama-3.1-8B-Instruct
            m = Path(model).name
            out_file = Path(f"output/{dataset}/{m}/introspection-rationale/{length_key}.json")
            meta_file = Path(f"output/{dataset}/{m}/introspection-rationale/{length_key}.meta.json")
            out_file.parent.mkdir(exist_ok=True, parents=True)

            if out_file.exists():
                print(f"Exists (skip): {str(out_file)}")
                continue
            else:
                print(f"Processing: {dataset} | {length_key}")

            df_questions = load_questions(dataset)
            df_ratings, retries = rate_questions(
                llm,
                df_questions,
                task=TASKS[dataset],
                length_constraint=length_constraint,
                n=5,
            )
            with pd.option_context('display.max_columns', None):
                print(df_ratings)
            df_ratings = df_ratings.reset_index()
            df_ratings.to_json(out_file, orient="records")

            with open(meta_file, 'w') as fout:
                json.dump({'retries': retries}, fout)


if __name__ == "__main__":
    # override=True because litellm sets OPEN_API_KEY='' on import, and dotenv does not touch variables which already have a value
    load_dotenv(find_dotenv(), override=True)
    main()
