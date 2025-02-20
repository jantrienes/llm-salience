import itertools
import json
import logging

import click
import outlines
import pandas as pd

from info_salience.claim_extraction import extract_facts_from_texts
from info_salience.llm import VLLMGenerator


@outlines.prompt
def qa_prompt(text, question):
    """
    Answer the following question given the text. If the question cannot be answered with the text, reply "no answer".

    ## Text
    {{ text }}

    ## Question
    {{ question }}

    First, carefully read and analyze both the text and the question. Then provide the answer. Please follow these guidelines:
    - If the question cannot be answered, reply with "no answer"
    - Use only information explicitly stated in or directly implied by the text
    - Do not include any external knowledge or personal opinions
    - Aim for concise answers that include all important points relevant to the question

    Please use this format for your response:
    Question: [restate the question exactly]
    Answer: [the answer based on the text or "no answer"]
    """


def parse_response(response):
    response = response.strip()
    try:
        ix = response.index("Answer:")
        ix = ix + len("Answer:")
        answer = response[ix:].strip()
    except ValueError:
        print("=" * 10, "failed to parse:", "=" * 10, "\n", response)
        answer = None

    return answer


def question_answering(llm, texts, questions):
    prompts = []
    for text, question in zip(texts, questions):
        prompt = [{"role": "user", "content": qa_prompt(text, question)}]
        prompts.append(prompt)
    print("=" * 40, "Example QA Prompt", "=" * 40)
    print(prompts[0][0]["content"])

    responses = llm.generate(prompts, temperature=0.7, min_tokens=2, max_tokens=512)

    answers = []
    for prompt, response in zip(prompts, responses):
        response = response[0]
        try:
            answer = parse_response(response)
        except Exception:
            answer = None
            logging.exception(
                f"Failed to parse the following response:\n {response}\n\n{prompt}"
            )
        answers.append(answer)

    return answers


def is_non_answer(answer):
    if not answer:
        return True
    if answer.lower().startswith("yes"):
        return False
    return any(
        phrase in answer.lower() for phrase in ["the text", "no mention", "no answer"]
    )


@click.command()
@click.option(
    "--documents_json",
    help="Path to documents.",
    required=True,
)
@click.option(
    "--questions_json",
    help="Path to discord questions.",
    required=True,
)
@click.option(
    "--answers_json",
    help="Path to store answers at.",
    required=True,
)
@click.option(
    "--answer_facts_json",
    help="Path to store answer facts at.",
    required=True,
)
def main(documents_json, questions_json, answers_json, answer_facts_json):
    with open(documents_json) as fin:
        documents = json.load(fin)

    with open(questions_json) as fin:
        questions = json.load(fin)

    llm = VLLMGenerator("meta-llama/Meta-Llama-3.1-8B-Instruct")

    ######################################################
    # Generate answers for discord questions on the source document
    ######################################################
    print("Generate answers for discord questions")
    data = [
        (
            doc["doc_id"],
            doc["text"],
            question["centroid"],
            question["cluster_id"],
        )
        for doc, question in itertools.product(documents, questions)
    ]
    doc_ids, texts, qs, q_ids = zip(*data)

    answers = question_answering(llm=llm, texts=texts, questions=qs)
    answers = ["no answer" if is_non_answer(answer) else answer for answer in answers]
    df_answers = pd.DataFrame(
        {
            "doc_id": doc_ids,
            "cluster_id": q_ids,
            "question": qs,
            "reference_answer": answers,
        }
    )
    df_answers.to_json(answers_json, orient="records")

    ######################################################
    # Split answers into a list of atomic claims
    ######################################################
    print("Split each answer sentence into list of atomic claims")
    df_filtered = df_answers[df_answers["reference_answer"] != "no answer"]
    # Tuples of (text_id, sent_id, sent, fact)
    fact_data = extract_facts_from_texts(
        df_filtered["reference_answer"].values, llm=llm
    )
    text_ids, sent_ids, sents, facts = zip(*fact_data)
    df_facts = pd.DataFrame(
        {
            "doc_id": [df_filtered.iloc[i]["doc_id"] for i in text_ids],
            "cluster_id": [df_filtered.iloc[i]["cluster_id"] for i in text_ids],
            "question": [df_filtered.iloc[i]["question"] for i in text_ids],
            "sent_id": sent_ids,
            "sent": sents,
            "fact": facts,
        }
    )
    df_facts.to_json(answer_facts_json, orient="records")


if __name__ == "__main__":
    main()
