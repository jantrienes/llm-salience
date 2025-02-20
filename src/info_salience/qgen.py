import json
from typing import List
from json_repair import repair_json
import outlines

TOPICS = {
    "pubmed": "Randomized controlled trials (RCT) in the clinical domain.",
    "astro-ph": "Discussion section in astrophysics papers.",
    "cs-cl": "Related work section in NLP papers.",
    "qmsum": "Meeting transcripts.",
}


@outlines.prompt
def build_prompt(documents, genre):
    """Your task is to analyze summaries of different lengths within a given genre. Your goal is to create question-answer pairs that capture the essence of information typically included in various summary lengths. Below is the dataset where each document was summarized in 5 different lengths.

    # Dataset

    {% for document in documents %}
    ## Document {{ loop.index }}

    {% for length, text in document.items() %}
    ### Summary {{ length }}
    {{ text }}

    {% endfor %}
    {% endfor %}

    # Genre
    The genre of the documents: {{ genre }}

    # Task
    Each text in the dataset has been summarized in 5 different lengths (in 10, 20, 50, 100, and 200 words). Your task is to analyze the summaries and identify the types of information typically included at each summary length. To do this, please proceed as follows:

    1. Carefully read the summaries, paying attion to what information is included or omitted.
    2. For each summary length, create a set of question-answer pairs that represent typical information included at this length. The questions should be general enough to apply to many documents in this genre, while the answers will naturally be different across documents.

    Important guidelines:
    - Ensure that your questions are relevant to the genre and capture information that would be commonly found in texts of this type.
    - It is really important that the questions are answerable with most documents in this genre, not just with the ones presented here! To this end, state a prototypical answer to each question.
    - The questions should be unique to each length. That means, do not repeat a question if it is already sufficiently covered at the shorter length.
    - Start with question words (What, How, Why, Which) rather than 'Can you'
    - Make the topic the grammatical subject of a question.
    - Keep the questions concise and focused.
    - Create at least 3-5 questions for each summary length.

    Structure your response as a valid json object with the following format:

    {
        "questions_10_words": [
            {
                "question": "",
                "example_answer": "",
            }
        ],
        "questions_20_words": [
            {
                "question": "",
                "example_answer": "",
            }
        ],
        "questions_50_words": [
            {
                "question": "",
                "example_answer": "",
            }
        ],
        "questions_100_words": [
            {
                "question": "",
                "example_answer": "",
            }
        ],
        "questions_200_words": [
            {
                "question": "",
                "example_answer": "",
            }
        ]
    }
    """


def parse_response(response):
    try:
        response_fixed = repair_json(response)
        result = json.loads(response_fixed)
    except SyntaxError:
        print("=" * 10, "failed to parse:", "=" * 10, "\n", response)
        result = {
            "questions_10_words": [],
            "questions_20_words": [],
            "questions_50_words": [],
            "questions_100_words": [],
            "questions_200_words": [],
        }

    return result


def get_prompts(df_summaries, topic, batch_size):
    docs = []
    for index, row in df_summaries.iterrows():
        docs.append(
            {
                f"{length} words": row[f"summary_{length}w"]
                for length in [10, 20, 50, 100, 200]
            }
        )

    prompts = []
    for i in range(0, len(docs), batch_size):
        prompt = build_prompt(docs[i : i + batch_size], TOPICS[topic])
        messages = [{"role": "user", "content": prompt}]
        prompts.append(messages)
    return prompts


def generate_questions(
    llm, df_summaries, topic, batch_size=5, debug=False
) -> List[str]:
    prompts = get_prompts(df_summaries, topic, batch_size)
    responses = llm.generate(prompts)
    responses = [parse_response(response[0]) for response in responses]

    if debug:
        for response in responses:
            print("=" * 30)
            for k, v in response.items():
                print(k)
                for q in v:
                    print("Q:", q["question"])
                    # print('A:', q['example_answer'])
                print()

    questions = []
    for response in responses:
        for length, qa_pairs in response.items():
            for qa_pair in qa_pairs:
                questions.append(qa_pair["question"])

    return questions
