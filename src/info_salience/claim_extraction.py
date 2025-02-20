import ast
import json
from pathlib import Path
from typing import List

import click
import pandas as pd
import pysbd
from json_repair import repair_json

from info_salience.llm import VLLMGenerator


USER_PROMPT = """
You split sentences into a list of facts that we explicitly know from the sentence. Make each fact as atomic as possible.

Sentence: Protein-rich nutrition is necessary for wound healing after surgery.
Output:
[
    "Protein-rich nutrition is necessary for wound healing.",
    "Wound healing occurs after surgery."
]

Sentence: In this study, the benefit of preoperative nutritional support was investigated for non-small cell lung cancer patients who underwent anatomic resection.
Output:
[
    "The study investigated the benefit of preoperative nutritional support.",
    "The study considers patients with non-small cell lung cancer.",
    "The study considers patients who underwent anatomic resection."
]

Sentence: A prospective study was planned with the approval of our institutional review board.
Output:
[
    "A prospective study was planned.",
    "The study was approved by our institutional review board."
]

Sentence: Fifty-eight patients who underwent anatomic resection in our department between January 2014 and December 2014 were randomized.
Output:
[
    "Fifty-eight patients underwent anatomic resection.",
    "The anatomic resections took place in our department.",
    "The anatomic resections took place between January 2014 and December 2014.",
    "The patients were randomized."
]

Sentence: Thirty-one patients were applied a preoperative nutrition program with immune modulating formulae (enriched with arginine, omega-3 fatty acids and nucleotides) for ten days.
Output:
[
    "Thirty-one patients were applied a preoperative nutrition program.",
    "The preoperative nutrition program lasted for ten days.",
    "The preoperative nutrition program used immune modulating formulae.",
    "The immune modulating formulae were enriched with arginine.",
    "The immune modulating formulae were enriched with omega-3 fatty acids.",
    "The immune modulating formulae were enriched with nucleotides."
]

Sentence: There were 27 patients in the control group who were fed with only normal diet.
Output:
[
    "There were 27 patients in the control group.",
    "The control group was fed with only normal diet."
]

Sentence: Patients who were malnourished, diabetic or who had undergone bronchoplastic procedures or neoadjuvant therapy were excluded from the study.
Output:
[
    "Patients who were malnourished were excluded from the study.",
    "Patients who were diabetic were excluded from the study.",
    "Patients who had undergone bronchoplastic procedures were excluded from the study.",
    "Patients who had undergone neoadjuvant therapy were excluded from the study."
]

Sentence: Patients’ baseline serum albumin levels, defined as the serum albumin level before any nutrition program, and the serum albumin levels on the postoperative third day were calculated and recorded with the other data.
Output:
[
    "Patients have a baseline serum albumin level.",
    "The baseline serum albumin level is defined as the serum level before any nutrition program.",
    "Patients have a serum albumin level on the postoperative third day.",
    "Serum albumin levels were calculated.",
    "Serum albumin levels were recorded with the other data."
]

Sentence: Anatomic resection was performed by thoracotomy in 20 patients, and 11 patients were operated by videothoracoscopy in the nutrition program group.
Output:
[
    "Anatomic resection was performed by thoracotomy in 20 patients in the nutrition program group.",
    "Anatomic resection was performed by videothoracoscopy in 11 patients in the nutrition program group."
]

Sentence: On the other hand 16 patients were operated by thoracotomy and 11 patients were operated by videothoracoscopy in the control group.
Output:
[
    "16 patients were operated by thoracotomy in the control group.",
    "11 patients were operated by videothoracoscopy in the control group."
]

Sentence: In the control group, the patients’ albumin levels decreased to 25.71 % of the baseline on the postoperative third day, but this reduction was only 14.69 % for nutrition program group patients and the difference was statistically significant (p ≺ 0.001).
Output:
[
    "There is a control group and a nutrition program group.",
    "The patients' albumin levels were measured at baseline.",
    "The patients' albumin levels were measured on the postoperative third day.",
    "The patients' albumin levels of the control group decreased to 25.71% of the baseline.",
    "The patients' albumin levels of the nutrition program group decreased to 14.69% of the baseline.",
    "The difference in albumin level reduction between the control group and the nutrition program group was statistically significant (p < 0.001)."
]

Sentence: Complications developed in 12 patients (44.4 %) in the control group compared to 6 patients in the nutrition group (p = 0.049).
Output:
[
    "Complications developed in 12 patients (44.4 %) in the control group.",
    "Complications developed in 6 patients in the nutrition group.",
    "The difference in complication rates between the control group and nutrition groups was statistically significant (p = 0.049)."
]

Sentence: The mean chest tube drainage time was 6 (1-42) days in the control group against 4 (2-15) days for the nutrition program group (p = 0.019).
Output:
[
    "The mean chest tube drainage time was 6 (1-42) days in the control group.",
    "The mean chest tube drainage time was 4 (2-15) days for the nutrition program group.",
    "The difference in mean chest tube drainage time between the control and nutrition groups was statistically,significant (p = 0.019)."
]

Sentence: Our study showed that preoperative nutrition is beneficial in decreasing the complications and chest tube removal time in non-small cell lung cancer patients that were applied anatomic resection with a reduction of 25% in the postoperative albumin levels of non-malnourished patients who underwent resection.
Output:
[
    "Preoperative nutrition is beneficial in decreasing complications.",
    "Preoperative nutrition is beneficial in decreasing chest tube removal time.",
    "The study considered patients with non-small lung cancer.",
    "The study considered patients that were applied anatomic resection.",
    "The study considered non-malnourished patients.",
    "There was a reduction of 25% in the postoperative albumin levels of patients."
]

Here is a new sentence. Please split it into a list of facts that we explicitly know from the sentence. Make each fact as atomic as possible. Output the facts as Python list. Only output the list, nothing more.

Sentence: {sent}
Output:
""".strip()


def get_messages(sent: str):
    user_prompt = USER_PROMPT.format(sent=sent)
    return [
        {"role": "user", "content": user_prompt},
    ]


def parse_response(response):
    try:
        response_fixed = repair_json(response)
        items = ast.literal_eval(response_fixed)
        assert isinstance(items, list)
        assert all(isinstance(item, str) for item in items)
    except (SyntaxError, AssertionError):
        print("=" * 10, "failed to parse:", "=" * 10, "\n", response)
        items = response

    return items


def extract_facts(sents: List[str], llm):
    messages = [get_messages(sent) for sent in sents]
    responses = llm.generate(messages, temperature=0, max_tokens=1024)
    outputs = [parse_response(r[0]) for r in responses]
    return outputs


def extract_facts_from_texts(texts: List[str], llm):
    text_ids = []
    sent_ids = []
    all_sents = []

    seg = pysbd.Segmenter(language="en", clean=False)
    for i, text in enumerate(texts):
        sents = seg.segment(text)
        for sent_id, sent in enumerate(sents):
            text_ids.append(i)
            sent_ids.append(sent_id)
            all_sents.append(sent)

    facts = extract_facts(all_sents, llm)

    data = []
    for i in range(len(all_sents)):
        for fact in facts[i]:
            data.append((text_ids[i], sent_ids[i], all_sents[i], fact))

    return data


@click.command()
@click.option(
    "--input_json",
    required=True,
    type=str,
    help="Path to the input documents",
)
@click.option(
    "--output_json", required=True, type=str, help="Path for storing outputs."
)
def main(input_json, output_json):
    with open(input_json) as fin:
        docs = json.load(fin)

    doc_ids = [doc["doc_id"] for doc in docs]
    texts = [doc["text"] for doc in docs]

    llm = VLLMGenerator("meta-llama/Meta-Llama-3.1-8B-Instruct")
    fact_data = extract_facts_from_texts(texts, llm)
    text_ids, sent_ids, sents, facts = zip(*fact_data)
    df = pd.DataFrame(
        {
            "doc_id": [doc_ids[i] for i in text_ids],
            "sent_id": sent_ids,
            "sent": sents,
            "fact": facts,
        }
    )

    Path(output_json).parent.mkdir(exist_ok=True, parents=True)
    df.to_json(output_json, orient="records")


if __name__ == "__main__":
    main()
