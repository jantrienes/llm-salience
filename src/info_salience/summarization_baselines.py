import dataclasses
import json
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from summa.summarizer import summarize as textrank
from tqdm import tqdm

from info_salience.summarization import stats


@dataclasses.dataclass
class Sentence:
    index: int
    text: str
    score: Optional[float] = None

    def __len__(self):
        return len(word_tokenize(self.text))


def get_sentences(text):
    return [Sentence(i, sent) for i, sent in enumerate(sent_tokenize(text))]


def join_sentences(sentences):
    return " ".join(sent.text for sent in sentences)


def select_sentences_with_budget(sentences, words):
    """Given a list of sentences, returns a list of sentences with a total word count similar to the word count provided.

    Adapted from: https://github.com/summanlp/textrank/blob/d9252a233c93ec43693e0f145a025ae534b275b1/summa/summarizer.py#L77-L95
    """
    word_count = 0
    selected_sentences = []
    # Loops until the word count is reached.
    for sentence in sentences:
        words_in_sentence = len(sentence)

        # Checks if the inclusion of the sentence gives a better approximation
        # to the word parameter.
        if abs(words - word_count - words_in_sentence) > abs(words - word_count):
            return selected_sentences

        selected_sentences.append(sentence)
        word_count += words_in_sentence

    return selected_sentences


def summarize_lead(text, budget):
    sentences = get_sentences(text)
    sentences = sentences[:budget]
    return join_sentences(sentences)


def summarize_lead_1(text, budget):
    # ignoring budget parameter to have common signature with other functions
    return summarize_lead(text, budget=1)


def summarize_lead_words(text, budget):
    sentences = get_sentences(text)
    summary = select_sentences_with_budget(sentences, budget)
    if not summary:
        summary = sentences[:1]
    return join_sentences(summary)


def summarize_random(text, budget, rng=np.random.default_rng()):
    sentences = get_sentences(text)
    rng.shuffle(sentences)
    summary = select_sentences_with_budget(sentences, budget)
    if not summary:
        summary = sentences[:1]
    summary = sorted(summary, key=lambda sent: sent.index)
    return join_sentences(summary)


def summarize_greedy(text, budget):
    sentences = get_sentences(text)
    sentences = sorted(sentences, key=lambda sent: len(sent))
    summary = select_sentences_with_budget(sentences, budget)
    if not summary:
        summary = sentences[:1]
    summary = sorted(summary, key=lambda sent: sent.index)
    return join_sentences(summary)


def summarize_textrank(text, budget):
    scored_sentences = textrank(text, ratio=1, scores=True)
    sentences = [
        Sentence(i, sent, score) for i, (sent, score) in enumerate(scored_sentences)
    ]
    sentences = sorted(sentences, key=lambda sent: sent.score, reverse=True)
    summary = select_sentences_with_budget(sentences, budget)
    if not summary:
        summary = sentences[:1]
    summary = sorted(summary, key=lambda sent: sent.index)
    return join_sentences(summary)


@click.command()
@click.option(
    "--documents_json",
    help="Path to documents.",
    required=True,
)
@click.option(
    "--output_path",
    help="Path to store outputs in (separate directories will be created here).",
    required=True,
)
def main(documents_json, output_path):
    with open(documents_json) as fin:
        docs = json.load(fin)

    ids = [doc["doc_id"] for doc in docs]
    texts = [doc["text"] for doc in docs]
    lengths = [10, 20, 50, 100, 200]

    def summarize_all(texts, summarize_f, out_file, **kwargs):
        summaries = {}
        for length in lengths:
            summaries[f"summary_{length}w"] = [
                summarize_f(text, budget=length, **kwargs)
                for text in tqdm(texts, desc=summarize_f.__name__)
            ]

        print("=" * 80)
        print(summarize_f.__name__, repr(dict(**kwargs)))
        print("=" * 80)
        stats(summaries)

        out_file.parent.mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame({"doc_id": ids, "text": texts, **summaries})
        df.to_json(out_file, orient="records")

    summarize_all(
        texts,
        summarize_lead_1,
        out_file=Path(output_path) / "lead_1" / "summaries" / "output.json",
    )

    summarize_all(
        texts,
        summarize_lead_words,
        out_file=Path(output_path) / "lead_n" / "summaries" / "output.json",
    )

    for i in range(5):
        rng = np.random.default_rng(seed=i)
        summarize_all(
            texts,
            summarize_random,
            out_file=Path(output_path) / "random" / "summaries" / f"output-{i}.json",
            rng=rng,
        )

    summarize_all(
        texts,
        summarize_greedy,
        out_file=Path(output_path) / "greedy" / "summaries" / "output.json",
    )

    summarize_all(
        texts,
        summarize_textrank,
        out_file=Path(output_path) / "textrank" / "summaries" / "output.json",
    )


if __name__ == "__main__":
    main()
