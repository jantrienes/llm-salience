import re
from pathlib import Path

import pandas as pd
import tiktoken


def detokenize(text):
    # removes spaces before punctuation and replaces multiple consecutive whitespace with single space
    # does not handle quotation marks
    cleaned_text = re.sub(r"\s([?.!,\'](?:\s|$))", r"\1", text)
    cleaned_text = re.sub(r"\s+\'", "'", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text


def format_meeting(dialogue_turns):
    text = ""
    for turn in dialogue_turns:
        text += turn["speaker"].upper() + "\n"
        text += detokenize(turn["content"]) + "\n"
        text += "\n"
    text = text.strip()
    return text


def format_speakers(dialogue_turns):
    speakers = set(turn["speaker"] for turn in dialogue_turns)
    speakers = sorted(list(speakers))
    return ", ".join(speakers)


def format_text(speakers, transcript):
    return f"Speakers: {speakers}\n\n{transcript}"


def load_data(base_path="data/raw/qmsum/"):
    dfs = []
    for domain in ["Academic", "Committee", "Product"]:
        for split in ["train", "val", "test"]:
            df = pd.read_json(f"{base_path}/{domain}/jsonl/{split}.jsonl", lines=True)
            df["domain"] = domain
            df["split"] = split
            df["doc_id"] = domain + "-" + split + "-" + df.index.astype(str)
            dfs.append(df)

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df["speakers"] = df["meeting_transcripts"].apply(lambda x: format_speakers(x))
    df["text"] = df.apply(
        lambda row: format_text(
            row["speakers"], format_meeting(row["meeting_transcripts"])
        ),
        axis=1,
    )
    return df


def main():
    # Remove meeting transcripts which are longer than 30,000 tokens according to GPT-4 tokenizer.
    enc = tiktoken.get_encoding("o200k_base")
    df = load_data()
    lengths = df["text"].apply(lambda x: len(enc.encode(x)))
    df = df[lengths <= 30000]

    # Take a random sample of 30 meetings from each of the three domains (academia, committee, product)
    df = df.groupby("domain").sample(30, random_state=42)

    # Export metadata
    out_file = Path("data/processed/qmsum-generic/metadata.json")
    out_file.parent.mkdir(exist_ok=True, parents=True)
    cols = [
        "doc_id",
        "domain",
        "split",
        "speakers",
        "topic_list",
        "general_query_list",
        "specific_query_list",
        "meeting_transcripts",
    ]
    df[cols].to_json(out_file, orient="records")

    # Export for generic summarization
    out_file = Path("data/processed/qmsum-generic/documents.json")
    out_file.parent.mkdir(exist_ok=True, parents=True)
    df_generic = df[["doc_id", "text"]]
    df_generic.to_json(out_file, orient="records")


if __name__ == "__main__":
    main()
