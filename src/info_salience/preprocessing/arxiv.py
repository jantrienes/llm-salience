import json
import re
from pathlib import Path

import click
import pandas as pd
import tiktoken


def load_data(base_path="data/raw/astro-ph/"):
    papers = []

    # Get all subdirectories in the dataset path
    paths = sorted(list(Path(base_path).glob("*")))
    paths = [path for path in paths if not path.is_file()]
    print(f"Found {len(paths)} papers.")

    for path in paths:
        paper = {}
        for file_path in path.iterdir():
            if "metadata" in file_path.name:
                with open(file_path) as fin:
                    metadata = json.load(fin)
                    paper = {**paper, **metadata}
            elif "discussion" in file_path.name:
                with open(file_path) as fin:
                    # first two lines are either blank or contain a fragment of the section title,
                    # due to poor parsing: e.g., "and Conclusions" for "# Discussion and Conclusions"
                    lines = fin.readlines()
                    lines = lines[2:]
                    text = "".join(lines)
                    text = text.strip()
                    # Unwrap lines: Replace single newlines with a space, but keep double newlines intact
                    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
                    paper["text"] = text

        paper["doc_id"] = paper.pop("arxiv_full_id")
        papers.append(paper)

    df = pd.DataFrame(papers)
    df = df[["doc_id", "text"]]
    return df


@click.command()
@click.option(
    "--raw_path",
    help="Path to raw arxiv dump.",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--output_json",
    help="Path to processed documents to.",
    required=True,
)
def main(raw_path, output_json):
    df = load_data(raw_path)

    print(f"Print number of papers before length filtering: {len(df)}")
    enc = tiktoken.get_encoding("o200k_base")
    lengths = df["text"].apply(lambda x: len(enc.encode(x)))
    df = df[lengths <= 2000]
    print(f"Print number of papers after length filtering: {len(df)}")

    out_path = Path(output_json)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_json(out_path, orient="records")


if __name__ == "__main__":
    main()
