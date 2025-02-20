from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("data/raw/pubmed/articles.json")
    df = df.sample(200, random_state=42)
    df = df[["pmid", "abstract_str"]]
    df = df.rename({"pmid": "doc_id", "abstract_str": "text"}, axis=1)

    # Write full dataset
    out_path = Path("data/processed/pubmed-sample/documents.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_json(out_path, orient="records")

    # Write a dummy dataset for debugging purposes
    out_path = Path("data/processed/dummy/documents.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.head(10).to_json(out_path, orient="records", indent=4)
