import argparse
from pathlib import Path

import pandas as pd
from minicheck.minicheck import MiniCheck


def main(args):
    facts_path = Path(args.facts_path)
    if facts_path.name == "facts.json":
        out_dir = "nli"
    elif facts_path.name == "discord_facts.json":
        out_dir = "discord-qa-nli"
    else:
        raise ValueError("Unknown facts type.")

    summaries_path = Path(args.summaries_path)
    out_file = summaries_path.parent.parent / out_dir / summaries_path.name
    out_file.parent.mkdir(exist_ok=True, parents=True)
    if out_file.exists():
        print(f"Already exist: {out_file}. Skip.")
        return

    # df_facts:     | doc_id | sent_id | fact |
    # df_summaries: | doc_id | summary_10w | summary_20w | ... |
    df_facts = pd.read_json(args.facts_path)
    df_summaries = pd.read_json(args.summaries_path)

    df = pd.merge(df_facts, df_summaries, on="doc_id", how="left", validate="m:1")
    assert len(df_facts) == len(df)

    # other minicheck models: ['roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B']
    # https://github.com/Liyan06/MiniCheck/tree/main
    scorer = MiniCheck(model_name="Bespoke-MiniCheck-7B", enable_prefix_caching=True)

    # get the summary columns (summary_10w, summary_20w, ...)
    summary_cols = [c for c in df.columns if c.startswith("summary_") and not c.startswith('summary_e')]
    print(summary_cols)

    # new frame for storing only the classification labels
    if out_dir == "nli":
        result_cols = ["doc_id", "sent_id", "fact"]
    elif out_dir == "discord-qa-nli":
        result_cols = ["doc_id", "cluster_id", "question", "sent_id", "sent", "fact"]
    df_out = df[result_cols].copy()

    for target_length in summary_cols:
        docs = df[target_length].fillna("").tolist()
        claims = df["fact"].tolist()
        labels, probas, _, _ = scorer.score(docs=docs, claims=claims)
        df_out[target_length + "_nli_pred"] = labels
        # df_out[target_length + "_nli_proba"] = probas

        # when premise is nan, reset the NLI label to nan.
        na_mask = df[target_length].isna()
        df_out.loc[na_mask, target_length + "_nli_pred"] = None
        # df_out.loc[na_mask, target_length + "_nli_proba"] = None

    df_out.to_json(out_file, orient="records")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--facts_path",
        type=str,
        required=True,
        help="JSON with splitted facts.",
    )
    parser.add_argument(
        "--summaries_path",
        type=str,
        required=True,
        help="JSON with generated summaries.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())
