import json
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm

XML_PATH = Path("data/raw/pubmed/xml/")
OUT_PATH = Path("data/raw/pubmed/articles.json")
QUERY = """
("randomized controlled trial"[Publication Type])
AND (Humans[MeSH Terms])
AND (2024/01/01:2024/12/31[Date - Publication])
AND (2024/01/01:2024/12/31[Date - Entry])
AND (english[la])'
""".strip()


def search_articles(query, retstart=0, retmax=100):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retstart": retstart,
        "retmax": retmax,
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    ids = data["esearchresult"]["idlist"]
    total = int(data["esearchresult"]["count"])
    return ids, total


def fetch_articles(ids):
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
    fetch_response = requests.get(fetch_url, params=fetch_params)
    fetch_data = fetch_response.text
    return fetch_data


def parse(xml):
    root = ET.fromstring(xml)
    articles = []

    for article in root.findall("PubmedArticle"):
        d = {}
        d["pmid"] = article.find("MedlineCitation/PMID").text
        d["title"] = "".join(
            article.find("MedlineCitation/Article/ArticleTitle").itertext()
        )
        date = article.find("PubmedData/History/PubMedPubDate[@PubStatus = 'entrez']")
        date = f"{date.find('Year').text}/{date.find('Month').text}/{date.find('Day').text}"
        d["date"] = date

        # Abstracts can have multiple sections
        sections = []
        for section in article.findall("MedlineCitation/Article/Abstract/AbstractText"):
            label = section.get("Label")
            if label:
                s = {"label": label.upper(), "text": "".join(section.itertext())}
                sections.append(s)
            else:
                current_label = ""
                current_text = ""

                for node in section.iter():
                    if node.tag == "b":
                        if current_label or current_text:
                            sections.append(
                                {"label": current_label, "text": current_text}
                            )

                        current_label = (
                            "".join(node.itertext()).strip().removesuffix(":").upper()
                        )
                        current_text = node.tail.removeprefix(": ") if node.tail else ""
                    else:
                        current_text += node.text if node.text else ""
                        current_text += node.tail if node.tail else ""

                sections.append({"label": current_label, "text": current_text})
        d["abstract"] = sections

        abstract_str = ""
        for section in sections:
            if section["label"]:
                abstract_str += section["label"] + ":\n"
            abstract_str += section["text"] + "\n\n"
        d["abstract_str"] = abstract_str.strip()

        # Plain language summary (if present)
        pls = None
        for node in article.findall("MedlineCitation/OtherAbstract"):
            if node.get("Type") == "plain-language-summary":
                pls = "".join(node.itertext())
        d["pls"] = pls

        articles.append(d)
    return articles


def download():
    retmax = 100
    _, total = search_articles(QUERY)
    num_pages = (total + retmax - 1) // retmax

    data_files = []
    for page in tqdm(range(num_pages), total=num_pages, desc="Fetch articles"):
        retstart = page * retmax
        ids, _ = search_articles(QUERY, retstart=retstart, retmax=retmax)
        data = fetch_articles(ids)
        out_path = XML_PATH / f"p{page:02}.xml"
        with open(out_path, "w") as fout:
            fout.write(data)
            data_files.append(out_path)
    return data_files


def main():
    data_files = sorted(list(XML_PATH.glob("*.xml")))
    if not data_files:
        data_files = download()

    articles = []
    for f in data_files:
        with open(f) as fin:
            xml = fin.read()
        articles += parse(xml)

    filtered = [
        a
        for a in articles
        if len(a["abstract_str"].split()) >= 100 and a["pls"] is None
    ]
    print(f"Total articles: {len(articles)}")
    print(f"Filtered: {len(filtered)}")

    # drop obsolete field
    for a in articles:
        a.pop("pls")

    with open(OUT_PATH, "w") as fout:
        json.dump(filtered, fout)


if __name__ == "__main__":
    main()
