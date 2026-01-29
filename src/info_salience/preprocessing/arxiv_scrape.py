"""
Fetch and store arXiv source bundle.

Code credits: Sebastian Joseph (UT Austin).
"""

from urllib.request import urlopen

import requests

import cgi
import os

import time

import json

ex_recs = json.load(open('arxiv_ids.example.json'))

cat = "cs.CL"

out_dir = f"records_{cat}"

url_lst = ex_recs[:500]

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for i, url in enumerate(url_lst):

    end = url.split('/')[-1]

    end_dash = end.replace(".", "-")

    check_list = [l for l in os.listdir(out_dir) if end in l or end_dash in l]


    if len(check_list) > 0:
        continue

    print(url)
    print(i)

    export_url = url

    try:
        res = urlopen(export_url)
        _, params = cgi.parse_header(res.headers.get('Content-Disposition', ''))
        fname = params['filename']
    except Exception as err:
        print(err)
        print("NOT FOUND")
        continue

    if fname in os.listdir(out_dir):
        print(fname)
        time.sleep(0.7)
        continue

    time.sleep(1)

    response = requests.get(export_url, params={"downloadFormat": "tar"})

    print(response.ok)

    print(response.status_code)

    with open(os.path.join(out_dir, fname), "wb") as f:
        f.write(response.content)


    time.sleep(1)
