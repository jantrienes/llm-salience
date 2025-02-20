from pathlib import Path
import pandas as pd
import json

# Note: set this to CSV from google forms with all submissions
raw_path = 'data-submission.csv'
out_path = Path('data/annotations/human-salience/')
out_path.mkdir(exist_ok=True)

df = pd.read_csv(raw_path)
for index, row in df.iterrows():
    data = row['Response data (json)']
    data = json.loads(data)

    data['duration'] = row['Duration']
    dataset = data['dataset']
    annotator = data['annotator']

    out_file = out_path / f'{dataset}-{annotator}.json'
    print(out_file)
    with open(out_file, 'w') as fout:
        json.dump(data, fout, indent=4)
