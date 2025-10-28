import requests
import json
import tqdm
import os

DOWNJSON = False
DOWNIMG = True

if DOWNJSON:
    resp = requests.get("https://cards.fabtcg.com/api/search/v1/cards/")
    count = resp.json()['count']

    pagesize = 100
    assert pagesize <= 1000

    results = []
    for i in tqdm.tqdm(range(0, count, pagesize)):
        resp = requests.get(f"https://cards.fabtcg.com/api/search/v1/cards/?offset={i}&limit=pagesize")
        resp = resp.json()
        results += resp['results']
        assert not resp['errors']

    with open('cards.json', 'w') as file:
        json.dump(results, file)

if DOWNIMG:
    with open("cards.json", 'r') as file:
        parsed = json.load(file)

    assert os.path.exists("images")

    for i, data in enumerate(tqdm.tqdm(parsed)):
        url = data['image']['large']
        filename = url.split('/')[-1]
        filepath = f"images/{filename}"

        assert not os.path.exists(filepath)
        with open(filepath, 'wb') as f:
            f.write(requests.get(url).content)

        parsed[i]['filename'] = filename

    with open("cards.json", 'w') as file:
        json.dump(parsed, file)
