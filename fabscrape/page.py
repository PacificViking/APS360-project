import pprint
import requests
from bs4 import BeautifulSoup
import sys

def full_link(i):
    if not i.startswith('/'):
        i = '/' + i
    return "https://cards.fabtcg.com" + i

def list_url(i):
    assert i <= 41
    return f"https://cards.fabtcg.com/results/?offset={i*100}&limit=100"

def scrapelist(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    return set([i['href'] for i in soup.select("a.search-result__link")])

#print(scrapelist(list_url(0)))

def enumeratepages(soup):
    # these have duplicates, just enumerate
    return set([i['href'] for i in soup.select(".card-details__image-icons > a")] + [i.text for i in soup.select(".details__variants > div > div > div[data-component-variant-link]")])

def scrapepage(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    print(enumeratepages(soup))
    results = {}

    results['url'] = url[url.find("cards.fabtcg.com"):].replace("cards.fabtcg.com", "")
    if results['url'].endswith('/'):
        results['url'] = results['url'][:-1]

    results["name-pitch"] = [i for i in results['url'].split('/') if i][1]

    printsoup = soup.select('.card-details-data--print')[0]

    results["title"] = printsoup.select(".card-details-data__title-text > div > p")[0].text
    results["type"] = printsoup.select(".card-details-data__footer-text > div > p")[0].text

    for i in printsoup.select("span.sr-only"):
        text = i.text.lower().strip()[:-1]
        if text == "cost":
            value = i.next_sibling.next_sibling.text
        else:
            value = i.next_sibling.text
        results[text] = value

    results["text"] = printsoup.select(".card-details-data__blurb")[0].encode_contents().strip()

    label = soup.select(".card-details__production-details-wrapper > p")[0]
    results["set"], results["set_id"], results["rarity"], results["language"] = [i.strip() for i in label.text.split("•")]
    results["artist"] = soup.select(".card-details__production-details-wrapper > p > a")[0].text

    results["set_name"] = soup.select(".card-details__meta-title > h2 > a")[0].text

    #assert results["language"] == "EN"
    
    results["img"] = soup.select(".card-details__face > img")[0]['src']

    return results

#pprint.pp(scrapepage("https://cards.fabtcg.com/card/song-of-sinew-2/SUP134/"))
#pprint.pp(scrapepage("https://cards.fabtcg.com/card/ancestral-empowerment-1/WTR082/"))  # these urls contain pitch value, theres a different page for every pitch value
#pprint.pp(scrapepage("https://cards.fabtcg.com/card/zipper-hit-3/ARC031/"))
#pprint.pp(scrapepage("https://cards.fabtcg.com/card/zipper-hit-3/ES_1HD027/"))
