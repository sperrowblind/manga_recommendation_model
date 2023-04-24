import requests as re
from bs4 import BeautifulSoup

def find_latest_manga_recommendations(limit):
    manganato_url = 'https://manganato.com/'
    text = re.get(manganato_url).content
    soup = BeautifulSoup(text, 'html.parser')
    titles = [title.text.strip() for title in soup.find_all(class_='item-title')]
    if limit == 40:
        return titles
    return titles[0:limit]

