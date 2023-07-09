import requests as re
from bs4 import BeautifulSoup
from .new_manga_route_constants import MANGANATO_GENRES, MANGANATO_GENRE_BASE_PREFIX, BASE_MANGANATO_URL, MANGANATO_END_URL

def find_latest_manga_recommendations(limit, genre_list):
    if len(genre_list) == 0:
        manganato_url = 'https://manganato.com/'
        text = re.get(manganato_url).content
        soup = BeautifulSoup(text, 'html.parser')
        titles = [title.text.strip() for title in soup.find_all(class_='item-title')]
    else:
        genres = MANGANATO_GENRE_BASE_PREFIX.join(MANGANATO_GENRES.get(genre) for genre in genre_list)
        manganato_url = BASE_MANGANATO_URL + MANGANATO_GENRE_BASE_PREFIX + genres + MANGANATO_END_URL
        text = re.get(manganato_url).content
        soup = BeautifulSoup(text, 'html.parser')
        titles = [title.text.strip() for title in soup.find_all(class_='genres-item-name text-nowrap a-h')]
    if limit == 40:
        return titles
    return titles[0:limit]

