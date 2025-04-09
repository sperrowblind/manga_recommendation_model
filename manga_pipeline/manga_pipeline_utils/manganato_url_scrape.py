import re
import os
from bs4 import BeautifulSoup
from difflib import SequenceMatcher



CACHE_FILE = 'manganato_url_cache.txt'

def clean_title(title):
    """Clean title string by removing non-alphanumeric characters."""
    return re.sub(r'\W+', ' ', title).strip()

def reformat_url_name(title):
    """Return URL name for given title."""
    return title.replace(' ', '_').lower().strip()

def load_cache():
    """Load the cache from file."""
    cache = {}
    if os.path.isfile(CACHE_FILE):
        with open(CACHE_FILE, 'r') as file:
            for line in file:
                try:
                    title, result = line.strip().split(':', maxsplit=1)
                    cache[title] = result
                except ValueError:
                    continue
    return cache

def save_cache(cache):
    """Save the cache to file."""
    with open(CACHE_FILE, 'w') as file:
        for title, result in cache.items():
            file.write(f'{title}:{result}\n')

async def scrape_manganato_url(session, title, cache):
    """Scrape Manganato URL for given title."""
    if title in cache:
        return cache[title]

    url = f'https://natomanga.com/search/story/{reformat_url_name(title)}'
    async with session.get(url) as response:
        text = await response.text()
        soup = BeautifulSoup(text, 'html.parser')
        links = soup.find_all(class_='item-img bookmark_check', href=True)
        title = title.replace('_', ' ').strip()
        for link in links:
            if len(links) == 1 or SequenceMatcher(a=link['title'].lower(), b=title).ratio() >= .75:
                result = link['href']
                cache[title] = result
                save_cache(cache)
                return result

    return None

