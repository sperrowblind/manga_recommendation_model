import re
from bs4 import BeautifulSoup
import urllib.parse
import numpy as np


def parse_page_content(page_content, url):
    manga_info = [url, '', '', '', '', '', '', '']

    soup = BeautifulSoup(page_content, 'html.parser')

    # Parse genre
    genre_element = soup.find('th', text='Genre')
    if genre_element:
        genre_info = genre_element.find_next('td').text.strip()
        manga_info[1] = genre_info

    # Parse publisher
    publisher_element = soup.find('th', text='Published\xa0by')
    if publisher_element:
        publisher_info = publisher_element.find_next('td').text.strip()
        manga_info[2] = publisher_info

    # Parse demographic
    demographic_element = soup.find('th', text='Demographic')
    if demographic_element:
        demographic_info = demographic_element.find_next('td').text.strip()
        manga_info[5] = demographic_info

    # Parse magazine
    magazine_element = soup.find('th', text='Magazine')
    if magazine_element:
        magazine_info = magazine_element.find_next('td').text.strip()
        manga_info[4] = magazine_info

    # Parse volumes
    volumes_element = soup.find('th', text='Volumes')
    if volumes_element:
        volumes_info = volumes_element.find_next('td').text.strip()
        manga_info[7] = volumes_info

    # Parse original run
    original_run_element = soup.find('th', text='Original run')
    if original_run_element:
        original_run_info = original_run_element.find_next('td').text.strip()
        manga_info[6] = original_run_info

    # Parse English publisher
    english_publisher_element = soup.find('th', text='English publisher')
    if english_publisher_element:
        english_publisher_info = english_publisher_element.find_next('td').text.strip()
        manga_info[3] = english_publisher_info

    return manga_info

async def get_page_content(session, page_url):
    base_url = "https://en.wikipedia.org/w/api.php"
    page_title = page_url.split("/")[-1]
    page_title = urllib.parse.unquote(page_title)
    params = {
        "action": "parse",
        "format": "json",
        "page": page_title,
        "prop": "text|displaytitle|iwlinks|categories|templates|images|sections|properties|revid|parsewarnings",
        "disablelimitreport": "true",
        "disableeditsection": "true",
        "disablestylededuplication": "true",
        "disabletoc": "true",
        "disableeditlinks": "true",
        "disabletoclinks": "true",
        "inprop": "url"
    }

    async with session.get(base_url, params=params) as response:
        data = await response.json()
        try:
            html_content = data["parse"]["text"]["*"]
            soup = BeautifulSoup(html_content, "html.parser")

            infobox = soup.find("table", class_="infobox")
            if infobox:
                return parse_page_content(str(infobox), page_url)
            return [page_url, '', '', '', '', '', '', '']
        except:
            return parse_page_content(data, page_url)


def clean_genre_data(genres):
    if not isinstance(genres, str):
        return np.nan
    genres = re.sub(r'\[\d+\]', ',', genres)  # Replace [1], [2], etc. with a comma
    genres = re.sub(r'(?<=[a-z])(?=[A-Z])', ',', genres)  # Add a comma before lowercase next to uppercase
    cleaned_genres = [genre.strip() for genre in genres.split(',') if genre.strip()]
    cleaned_genres = ["wiki_" + genre.lower() for genre in cleaned_genres]
    return cleaned_genres

def clean_publisher_data(publisher, column):
    if not isinstance(publisher, str):
        return np.nan

    publisher = re.sub(r'\([^)]+\)', ',', publisher)  # Replace string subsets within parentheses with a comma
    publisher = re.sub(r'(?<=[a-z])(?=[A-Z])(?<!manga)(?<!ONE)', ',', publisher)
    cleaned_publishers = [pub.strip() for pub in publisher.split(',') if pub.strip()]
    cleaned_publishers = ["wiki_" + pub.lower() + column for pub in cleaned_publishers]
    return cleaned_publishers


def clean_manganato_data(df):
    for index, row in df.iterrows():
        ## fix views, votes, and avg_rating
        if isinstance(df.loc[index, 'views'], str) and 'M' in df.loc[index, 'views']:
            df.loc[index, 'views'] = df.loc[index, 'views'].strip('M')
            if '.' in df.loc[index, 'views']:
                df.loc[index, 'views'] = df.loc[index, 'views'].replace('.', '')
                df.loc[index, 'views'] += '00000'
            else:
                df.loc[index, 'views'] += '000000'
            df.loc[index, 'views'] = int(df.loc[index, 'views'])
            df.loc[index, 'votes'] = int(df.loc[index, 'votes'])
            df.loc[index, 'avg_rating'] = float(df.loc[index, 'avg_rating'])
        elif isinstance(df.loc[index, 'views'], str) and 'K' in df.loc[index, 'views']:
            df.loc[index, 'views'] = df.loc[index, 'views'].strip('K')
            if '.' in df.loc[index, 'views']:
                df.loc[index, 'views'] = df.loc[index, 'views'].replace('.', '')
                df.loc[index, 'views'] += '00'
            else:
                df.loc[index, 'views'] += '000'
            df.loc[index, 'views'] = int(df.loc[index, 'views'])
            df.loc[index, 'votes'] = int(df.loc[index, 'votes'])
            df.loc[index, 'avg_rating'] = float(df.loc[index, 'avg_rating'])
        else:
            if df.loc[index, 'views'] != None:
                df.loc[index, 'views'] = int(df.loc[index, 'views'])
                df.loc[index, 'votes'] = int(df.loc[index, 'votes'])
                df.loc[index, 'avg_rating'] = float(df.loc[index, 'avg_rating'])
            else:
                df.loc[index, 'views'] = 0
                df.loc[index, 'votes'] = 0
                df.loc[index, 'avg_rating'] = 0
        ## fix last_chapter
        if isinstance(df.loc[index, 'last_chapter'], str):

            if 'chapter' in df.loc[index, 'last_chapter'] or 'Chapter' in df.loc[index, 'last_chapter']:
                df.loc[index, 'last_chapter'] = df.loc[index, 'last_chapter'].lower()
                char_index = df.loc[index, 'last_chapter'].find('c')
                df.loc[index, 'last_chapter'] = df.loc[index, 'last_chapter'][char_index:]
                df.loc[index, 'last_chapter'] = df.loc[index, 'last_chapter'].replace('chapter ', '')

            if ':' in df.loc[index, 'last_chapter']:
                df.loc[index, 'last_chapter'] = list(df.loc[index, 'last_chapter'].split(':'))[0]
            df.loc[index, 'last_chapter'] = df.loc[index, 'last_chapter'].strip(' ')
            if ' ' in df.loc[index, 'last_chapter']:
                df.loc[index, 'last_chapter'] = list(df.loc[index, 'last_chapter'].split(' '))[0]
            if not df.loc[index, 'last_chapter'][0].isdigit():
                num_index = 0
                for char in df.loc[index, 'last_chapter']:
                    if char.isdigit():
                        num_index = df.loc[index, 'last_chapter'].find(char)
                        df.loc[index, 'last_chapter'] = df.loc[index, 'last_chapter'][num_index:]

            if not any(char.isdigit() for char in df.loc[index, 'last_chapter']):
                df.loc[index, 'last_chapter'] = 1
            try:
                df.loc[index, 'last_chapter'] = float(df.loc[index, 'last_chapter'])
                if df.loc[index, 'last_chapter'] == 0:
                    df.loc[index, 'last_chapter'] = 1
            except:
                df.loc[index, 'last_chapter'] = 1
    return df

