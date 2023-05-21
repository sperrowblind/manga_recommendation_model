import requests
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import wikipedia
from difflib import SequenceMatcher
from datetime import datetime
import sqlite3
from sqlite3 import Error
import os
import ast
import urllib.parse
import json


def connect_db():
    try:
        con = sqlite3.connect("manga_recommendation_database.sqlite")
    except Error as con:
        pass
    return con

def clean_title(title):
    """Clean title string by removing non-alphanumeric characters."""
    return re.sub(r'\W+', ' ', title).strip()

def reformat_url_name(title):
    """Return URL name for given title."""
    return title.replace(' ', '_').lower().strip()

def scrape_manganato_url(title):
    """Scrape Manganato URL for given title."""
    url = f'https://manganato.com/search/story/{reformat_url_name(title)}'
    text = requests.get(url).content
    soup = BeautifulSoup(text, 'html.parser')
    links = soup.find_all(class_='item-img bookmark_check', href=True)
    title = title.replace('_', ' ').strip()
    for link in links:
        if len(links) == 1 or SequenceMatcher(a=link['title'].lower(), b=title).ratio() >= .75:
            return link['href']
    return None

def manganato_url_scrape(manga_list=None):
    """Scrape Manganato URLs for manga titles."""
    if manga_list is None:
        with connect_db() as con:
            df = pd.read_sql_query("SELECT * FROM model_data_raw", con)
    else:
        df = pd.DataFrame(manga_list, columns=['Title'])

    df_search = df[['Title']].copy()
    df_search['url_name'] = df_search['Title'].apply(clean_title)
    df_search['url_name'] = df_search['url_name'].apply(reformat_url_name)
    df_search['manganato_url'] = df_search['url_name'].apply(scrape_manganato_url)

    manual_input_manganato = df_search[df_search['manganato_url'].isnull()]
    with connect_db() as con:
        manual_input_manganato.to_sql("manganato_manual_input_needed", con, if_exists="replace")

    df_search.to_csv(os.path.join('csvs', 'manganato_urls.csv'))
    return df_search

def manganato_info_search(df_search):
    ## scrape for manga info from manganato
    df_manganato = df_search

    manga_info_df = []
    for index, row in df_manganato.iterrows():
        if df_manganato.loc[index, 'manganato_url'] == 'None' or df_manganato.loc[index, 'manganato_url'] is None:
            continue
        url = df_manganato.loc[index, 'manganato_url']
        text = requests.get(url).content
        soup = BeautifulSoup(text , 'html.parser')
        try:
            manga_info = []
            author = []
            genre = []
            manga_info.append(url)
            info = soup.find('div', {'class': 'story-info-right'})
            try:
                for tag in info.findAll('a', {'class': 'a-h'}):
                    try:
                        if tag['rel'][0] == 'nofollow':
                            author.append(tag.text.strip())
                    except:
                        genre.append(tag.text.strip())
                manga_info.append(author)
                manga_info.append(len(author))
                manga_info.append(genre)
                manga_info.append(len(genre))
            except:
                pass
            try:
                for tag in info.findAll('td', {'class': 'table-value'}):
                    if tag.text.strip() == 'Completed' or tag.text.strip() == 'completed':
                        manga_info.append('finished')
                        break
                    elif tag.text.strip() == 'Ongoing' or tag.text.strip() == 'ongoing' \
                        or 'Ongoing' in tag.text.strip() or 'ongoing' in tag.text.strip():
                        manga_info.append('ongoing')
                        break
            except:
                pass
            try:
                for tag in info.findAll('p'):
                    try:
                        check_view = tag.find('span', {'class': 'stre-label'})
                        if check_view.text.strip() == 'View :':
                            view = tag.find('span', {'class': 'stre-value'}).text.strip()
                    except:
                        continue
                manga_info.append(view)
            except:
                pass
            try:
                num_ratings = info.find('em', {'property': 'v:votes'})
                avg_rating = info.find('em', {'property': 'v:average'})
                manga_info.append(num_ratings.text.strip())
                manga_info.append(avg_rating.text.strip())
            except:
                pass
        except:
            pass
        try:
            info = soup.find('div', {'class': 'panel-story-chapter-list'})
            last_chapter = info.find('a', {'class': 'chapter-name text-nowrap'}).text.strip()
            manga_info.append(last_chapter)
        except:
            pass
        try:
            info = soup.find('div', {'class': 'panel-story-info-description'})
            description = info.text.strip()
            manga_info.append(description)
        except:
            pass

        manga_info_df.append(manga_info)

    manga_info_df = pd.DataFrame(manga_info_df, columns=['manganato_url', 'author', 'num_authors', 'genre', 'num_genres', 'status', 'views', 'votes', 'avg_rating', 'last_chapter', 'description'])
    manganato_info = pd.merge(df_manganato, manga_info_df, on='manganato_url', how='inner')
    try:
        manganato_info = manganato_info.drop(['Unnamed: 0'], axis=1)
    except:
        pass
    manganato_info.reset_index()

    author_frame = manganato_info[["manganato_url", "author"]]

    author_frame = author_frame.explode('author')
    author_frame = author_frame.drop_duplicates()

    author_frame = author_frame.pivot(index="manganato_url", columns='author', values='author')
    author_frame = author_frame.reset_index()
    author_frame = author_frame.groupby('manganato_url').agg(lambda x: x.max())
    manganato_info = pd.merge(manganato_info, author_frame, on='manganato_url', how='inner')


    df_author_and_genre = manganato_info[["genre", 'manganato_url']]

    df_author_and_genre = df_author_and_genre.explode('genre')
    df_author_and_genre = df_author_and_genre.drop_duplicates()

    df_author_and_genre = df_author_and_genre.pivot(index="manganato_url", columns='genre', values='genre')
    df_author_and_genre = df_author_and_genre.reset_index()
    df_author_and_genre = df_author_and_genre.groupby('manganato_url').agg(lambda x: x.max())
    manganato_info = pd.merge(manganato_info, df_author_and_genre, on='manganato_url', how='inner')
    try:
        manganato_info = manganato_info.drop(['nan_x', 'nan_y', 'genre'], axis=1)
    except:
        manganato_info = manganato_info.drop(['genre'], axis=1)

    manganato_info.drop_duplicates(subset=['Title'], inplace=True)
    manganato_info.to_csv(os.path.join('csvs', 'manganato_info.csv'))

    return manganato_info


def is_wikipedia_page(url, author):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        page_content = soup.get_text()
        if '200' in str(response):
            if isinstance(author, str):
                author = ast.literal_eval(author)
            for item in author:
                for word in item.split():
                    author_regex = r'(?i)\b{}\b'.format(re.escape(word))
                    if re.search(author_regex, page_content) and ('serialized' in page_content or 'manga series' in page_content or 'manhwa' in page_content):
                        return True
        return False
    except Exception as e:
        print(f"found exception at {url}: {e}")
        return False

def find_disambiguation_url(url, author):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        anchor_elements = soup.find_all("a")
        for link_element in anchor_elements:
            href = link_element.get("href")
            if href and '(manga)' in href:
                if is_wikipedia_page('https://en.wikipedia.org' + href, author):
                    return 'https://en.wikipedia.org' + href
        return 'None'
    except Exception as e:
        print(f"found exception at {url}: {e}")
        return 'None'

def final_wiki_url_search(title, author):
    try:
        url = 'https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={}'.format(title + ' manga')
        response = requests.get(url)
        data = response.json()
        if 'query' in data and 'search' in data['query']:
            search_results = data['query']['search']
            if search_results:
                first_result = search_results[0]
                page_title = first_result['title']
                url = 'https://en.wikipedia.org/w/api.php?action=query&format=json&prop=info&pageids={}&inprop=url'.format(first_result['pageid'])
                response = requests.get(url)
                data = response.json()
                if 'query' in data and 'pages' in data['query']:
                    page_info = data['query']['pages']
                    if page_info:
                        page = page_info[list(page_info.keys())[0]]
                        page_url = page['fullurl']
                        if is_wikipedia_page(page_url, author):
                            return page_url
    except Exception as e:
        print(f"found exception at {title}: {e}")
        return 'None'
    return 'None'

def wiki_url_scrape(manganato_info):
    try:
        manganato_info.drop(['Unnamed: 0'], axis=1, inplace=True)
    except:
        pass
    manganato_info.drop_duplicates(subset=['Title'], inplace=True)
    df_wiki = manganato_info
    df_wiki['wiki_url'] = 'None'

    df_wiki['wiki_url'] = df_wiki.apply(lambda x: 'https://en.wikipedia.org/wiki/' + x['Title'].strip().split('(')[0].replace(' ', '_') + "_(disambiguation)", axis=1)
    df_wiki['wiki_url'] = df_wiki.apply(lambda x: find_disambiguation_url(x['wiki_url'], x['author']), axis=1)
    df_wiki['first_find_is_valid'] = False
    df_wiki['first_find_is_valid'] = df_wiki.apply(lambda x: False if x['wiki_url'] == 'None' else True, axis=1)
    df_wiki['wiki_url'] = df_wiki.apply(lambda x: final_wiki_url_search(x['Title'].split('(')[0], x['author']) if not x['first_find_is_valid'] else x['wiki_url'], axis=1)
    df_wiki['second_find_is_valid'] = df_wiki.apply(lambda x: False if x['wiki_url'] == 'None' else True, axis=1)

    df_wiki['has_wikipedia_page'] = df_wiki.apply(lambda x: True if x['first_find_is_valid'] or x['second_find_is_valid'] else False, axis=1)
    columns_to_drop = ['first_find_is_valid', 'second_find_is_valid']
    df_wiki['compare_title'] = df_wiki['Title']
    df_wiki.drop(columns_to_drop, axis=1, inplace=True)

    manual_input_wiki = df_wiki[df_wiki['wiki_url'] == 'None']
    manual_input_wiki.to_csv(os.path.join('csvs', 'manual_input_wiki_urls.csv'))

    actual_wiki = df_wiki[df_wiki['wiki_url'] != 'None']
    actual_wiki.to_csv(os.path.join('csvs', 'valid_wiki_urls.csv'))
    df_wiki.to_csv(os.path.join('csvs', 'wiki_urls.csv'))
    return df_wiki



def get_page_content(page_url):
    base_url = "https://en.wikipedia.org/w/api.php"
    page_title = page_url.split("/")[-1]
    page_title = urllib.parse.unquote(page_title)
    params = {
        "action": "parse",
        "format": "json",
        "page": page_title,
        "prop": "text|displaytitle|iwlinks|categories|templates|images|sections|properties|revid|parsewarnings",
        "disablelimitreport": True,
        "disableeditsection": True,
        "disablestylededuplication": True,
        "disabletoc": True,
        "disableeditsection": True,
        "disableeditlinks": True,
        "disabletoclinks": True,
        "inprop": "url"
    }

    response = requests.get(base_url, params=params)
    data = response.json()
    try:
        html_content = data["parse"]["text"]["*"]
        soup = BeautifulSoup(html_content, "html.parser")

        infobox = soup.find("table", class_="infobox")
        if infobox:
            return str(infobox)
        return ""
    except:
        return data


def parse_page_content(page_content):
    manga_info = ['', '', '', '', '', '', '', '']

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
            df.loc[index, 'views'] = int(df.loc[index, 'views'])
            df.loc[index, 'votes'] = int(df.loc[index, 'votes'])
            df.loc[index, 'avg_rating'] = float(df.loc[index, 'avg_rating'])
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


def wiki_info_search(df_wiki):
    df_wikipedia = df_wiki.copy()
    manga_info_df = []

    for index, row in df_wikipedia.iterrows():
        if df_wikipedia.loc[index, 'wiki_url'] == 'None':
            continue

        url = df_wikipedia.loc[index, 'wiki_url']

        try:
            page_content = get_page_content(url)
            manga_info = parse_page_content(page_content)
            manga_info[0] = df_wikipedia.loc[index, 'wiki_url']

            manga_info_df.append(manga_info)

        except Exception as e:
            print(f"Error retrieving data for title: {df_wikipedia.loc[index, 'Title']}: {str(e)}")
            continue

    manga_info_df = pd.DataFrame(manga_info_df, columns=['wiki_url', 'wiki_genres', 'wiki_original_publisher', 'wiki_english_publisher', 'wiki_magazine', 'wiki_demographic', 'wiki_original_run', 'wiki_volumes'])
    manga_info_df['wiki_genres'] = manga_info_df['wiki_genres'].apply(clean_genre_data)
    manga_info_df.to_csv(os.path.join('csvs', 'wiki_info.csv'))

    wiki_urls = df_wiki[df_wiki['wiki_url'].notnull()]
    wiki_info_df = manga_info_df
    wiki_urls = pd.merge(wiki_urls, wiki_info_df, on='wiki_url', how='left')

    df_wiki_genre = wiki_urls[['wiki_genres', 'wiki_url']]
    df_wiki_genre = df_wiki_genre.explode('wiki_genres').drop_duplicates()
    df_wiki_genre = df_wiki_genre.pivot(index="wiki_url", columns='wiki_genres', values='wiki_genres').reset_index()
    df_wiki_genre = df_wiki_genre.groupby('wiki_url').agg(lambda x: x.max())
    df_wiki_genre.drop([''], axis=1, inplace=True, errors='ignore')
    wiki_urls = pd.merge(wiki_urls, df_wiki_genre, on='wiki_url', how='inner')
    wiki_urls = wiki_urls.drop(['wiki_genres'], axis=1)

    df = wiki_urls
    df.to_csv(os.path.join('csvs', "combined_non_transformed_data.csv"))
    df = clean_manganato_data(df)

    df['wiki_start_date'] = np.nan
    df['wiki_end_date'] = np.nan

    for index, row in df.iterrows():
        try:
            df.loc[index, 'wiki_demographic'] = df.loc[index, 'wiki_demographic'].split(',')[0]
        except:
            pass

        try:
            run_dates = df.loc[index, 'wiki_original_run'].split('–')
            df.loc[index, 'wiki_start_date'] = run_dates[0].strip()
            df.loc[index, 'wiki_end_date'] = run_dates[1].strip()

            date_formats = ['%B %d, %Y', '%B %Y', '%d %B %Y', '%Y']
            for fmt in date_formats:
                try:
                    df.loc[index, 'wiki_start_date'] = datetime.strptime(df.loc[index, 'wiki_start_date'], fmt)
                    break
                except:
                    pass
            for fmt in date_formats:
                try:
                    df.loc[index, 'wiki_end_date'] = datetime.strptime(df.loc[index, 'wiki_end_date'], fmt)
                    break
                except:
                    pass
        except:
            pass

        try:
            df.loc[index, 'wiki_volumes'] = int(df.loc[index, 'wiki_volumes'].split(' ')[0])
        except:
            df.loc[index, 'wiki_volumes'] = np.nan

    df['wiki_original_publisher'] = df['wiki_original_publisher'].apply(lambda publisher: clean_publisher_data(publisher, '_original_publisher'))
    df['wiki_english_publisher'] = df['wiki_english_publisher'].apply(lambda publisher: clean_publisher_data(publisher, '_english_publisher'))
    df['wiki_magazine'] = df['wiki_magazine'].apply(lambda publisher: clean_publisher_data(publisher, '_magazine'))

    df_wiki_publisher = df[['wiki_original_publisher', 'wiki_url']].reset_index(drop=True)
    df_wiki_publisher = df_wiki_publisher.explode('wiki_original_publisher')
    df_wiki_publisher.drop_duplicates(subset=['wiki_url'], inplace=True)
    df_wiki_publisher = df_wiki_publisher.pivot(index="wiki_url", columns='wiki_original_publisher', values='wiki_original_publisher')
    df_wiki_publisher.rename_axis(columns='wiki_original_publisher', inplace=True)
    df_wiki_publisher = df_wiki_publisher.groupby(level=0, axis=1).max().reset_index()

    df_wiki_english_publisher = df[['wiki_english_publisher', 'wiki_url']].reset_index(drop=True)
    df_wiki_english_publisher = df_wiki_english_publisher.explode('wiki_english_publisher')
    df_wiki_english_publisher.drop_duplicates(subset=['wiki_url'], inplace=True)
    df_wiki_english_publisher = df_wiki_english_publisher.pivot(index="wiki_url", columns='wiki_english_publisher', values='wiki_english_publisher')
    df_wiki_english_publisher.rename_axis(columns='wiki_english_publisher', inplace=True)
    df_wiki_english_publisher = df_wiki_english_publisher.groupby(level=0, axis=1).max().reset_index()

    df_wiki_magazine = df[['wiki_magazine', 'wiki_url']].reset_index(drop=True)
    df_wiki_magazine = df_wiki_magazine.explode('wiki_magazine')
    df_wiki_magazine.drop_duplicates(subset=['wiki_url'], inplace=True)
    df_wiki_magazine = df_wiki_magazine.pivot(index="wiki_url", columns='wiki_magazine', values='wiki_magazine')
    df_wiki_magazine.rename_axis(columns='wiki_magazine', inplace=True)
    df_wiki_magazine = df_wiki_magazine.groupby(level=0, axis=1).max().reset_index()

    df_wiki_publisher_1 = pd.merge(df_wiki_english_publisher, df_wiki_magazine, on='wiki_url', how='inner')
    df = df.drop(['wiki_original_publisher', 'wiki_english_publisher', 'wiki_magazine', 'wiki_original_run', 'author'], axis=1)
    df = pd.merge(df, df_wiki_publisher, on='wiki_url', how='inner')
    df = pd.merge(df, df_wiki_publisher_1, on='wiki_url', how='inner')

    columns_to_remove = ['Unnamed: 0', 'Unnamed: 189', 'wiki_', 'wiki_,']
    df = df.drop(columns=[col for col in df.columns if col in columns_to_remove or col == '' or col is np.nan])
    df.drop_duplicates(subset=['Title'], inplace=True)

    df.to_csv(os.path.join('csvs', "scraped_data.csv"))
    with connect_db() as con:
        df.to_sql("scraped_data", con, if_exists="replace")
    return df


def scrape_manga(manga_list=None):
    if not os.path.exists('csvs'):
        os.makedirs('csvs')
    print('STARTING SCRAPING')
    df = manganato_url_scrape(manga_list)
    print('FOUND MANGANATO URLS')
    df = manganato_info_search(df)
    print('FOUND MANGANATO INFO')
    df = wiki_url_scrape(df)
    print('FOUND WIKI URLS')
    df = wiki_info_search(df)
    print('FOUND WIKI INFO')
    return df



"""df = pd.read_csv(os.path.join('csvs', 'wiki_urls.csv'))
df = wiki_info_search(df)
df.to_csv('test_wiki_info_final.csv')"""




