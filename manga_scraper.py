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
    print(url)
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
    manual_input_manganato.to_csv('manganato_manual_input.csv')
    with connect_db() as con:
        manual_input_manganato.to_sql("manganato_manual_input_needed", con, if_exists="replace")

    df_search.to_csv('test.csv')
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
        soup = BeautifulSoup( text , 'html.parser')
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
                    elif tag.text.strip() == 'Ongoing' or tag.text.strip() == 'ongoing':
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

        manga_info_df.append(manga_info)

    manga_info_df = pd.DataFrame(manga_info_df, columns=['manganato_url', 'author', 'num_authors', 'genre', 'num_genres', 'status', 'views', 'votes', 'avg_rating', 'last_chapter'])
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
        manganato_info = manganato_info.drop(['nan_x', 'nan_y', 'author', 'genre'], axis=1)
    except:
        manganato_info = manganato_info.drop(['author', 'genre'], axis=1)

    manganato_info = manganato_info.drop_duplicates()
    manganato_info.to_csv('test_2.csv')

    return manganato_info

def wiki_url_scrape(manganato_info):
    ## scrape wikipedia for manga urls if exist
    try:
        manganato_info.drop(['Unnamed: 0'], axis=1, inplace=True)
    except:
        pass
    manganato_info.drop_duplicates(inplace=True)
    df_wiki = manganato_info
    df_wiki['wiki_url'] = 'None'
    for index, row in df_wiki.iterrows():
        page_url = 'None'
        try:
            page = wikipedia.page(df_wiki.loc[index, 'Title'])
            page_url = page.url
            if 'manga series' in str(page.content) or \
                'manhwa' in str(page.content):
                df_wiki.loc[index, 'wiki_url'] = page_url
        except:
            pass
        if df_wiki.loc[index, 'wiki_url'] != 'None':
            continue
        try:
            page = wikipedia.page(df_wiki.loc[index, 'url_name'])
            page_url = page.url
            if 'manga series' in str(page.content) or \
                'manhwa' in str(page.content):
                df_wiki.loc[index, 'wiki_url'] = page_url
        except:
            pass
    for index, row in df_wiki.iterrows():
        url_compare = df_wiki.loc[index, 'wiki_url'].replace('https://en.wikipedia.org/wiki/', '').replace('_', ' ').lower()
        if df_wiki.loc[index, 'wiki_url'] == 'None' \
            or ('list' in url_compare or '(film)' in url_compare):
            try:
                results = wikipedia.search(df_wiki.loc[index, 'Title'] + ' manga', results=3)
                for i in range(0, len(results)):
                    try:
                        page = wikipedia.page(results[i])
                        match = SequenceMatcher(a=results[i], b=df_wiki.loc[index, 'Title']).ratio()
                        if match >= .5 or 'manga' in str(results[i]):
                            try:
                                url = page.url
                                df_wiki.loc[index, 'wiki_url'] = url
                                break
                            except:
                                pass
                    except:
                        pass
            except:
                continue
        url_compare = df_wiki.loc[index, 'wiki_url'].replace('https://en.wikipedia.org/wiki/', '').replace('_', ' ').lower()
        if df_wiki.loc[index, 'wiki_url'] == 'None' \
            or ('list' in url_compare or '(film)' in url_compare) \
                or (('(' + df_wiki.loc[index, 'Title'] + ')') in df_wiki.loc[index, 'wiki_url'] and 'manga' not in df_wiki.loc[index, 'wiki_url']):
            df_wiki.loc[index, 'wiki_url'] = 'https://en.wikipedia.org/wiki/' + df_wiki.loc[index, 'Title'].strip().replace(' ', '_')

    manual_input_wiki = df_wiki[df_wiki['wiki_url'] == 'None']
    manual_input_wiki.to_csv('manual_input_wiki.csv')
    with connect_db() as con:
        manual_input_wiki.to_sql("wiki_manual_input_needed", con, if_exists="replace")
    df_wiki.to_csv('test_3.csv')
    return df_wiki

def wiki_info_search(df_wiki):
    ## scrape wikipedia urls
    df_wikipedia = df_wiki
    manga_info_df = []
    for index, row in df_wikipedia.iterrows():
        if df_wikipedia.loc[index, 'wiki_url'] == 'None':
            continue
        url = df_wikipedia.loc[index, 'wiki_url']
        text = requests.get(url).content
        soup = BeautifulSoup( text , 'html.parser')
        try:
            manga_info = ['', '', '', '', '', '', '', '']

            manga_info[0] = url

            switchGenre = False
            switchOrigPublish = False
            switchEngPublish = False
            switchMagazine = False
            switchDemo = False
            switchOrigRun = False
            switchVolumes = False

            wiki_manga_info = []

            try:
                for info in  soup.findAll('tbody'):
                    try:
                        for tag in info.findAll('tr'):
                            for label in tag:
                                newLabel = label.text.strip().lower()
                                if newLabel == 'genre':
                                    switchGenre = True
                                    continue
                                elif ('by' in newLabel and 'pub' in newLabel) or ('lished' in newLabel):
                                    switchOrigPublish = True
                                    continue
                                elif newLabel == 'english publisher':
                                    switchEngPublish = True
                                    continue
                                elif newLabel == 'magazine':
                                    switchMagazine = True
                                    continue
                                elif newLabel == 'demographic':
                                    switchDemo = True
                                    continue
                                elif newLabel == 'original run':
                                    switchOrigRun = True
                                    continue
                                elif newLabel == 'volumes':
                                    switchVolumes = True
                                    continue
                                for test in label:
                                    newTest = test.text.strip().lower()
                                    if '.mw' in newTest:
                                        continue
                                    wiki_manga_info.append(newTest)
                            if switchGenre and manga_info[1] == '':
                                manga_info[1] = wiki_manga_info
                            elif switchOrigPublish and manga_info[2] == '':
                                manga_info[2] = wiki_manga_info
                            elif switchEngPublish and manga_info[3] == '':
                                manga_info[3] = wiki_manga_info
                            elif switchMagazine and manga_info[4] == '':
                                manga_info[4] = wiki_manga_info
                            elif switchDemo and manga_info[5] == '':
                                manga_info[5] = wiki_manga_info
                            elif switchOrigRun and manga_info[6] == '':
                                manga_info[6] = wiki_manga_info
                            elif switchVolumes and manga_info[7] == '':
                                manga_info[7] = wiki_manga_info

                            switchGenre = False
                            switchOrigPublish = False
                            switchEngPublish = False
                            switchMagazine = False
                            switchDemo = False
                            switchOrigRun = False
                            switchVolumes = False
                            wiki_manga_info = []
                    except:
                        pass
            except:
                continue
        except:
            pass
        manga_info_df.append(manga_info)


    manga_info_df = pd.DataFrame(manga_info_df, columns=['wiki_url', 'wiki_genres', 'wiki_original_publisher', 'wiki_english_publisher', 'wiki_magazine', 'wiki_demographic', 'wiki_original_run', 'wiki_volumes'])
    manga_info_df.to_csv('test_wiki.csv')
    for index, row in manga_info_df.iterrows():
        try:
            manga_info_df.loc[index, 'wiki_genres'].remove('[1]')
        except:
            pass
        try:
            manga_info_df.loc[index, 'wiki_genres'].remove('[2]')
        except:
            pass
        try:
            manga_info_df.loc[index, 'wiki_genres'].remove('[3]')
        except:
            pass
        try:
            manga_info_df.loc[index, 'wiki_genres'].remove('[4]')
        except:
            pass
        try:
            manga_info_df.loc[index, 'wiki_genres'].remove('')
        except:
            pass
        try:
            manga_info_df.loc[index, 'wiki_genres'].remove(',')
        except:
            pass
        for i in range(0, len(manga_info_df.loc[index, 'wiki_genres'])):
            try:
                manga_info_df.loc[index, 'wiki_genres'][i] = 'wiki_' + manga_info_df.loc[index, 'wiki_genres'][i]
            except:
                pass
            try:
                manga_info_df.loc[index, 'wiki_genres'][i] = manga_info_df.loc[index, 'wiki_genres'][i].replace('[1]', ';')
            except:
                pass
            try:
                manga_info_df.loc[index, 'wiki_genres'][i] = manga_info_df.loc[index, 'wiki_genres'][i].replace('[2]', ';')
            except:
                pass
            try:
                manga_info_df.loc[index, 'wiki_genres'][i] = manga_info_df.loc[index, 'wiki_genres'][i].replace('[3]', ';')
            except:
                pass
            try:
                manga_info_df.loc[index, 'wiki_genres'][i] = manga_info_df.loc[index, 'wiki_genres'][i].replace('[4]', ';')
            except:
                pass
            try:
                manga_info_df.loc[index, 'wiki_genres'][i] = manga_info_df.loc[index, 'wiki_genres'][i].replace('[5]', ';')
            except:
                pass
        if len(manga_info_df.loc[index, 'wiki_genres']) == 1:
            try:
                manga_info_df.loc[index, 'wiki_genres'] = manga_info_df.loc[index, 'wiki_genres'][0]
                manga_info_df.loc[index, 'wiki_genres'] = manga_info_df.loc[index, 'wiki_genres'].strip().split(';')
                try:
                    manga_info_df.loc[index, 'wiki_genres'] = manga_info_df.loc[index, 'wiki_genres'][0].strip().split(',')
                except:
                    pass
            except:
                pass
        try:
            manga_info_df.loc[index, 'wiki_genres'].remove('')
        except:
            pass
        try:
            manga_info_df.loc[index, 'wiki_genres'].remove(',')
        except:
            pass



    manga_info_df.to_csv('test_4.csv')


    wiki_urls = df_wiki[df_wiki['wiki_url'] != None]
    wiki_info_df = manga_info_df

    wiki_urls = pd.merge(wiki_urls, wiki_info_df, on='wiki_url', how='inner')

    df_wiki_genre = wiki_urls[['wiki_genres', 'wiki_url']]

    df_wiki_genre = df_wiki_genre.explode('wiki_genres')
    df_wiki_genre = df_wiki_genre.drop_duplicates()

    df_wiki_genre = df_wiki_genre.pivot(index="wiki_url", columns='wiki_genres', values='wiki_genres')
    df_wiki_genre = df_wiki_genre.reset_index()
    df_wiki_genre = df_wiki_genre.groupby('wiki_url').agg(lambda x: x.max())

    try:
        df_wiki_genre.drop([''], axis=1)
    except:
        pass

    wiki_urls = pd.merge(wiki_urls, df_wiki_genre, on='wiki_url', how='inner')

    wiki_urls = wiki_urls.drop(['wiki_genres'], axis=1)

    df = wiki_urls

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
                df.loc[index, 'last_chapter'] = np.nan
            try:
                df.loc[index, 'last_chapter'] = float(df.loc[index, 'last_chapter'])
            except:
                df.loc[index, 'last_chapter'] = np.nan

        ## fix wiki_demographic
        try:
            df.loc[index, 'wiki_demographic'] = df.loc[index, 'wiki_demographic'][0]
            df.loc[index, 'wiki_demographic'] = df.loc[index, 'wiki_demographic'].strip('\'')

        except:
            df.loc[index, 'wiki_demographic'] = np.nan

        ## fix wiki_original_run
        try:
            df.loc[index, 'wiki_original_run'] = df.loc[index, 'wiki_original_run'][0]
            df.loc[index, 'wiki_original_run'] = df.loc[index, 'wiki_original_run'].lower()
            df.loc[index, 'wiki_original_run'] = df.loc[index, 'wiki_original_run'].strip('\'')
            try:
                if ' ' in df.loc[index, 'wiki_original_run']:
                    try:
                        df.loc[index, 'wiki_original_run'] = datetime.strptime(df.loc[index, 'wiki_original_run'], '%B %d, %Y')
                    except:
                        pass
                    try:
                        df.loc[index, 'wiki_original_run'] = datetime.strptime(df.loc[index, 'wiki_original_run'], '%B %Y')
                    except:
                        pass
                    try:
                        df.loc[index, 'wiki_original_run'] = datetime.strptime(df.loc[index, 'wiki_original_run'], '%d %B %Y')
                    except:
                        pass
                else:
                    df.loc[index, 'wiki_original_run'] = datetime.strptime(df.loc[index, 'wiki_original_run'], '%Y')
                    pass
            except:
                pass
        except:
            df.loc[index, 'wiki_original_run'] = np.nan

        ## fix wiki_volumes
        try:
            df.loc[index, 'wiki_volumes'] = df.loc[index, 'wiki_volumes'][0]
            df.loc[index, 'wiki_volumes'] = list(df.loc[index, 'wiki_volumes'].split(' '))[0]
            df.loc[index, 'wiki_volumes'] = int(df.loc[index, 'wiki_volumes'])
        except:
            df.loc[index, 'wiki_volumes'] = np.nan

        ## fix publisher, magazine, and english publisher
        try:
            df.loc[index, 'wiki_original_publisher'] = df.loc[index, 'wiki_original_publisher'][0]
        except:
            df.loc[index, 'wiki_original_publisher'] = np.nan
        try:
            df.loc[index, 'wiki_english_publisher'] = df.loc[index, 'wiki_english_publisher'][0]
        except:
            df.loc[index, 'wiki_english_publisher'] = np.nan
        try:
            df.loc[index, 'wiki_magazine'] = df.loc[index, 'wiki_magazine'][0]
        except:
            df.loc[index, 'wiki_magazine'] = np.nan

    columns_to_remove = ['Unnamed: 0', 'Unnamed: 189', 'wiki_', 'wiki_,']
    for column in df.columns:
        if column == '' or column == np.nan or column in columns_to_remove:
            df = df.drop([column], axis=1)

    df = df.drop_duplicates()
    df.to_csv("scraped_data.csv")
    with connect_db() as con:
        df.to_sql("scraped_data", con, if_exists="replace")
    return df

def scrape_manga(manga_list=None):
    print('STARTING SCRAPING')
    df = manganato_url_scrape(manga_list)
    print('FOUND MANGANATO URLS')
    df = manganato_info_search(df)
    print('FOUND MANGANATO INFO')
    df = wiki_url_scrape(df)
    print('FOUND wiki urls')
    df = wiki_info_search(df)
    print('found wiki info')
    return df



#df = pd.read_csv('read_manga.csv')
#df = manganato_url_scrape(df)
#df = pd.read_csv('test_3.csv')
#df = wiki_info_search(df)


