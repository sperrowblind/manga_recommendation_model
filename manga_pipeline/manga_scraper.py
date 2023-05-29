import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from sqlite3 import Error
import os
import asyncio
import aiohttp
import ssl

from manga_pipeline.manga_pipeline_utils.manganato_url_scrape import (
    clean_title,
    reformat_url_name,
    scrape_manganato_url,
    save_cache,
    load_cache
)

from manga_pipeline.manga_pipeline_utils.manganato_info_scrape import get_manga_info

from manga_pipeline.manga_pipeline_utils.wiki_url_scrape import process_row

from manga_pipeline.manga_pipeline_utils.wiki_info_scrape import (
    get_page_content,
    clean_genre_data,
    clean_manganato_data,
    clean_publisher_data
)


def connect_db():
    try:
        con = sqlite3.connect("manga_recommendation_database.sqlite")
    except Error as con:
        pass
    return con

async def manganato_url_scrape(manga_list=None):
    """Scrape Manganato URLs for manga titles."""
    if manga_list is None:
        with connect_db() as con:
            df = pd.read_sql_query("SELECT * FROM model_data_raw", con)
    else:
        df = pd.DataFrame(manga_list, columns=['Title'])

    df_search = df[['Title']].copy()
    df_search['url_name'] = df_search['Title'].apply(clean_title)
    df_search['url_name'] = df_search['url_name'].apply(reformat_url_name)

    cache = load_cache()
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        tasks = []
        for title in df_search['url_name']:
            task = asyncio.create_task(scrape_manganato_url(session, title, cache))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    df_search['manganato_url'] = results

    manual_input_manganato = df_search[df_search['manganato_url'].isnull()]
    with connect_db() as con:
        manual_input_manganato.to_sql("manganato_manual_input_needed", con, if_exists="replace")

    df_search.to_csv(os.path.join('csvs', 'manganato_urls.csv'))

    save_cache(cache)

    return df_search

async def manganato_info_search(df_search):
    df_manganato = df_search

    manga_info_df = []

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        tasks = []
        for index, row in df_manganato.iterrows():
            if df_manganato.loc[index, 'manganato_url'] == 'None' or df_manganato.loc[index, 'manganato_url'] is None:
                continue
            url = df_manganato.loc[index, 'manganato_url']
            task = asyncio.ensure_future(get_manga_info(session, url))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        for response in responses:
            if response is not None:
                manga_info_df.append(response)

    manga_info_df = pd.DataFrame(manga_info_df, columns=['manganato_url', 'author', 'num_authors', 'genre',
                                                         'num_genres', 'status', 'views', 'votes', 'avg_rating',
                                                         'last_chapter', 'description'])
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


async def wiki_url_scrape(manganato_info):
    try:
        manganato_info.drop(['Unnamed: 0'], axis=1, inplace=True)
    except:
        pass
    manganato_info.drop_duplicates(subset=['Title'], inplace=True)
    df_wiki = manganato_info
    df_wiki['wiki_url'] = 'None'

    df_wiki['wiki_url'] = df_wiki.apply(lambda x: 'https://en.wikipedia.org/wiki/' + x['Title'].strip().split('(')[0].replace(' ', '_') + "_(disambiguation)", axis=1)
    tasks = []
    for _, row in df_wiki.iterrows():
        tasks.append(asyncio.create_task(process_row(row)))

    wiki_urls = await asyncio.gather(*tasks)

    df_wiki['wiki_url'] = wiki_urls
    df_wiki['first_find_is_valid'] = df_wiki['wiki_url'] != 'None'
    df_wiki['second_find_is_valid'] = df_wiki['first_find_is_valid']
    df_wiki['has_wikipedia_page'] = df_wiki['first_find_is_valid'] | df_wiki['second_find_is_valid']
    columns_to_drop = ['first_find_is_valid', 'second_find_is_valid']
    df_wiki.drop(columns_to_drop, axis=1, inplace=True)

    manual_input_wiki = df_wiki[df_wiki['wiki_url'] == 'None']
    manual_input_wiki.to_csv(os.path.join('csvs', 'manual_input_wiki_urls.csv'))

    actual_wiki = df_wiki[df_wiki['wiki_url'] != 'None']
    actual_wiki.to_csv(os.path.join('csvs', 'valid_wiki_urls.csv'))
    df_wiki.to_csv(os.path.join('csvs', 'wiki_urls.csv'))
    return df_wiki


async def wiki_info_search(df_wiki):
    df_wikipedia = df_wiki.copy()
    manga_info_df = []

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        tasks = []
        no_index = []
        for index, row in df_wikipedia.iterrows():
            if df_wikipedia.loc[index, 'wiki_url'] == 'None':
                no_index.append(index)
                continue

            url = df_wikipedia.loc[index, 'wiki_url']
            task = asyncio.ensure_future(get_page_content(session, url))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        for index, response in enumerate(responses):
            if index in no_index:
                continue
            try:
                manga_info_df.append(response)
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
    df_wiki_publisher.drop_duplicates(inplace=True)
    df_wiki_publisher = df_wiki_publisher.pivot(index="wiki_url", columns='wiki_original_publisher', values='wiki_original_publisher')
    df_wiki_publisher.rename_axis(columns='wiki_original_publisher', inplace=True)
    df_wiki_publisher = df_wiki_publisher.groupby(level=0, axis=1).max().reset_index()

    df_wiki_english_publisher = df[['wiki_english_publisher', 'wiki_url']].reset_index(drop=True)
    df_wiki_english_publisher = df_wiki_english_publisher.explode('wiki_english_publisher')
    df_wiki_english_publisher.drop_duplicates(inplace=True)
    df_wiki_english_publisher = df_wiki_english_publisher.pivot(index="wiki_url", columns='wiki_english_publisher', values='wiki_english_publisher')
    df_wiki_english_publisher.rename_axis(columns='wiki_english_publisher', inplace=True)
    df_wiki_english_publisher = df_wiki_english_publisher.groupby(level=0, axis=1).max().reset_index()

    df_wiki_magazine = df[['wiki_magazine', 'wiki_url']].reset_index(drop=True)
    df_wiki_magazine = df_wiki_magazine.explode('wiki_magazine')
    df_wiki_magazine.drop_duplicates(inplace=True)
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


async def scrape_manga(manga_list=None):
    if not os.path.exists('csvs'):
        os.makedirs('csvs')
    print('STARTING SCRAPING')
    df = await manganato_url_scrape(manga_list)
    print('FOUND MANGANATO URLS')
    df = await manganato_info_search(df)
    print('FOUND MANGANATO INFO')
    df = await wiki_url_scrape(df)
    print('FOUND WIKI URLS')
    df = await wiki_info_search(df)
    print('FOUND WIKI INFO')
    return df

async def scrape_main():
    result = await scrape_manga()
    return result



"""df = pd.read_csv(os.path.join('csvs', 'wiki_urls.csv'))
df = wiki_info_search(df)
df.to_csv('test_wiki_info_final.csv')"""




