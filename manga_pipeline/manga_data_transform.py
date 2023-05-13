import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from sqlite3 import Error
import os

from .manga_scraper import clean_title


def connect_db():
    try:
        con = sqlite3.connect("manga_recommendation_database.sqlite")
    except Error as con:
        print("Unable to connect to database")
        pass
    return con

def manga_release_date_clean(x):
    try:
        x = x.strip(' 00:00:00')
        x = datetime.strptime(x, '%Y-%M-%d')
        x = x.toordinal()
        return x
    except:
        x = 0
        return x

def transform_data(df_transform = None):
    with connect_db() as con:
        df = pd.read_sql_query("SELECT * FROM scraped_data", con)

    df = df.applymap(lambda x: np.nan if isinstance(x, str) and x == '[]' else x)\
       .applymap(lambda x: x.strip('\'\"') if isinstance(x, str) else x)\
       .applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)\
       .applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["Title"] = df["Title"].apply(clean_title)

    if df_transform is None:
        final_df = df
        original_data_set = os.getenv('MODEL_DATA')
        if os.path.exists(original_data_set):
            custom_columns = pd.read_csv(original_data_set, usecols=['Title', 'Rating', 'Count of Title Characters'])
            custom_columns['Title'] = custom_columns['Title'].apply(clean_title)
            final_df = pd.merge(final_df, custom_columns, on='Title', how='inner')
        else:
            final_df = df
    else:
        final_df = df_transform
        final_df['Count of Title Characters'] = final_df['Title'].apply(lambda x: len(str(x)))

    # turn wiki_original_run to an ordinal number
    date_col = 'wiki_original_run'
    final_df[date_col] = final_df[date_col].apply(lambda x: manga_release_date_clean(x))

    # remove unneeded columns and attempt to drop duplicates
    to_drop = ['index', 'url_name', 'manganato_url', 'wiki_url']
    for column in to_drop:
        if column in final_df.columns:
            final_df.drop(column, axis=1, inplace=True)
    final_df.drop_duplicates(inplace=True)

    final_df.rename(columns={'Count of Title Characters': 'title_char_count'}, inplace=True)

    fill_values = {col: 0 if final_df[col].dtype != 'object' or col == 'wiki_volumes' or col == 'last_chapter' else 'NONE' for col in final_df.columns}
    final_df.fillna(fill_values, inplace=True)

    #get dummies for any categorical variables

    need_dummies = ['wiki_original_publisher', 'wiki_english_publisher', 'wiki_magazine', 'wiki_demographic', 'status']
    for column in final_df.columns:
        if len(final_df.index) > 0:
            if isinstance(final_df.loc[0, column], str) and column != 'Title' \
                and column not in need_dummies and column != 'wiki_original_run':
                final_df[column] = final_df[column].apply(lambda x: 1 if x != 'NONE' else 0)
    final_df = pd.get_dummies(final_df, columns=need_dummies,drop_first=True)

    columns_to_delete = ['wiki_original_publisher_NONE', 'nan']
    for column in final_df.columns:
        if column in columns_to_delete:
            final_df.drop([column], axis=1, inplace=True)

    if not os.path.exists('csvs'):
        os.makedirs('csvs')

    if df_transform is None:
        final_df.to_csv(os.path.join('csvs', 'final_data.csv'))
        with connect_db() as con:
            final_df.to_sql("model_data", con, if_exists="replace")
    else:
        final_df.to_csv(os.path.join('csvs', 'new_transformed_data.csv'))
    return final_df

def columns_for_model(df):
    original_df = pd.read_csv("model_df.csv")
    column_to_delete = 'Unnamed: 0'
    original_df.drop([column_to_delete], axis=1, inplace=True, errors='ignore')

    needed_columns = original_df.columns.tolist() + ['Title']
    df = df.reindex(columns=needed_columns, fill_value=0)
    return df[needed_columns]

