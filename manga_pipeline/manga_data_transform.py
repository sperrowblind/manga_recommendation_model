import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from sqlite3 import Error
from sklearn.feature_extraction.text import CountVectorizer
import os

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from .manga_scraper import clean_title


def connect_db():
    try:
        con = sqlite3.connect("manga_recommendation_database.sqlite")
    except Error as con:
        print("Unable to connect to database")
        pass
    return con

def manga_release_date_clean(x):
    if isinstance(x, str) and x.lower() == 'present':
        return -1
    try:
        x = x.strip(' 00:00:00')
        x = datetime.strptime(x, '%Y-%M-%d')
        x = x.toordinal()
        return x
    except:
        x = 0
        return x

def clean_text_nltk(description):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    description = description.replace('Description :', '')
    description = re.sub('[^a-zA-Z]', ' ', description)
    description = description.lower()
    description = description.split()
    description = [ps.stem(word) for word in description if not word in stop_words] # use stop_words instead of stopwords
    description = ' '.join(description)
    return description

def transform_data(df_transform = None):
    print('1')
    with connect_db() as con:
        df = pd.read_sql_query("SELECT * FROM scraped_data", con)

    df = df.applymap(lambda x: np.nan if isinstance(x, str) and x == '[]' else x)\
       .applymap(lambda x: x.strip('\'\"') if isinstance(x, str) else x)\
       .applymap(lambda x: x.strip('[]') if isinstance(x, str) else x)\
       .applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["Title"] = df["Title"].apply(clean_title)
    print('2')
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
    print('3')
    # turn wiki_start and end to an ordinal number
    start_date_col = 'wiki_start_date'
    final_df[start_date_col] = final_df[start_date_col].apply(lambda x: manga_release_date_clean(x))
    end_date_col = 'wiki_end_date'
    final_df[end_date_col] = final_df[end_date_col].apply(lambda x: manga_release_date_clean(x))

    # remove unneeded columns and attempt to drop duplicates
    to_drop = ['index', 'url_name', 'manganato_url', 'wiki_url']
    for column in to_drop:
        if column in final_df.columns:
            final_df.drop(column, axis=1, inplace=True)
    final_df.drop_duplicates(subset=['Title'], inplace=True)
    print('4')
    final_df.rename(columns={'Count of Title Characters': 'title_char_count'}, inplace=True)

    fill_values = {col: 0 if final_df[col].dtype != 'object' or col == 'wiki_volumes' or col == 'last_chapter' else 'NONE' for col in final_df.columns}
    final_df.fillna(fill_values, inplace=True)

    final_df['last_chapter'] = final_df['last_chapter'].apply(lambda x: float(x))
    print('5')
    #get dummies for any categorical variables

    need_dummies = ['wiki_demographic', 'status']
    for column in final_df.columns:
        if len(final_df.index) > 0:
            if isinstance(final_df.loc[0, column], str) and column != 'Title' \
                and column not in need_dummies and column != 'wiki_original_run' and column != 'description' and column != 'last_chapter':
                final_df[column] = final_df[column].apply(lambda x: 1 if x != 'NONE' else 0)
    final_df = pd.get_dummies(final_df, columns=need_dummies,drop_first=True)
    print('6')
    columns_to_delete = ['wiki_original_publisher_NONE', 'nan' , 'nan_x', 'compare_title', 'nan_y']
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
    original_df = pd.read_csv(os.path.join('csvs', "model_df.csv"))
    column_to_delete = 'Unnamed: 0'
    original_df.drop([column_to_delete], axis=1, inplace=True, errors='ignore')

    needed_columns = original_df.columns.tolist() + ['Title']
    df = df.reindex(columns=needed_columns, fill_value=0)
    return df[needed_columns]

def get_word_count_df(df):

    # clean description
    description_col = 'description'
    df[description_col] = df[description_col].apply(clean_text_nltk)
    cv = CountVectorizer()
    words = cv.fit_transform(df[description_col])
    word_counts = words.toarray()
    column_names = cv.get_feature_names_out()
    df['word_count_sum'] = words.sum(axis=1)

    word_counts_df = pd.DataFrame(words.toarray(), columns=column_names)
    word_counts_df.to_csv(os.path.join('csvs', 'word_counts_df.csv'))

    df = pd.concat([df, word_counts_df], axis=1)
    df.drop([description_col], axis=1, inplace=True)

    return df

# This code is to ensure that stopwords are downloaded
import ssl

try:
    # Check if the stopwords corpus is already downloaded
    nltk.data.find('corpora/stopwords')
except LookupError:
    # Download the stopwords corpus if it's not found
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')
