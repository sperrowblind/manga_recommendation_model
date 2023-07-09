import pickle
from flask import Markup
from html import unescape
import pandas as pd

from ..manga_pipeline.manga_data_transform import connect_db


def make_hyperlink(website):
    try:
        return f'<a href="{website}" target="_blank">{website}</a>'
    except:
        return website


def load_model():
    with open('final_model_6.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def find_predictions(df, model):
    titles = pd.DataFrame(df['Title'])

    df.drop(['Title'], axis=1, inplace=True)

    prediction = model.predict(df)
    df_predictions = pd.DataFrame(prediction, columns = ['Predicted Rating'])
    prediction = titles.join(df_predictions)
    prediction['Predicted Rating'] = prediction['Predicted Rating'].fillna(1.0).astype(int)

    with connect_db() as con:
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS predicted_data (Title TEXT, Predicted_Rating INT, URL TEXT, Description TEXT, Model_Version TEXT, PRIMARY KEY (Title, Model_Version))")
        manganato_data = pd.read_sql("SELECT Title, manganato_url FROM manganato_data", con)
        prediction = pd.merge(prediction, manganato_data, on='Title', how='left')
        description_data = pd.read_sql("SELECT Title, description FROM scraped_data", con)

        prediction = pd.merge(prediction, description_data, on='Title', how='left')
        prediction['manganato_url'] = prediction['manganato_url'].apply(make_hyperlink)

        max_description_length = 100
        prediction['description'] = prediction['description'].apply(lambda x: x.replace('Description :\n', ''))
        prediction['description'] = prediction['description'].apply(lambda x: x.replace('\n', ' '))
        prediction['truncated_description'] = prediction['description'].str[:max_description_length]
        prediction['truncated_description'] += '...'
        prediction.drop('description', axis=1, inplace=True)
        prediction.rename(columns={'truncated_description': 'description'}, inplace=True)
        prediction['Title'] = prediction['Title'].apply(lambda x: x.title())

        for _, row in prediction.iterrows():
            title = row['Title']
            predicted_rating = row['Predicted Rating']
            url = row['manganato_url']
            description = row['description']
            cur.execute("SELECT * FROM predicted_data WHERE Title=? AND Model_Version=?", (title, '1.0.6'))
            existing_row = cur.fetchone()
            if existing_row is None:
                cur.execute("INSERT INTO predicted_data (Title, Predicted_Rating, URL, Description, Model_Version) VALUES (?, ?, ?, ?, ?)", (title, predicted_rating, url, description, '1.0.6'))
            elif existing_row[1] != predicted_rating:
                cur.execute("UPDATE predicted_data SET Predicted_Rating=?, Model_Version=? WHERE Title=?", (predicted_rating, '1.0.6', title))
        con.commit()
    prediction_table = Markup(unescape(prediction.to_html(index=False, classes='table table-striped')))
    return prediction_table

