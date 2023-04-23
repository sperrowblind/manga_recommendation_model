from manga_scraper import scrape_manga
from manga_data_transform import transform_data
import sqlite3
from sqlite3 import Error
import pandas as pd
pd.options.mode.chained_assignment = None

def connect_db():
    try:
        con = sqlite3.connect("manga_recommendation_database.sqlite")
    except Error as con:
        pass
    return con

if __name__ == '__main__':

    with connect_db() as con:
        read_manga = pd.read_csv('read_manga.csv')
        read_manga.to_sql("model_data_raw", con, if_exists="replace")
    scrape_manga()
    transform_data()

