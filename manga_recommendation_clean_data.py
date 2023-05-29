from manga_pipeline.manga_scraper import scrape_main
from manga_pipeline.manga_data_transform import transform_data, connect_db
import pandas as pd
from os import getenv
import asyncio
pd.options.mode.chained_assignment = None


if __name__ == '__main__':

    with connect_db() as con:
        data_csv = getenv('MODEL_DATA')
        read_manga = pd.read_csv(data_csv)
        read_manga.to_sql("model_data_raw", con, if_exists="replace")
    asyncio.run(scrape_main())
    print("COMPLETED SCRAPING")
    print("BEGINNING TRANSFORMING DATA")
    transform_data()
    print("COMPLETED TRANSFORMING DATA")
    print("DONE")


