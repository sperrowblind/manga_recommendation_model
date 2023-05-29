import pandas as pd
from manga_pipeline.manga_data_transform import connect_db


with connect_db() as con:

    scraped = pd.read_sql_query("SELECT Title FROM scraped_data", con)
    scraped_set = set(scraped['Title'])
    final = pd.read_sql_query("SELECT Title FROM model_data", con)
    final_set = set(final['Title'])
    read = pd.read_sql_query("SELECT Title FROM model_data_raw", con)
    read_set = set(read['Title'])
    manganato_missing = pd.read_sql_query("SELECT Title FROM manganato_manual_input_needed", con)
    manganato_set = set(manganato_missing['Title'])


print('NOTE: Some titles may appear to be missing but still be retained in certain data files')
print('The most important missing data comes from if there is more missing than in manganato_manual_input')

print('Find missing from scraped_data and final:')

unread_titles = scraped_set - final_set
for title in unread_titles:
    print(title)

print('')
print('find missing from read_manga and final:')

unread_titles = read_set - final_set
for title in unread_titles:
    print(title)

print('')
print('find missing from read_manga and scraped:')

unread_titles = read_set - scraped_set
for title in unread_titles:
    print(title)

print('')
print('find missing from final and manganato_missing:')

unread_titles = manganato_set - final_set
for title in unread_titles:
    print(title)

