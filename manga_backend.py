from flask import Flask, request, render_template
import pandas as pd

from .manga_pipeline.manga_scraper import connect_db, scrape_manga
from .manga_pipeline.manga_data_transform import transform_data, columns_for_model, get_word_count_df
from .backend_utils.predict_route_utils import load_model
from .backend_utils.search_route_utils import search_find_matches
from .backend_utils.new_manga_route_utils import find_latest_manga_recommendations
from .backend_utils.info_route_utils import get_predicted_titles_per_rating_graph


app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                df = pd.read_csv(file)
                title = df.iloc[:, 0].tolist()
        else:
            title = request.form['manga-title']
            title = title.split(',')

        df = scrape_manga(title)
        df = transform_data(df)
        df = get_word_count_df(df)

        # Load the pkl model
        model = load_model()

        df = columns_for_model(df)
        if df.shape[0] == 0:
            return render_template('error.html', error='No titles were found with the given input')

        titles = pd.DataFrame(df['Title'])
        titles = titles.applymap(lambda x: x.title())

        df.drop(['Title'], axis=1, inplace=True)

        prediction = model.predict(df)
        df_predictions = pd.DataFrame(prediction, columns = ['Predicted Rating'])
        prediction = titles.join(df_predictions)
        prediction['Predicted Rating'] = prediction['Predicted Rating'].fillna(1.0).astype(int)
        with connect_db() as con:
            cur = con.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS predicted_data (Title TEXT PRIMARY KEY, Predicted_Rating INT)")
            for _, row in prediction.iterrows():
                title = row['Title']
                predicted_rating = row['Predicted Rating']
                cur.execute("SELECT * FROM predicted_data WHERE Title=?", (title,))
                existing_row = cur.fetchone()
                if existing_row is None:
                    cur.execute("INSERT INTO predicted_data (Title, Predicted_Rating) VALUES (?, ?)", (title, predicted_rating))
                elif existing_row[1] != predicted_rating:
                    cur.execute("UPDATE predicted_data SET Predicted_Rating=? WHERE Title=?", (predicted_rating, title))
            con.commit()
        return render_template('prediction.html', prediction=prediction.to_html(index=False, classes='table table-striped'))
    except Exception as e:
        return render_template('error.html', error=e)

@app.route('/new_manga', methods=['GET'])
def new_manga():
    limit = int(request.args.get('limit', 40))
    titles = find_latest_manga_recommendations(limit)
    try:
        df = scrape_manga(titles)
        print('BEGINNING TO TRANSFORM DATA')
        df = transform_data(df)
        df = get_word_count_df(df)

        # Load the pkl model
        model = load_model()

        df = columns_for_model(df)
        if df.shape[0] == 0:
            return render_template('error.html')

        titles = pd.DataFrame(df['Title'])
        titles = titles.applymap(lambda x: x.title())

        df.drop(['Title'], axis=1, inplace=True)

        prediction = model.predict(df)
        df_predictions = pd.DataFrame(prediction, columns = ['Predicted Rating'])
        prediction = titles.join(df_predictions)
        prediction['Predicted Rating'] = prediction['Predicted Rating'].fillna(1.0).astype(int)
        with connect_db() as con:
            cur = con.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS predicted_data (Title TEXT PRIMARY KEY, Predicted_Rating INT)")
            for _, row in prediction.iterrows():
                title = row['Title']
                predicted_rating = row['Predicted Rating']
                cur.execute("SELECT * FROM predicted_data WHERE Title=?", (title,))
                existing_row = cur.fetchone()
                if existing_row is None:
                    cur.execute("INSERT INTO predicted_data (Title, Predicted_Rating) VALUES (?, ?)", (title, predicted_rating))
                elif existing_row[1] != predicted_rating:
                    cur.execute("UPDATE predicted_data SET Predicted_Rating=? WHERE Title=?", (predicted_rating, title))
            con.commit()
        return render_template('new_manga.html', new_manga_results=prediction.to_html(index=False, classes='table table-striped'))
    except Exception as e:
        return render_template('error.html', error=e)


@app.route('/search', methods=['POST'])
def search():
    manga_title = request.form['search']
    query = "SELECT DISTINCT Title, Rating, 'Train Dataset' as Source FROM model_data \
             UNION \
             SELECT DISTINCT Title, Predicted_Rating as Rating, 'Predicted Dataset' as Source FROM predicted_data"
    try:
        with connect_db() as con:
            cursor = con.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            df = pd.DataFrame(result, columns=['Title', 'Rating', 'Source'])
            df['Rating'] = df['Rating'].fillna(1.0).astype(int)
            matches = search_find_matches(result, manga_title)
            if matches:
                matches = sorted(matches, key=lambda x: x[3], reverse=True)
                matches = [(title, rating, source) for title, rating, source, _ in matches]
                df_filtered = df[df['Title'].isin([title for title, _, _ in matches])]
                search_html = render_template('found_search.html', search=df_filtered.to_html(index=False, classes='table table-striped'))
                return search_html
            else:
                search_html = render_template('no_search_found.html')
                return search_html
    except Exception as e:
        return render_template('error.html', error=e)

@app.route('/info', methods=['GET'])
def info():
    predicted_vs_rating = get_predicted_titles_per_rating_graph()
    return render_template('model_info.html', graph_image_data=predicted_vs_rating)
