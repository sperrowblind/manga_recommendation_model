from flask import Flask, request, render_template
import pandas as pd

from .manga_pipeline.manga_scraper import connect_db, scrape_manga
from .manga_pipeline.manga_data_transform import transform_data, columns_for_model, get_word_count_df
from .backend_utils.predict_route_utils import load_model, find_predictions
from .backend_utils.search_route_utils import search_find_matches
from .backend_utils.new_manga_route_utils import find_latest_manga_recommendations
from .backend_utils.info_route_utils import get_predicted_titles_per_rating_graph, get_metrics_table


app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
async def predict():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                df = pd.read_csv(file)
                titles = df.iloc[:, 0].tolist()
        else:
            titles = request.form['manga-title']
            titles = titles.split(',')

        df = await scrape_manga(titles)
        df = transform_data(df)
        df = get_word_count_df(df)

        # Load the pkl model
        model = load_model()

        df = columns_for_model(df)
        if df.shape[0] == 0:
            return render_template('error.html', error='No titles were found with the given input')

        prediction_table = find_predictions(df, model)

        return render_template('prediction.html', prediction=prediction_table)
    except Exception as e:
        return render_template('error.html', error=e)

@app.route('/new_manga', methods=['GET'])
async def new_manga():
    limit = int(request.args.get('limit', 40))

    genres = request.args.getlist('genre')

    titles = find_latest_manga_recommendations(limit, genres)

    if len(titles) == 0:
        return render_template('error.html', error="No titles were found")
    try:
        df = await scrape_manga(titles)
        print('BEGINNING TO TRANSFORM DATA')
        df = transform_data(df)
        df = get_word_count_df(df)

        # Load the pkl model
        model = load_model()

       # df_columns = set(df.columns)

        df = columns_for_model(df)

        if df.shape[0] == 0:
            return render_template('error.html')

        #model_columns = set(df.columns)

        # Compare the column names
        #missing_columns = model_columns - df_columns
        #extra_columns = df_columns - model_columns

        #print("Missing columns in df:", missing_columns)
        #print("Extra columns in df:", extra_columns)

        prediction_table = find_predictions(df, model)

        return render_template('new_manga.html', new_manga_results=prediction_table)
    except Exception as e:
        return render_template('error.html', error=e)


@app.route('/search', methods=['POST'])
def search():
    manga_title = request.form['search']
    query = "SELECT DISTINCT Title, Rating, 'Train Dataset' as Source, 'N/A' as Model_Version FROM model_data \
             UNION \
             SELECT DISTINCT Title, Predicted_Rating as Rating, 'Predicted Dataset' as Source, Model_Version FROM predicted_data"
    try:
        with connect_db() as con:
            cursor = con.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            df = pd.DataFrame(result, columns=['Title', 'Rating', 'Source', 'Model_Version'])
            df['Rating'] = df['Rating'].fillna(1.0).astype(int)
            matches = search_find_matches(result, manga_title)
            if matches:
                matches = sorted(matches, key=lambda x: x[4], reverse=True)
                matches = [(title, rating, source, model_version, _) for title, rating, source, model_version, _ in matches]
                df_filtered = df[df['Title'].isin([title for title, _, _ , _, _,in matches])]
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
    metrics = get_metrics_table()
    return render_template('model_info.html', graph_image_data=predicted_vs_rating, model_metrics=metrics)
