import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from PIL import Image

from ..manga_pipeline.manga_data_transform import connect_db



def get_predicted_titles_per_rating_graph():
    with connect_db() as con:
        df = pd.read_sql_query("SELECT Predicted_Rating AS rating, count(*) AS count FROM predicted_data GROUP BY rating", con)
        print(df)

        plt.switch_backend('Agg')
        fig, ax = plt.subplots()
        ax.barh(df['rating'], df['count'], color=['#7D26CD'])
        ax.set_xlabel('Rating')
        ax.set_ylabel('Number of Predicted Titles')
        ax.set_title('Number of Predicted Titles with Each Rating')
        ax.set_facecolor("#26428b")

        buffer = io.BytesIO()
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        graph_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())

        buffer = io.BytesIO()
        graph_image.save(buffer, format='PNG')
        graph_image_data = base64.b64encode(buffer.getvalue()).decode()

        return graph_image_data

def get_metrics_table():
    with connect_db() as conn:

        query = "SELECT metric, value, insert_date FROM model_metrics"
        df = pd.read_sql_query(query, conn)
        html_table = df.to_html(index=False, justify='center')
        return html_table

