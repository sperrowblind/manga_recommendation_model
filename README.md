# Manga Recommendation Model Application
This is the Manga Recommendation Model Application. I started this project after seeing the ratings that I give to manga on my reading list followed a normal distribution, and I wanted to see if I could attempt to create a predictive model to give me ideas of other manga that I may enjoy in the future.
This application works by creating a data set from a list of manga titles and scraping multiple websites for relevant data. Following the creation of the data set, model selection is performed to create a predictive model, where it is then saved. The current model is final_model_6 (It does not perform as well as final_model_4, more training is necessary in the future). A flask application was created to provide a frontend for interacting with this model. The features of this project allow for a user to predict ratings for manually entered titles or with a csv, search for newly released manga by genre and get predictions, and to see previous ratings given to predicted manga.

Users can train there own models if they create a spreadsheet consisting of titles, personal ratings, and a count of title characters (This will be done automatically in the future); however, a demo csv is provided for training a new model (demo_read_manga.csv). After a new model is created, be sure to update backend_utils/predict_route_utils/load_model() with the new model name, as well as add in a new version number to backend_utils/predict_route_utils/find_predictions(df, model).

## Startup
Open up your terminal

Clone the project

Python is required for this project so be sure to install it if not done already

Create a virtual environment with 
`python3 -m venv venv`

(I recommend using python3.9)

Enter the virtual environment with 
`source venv/bin/activate`

Run 
`pip install -r requirements.txt`

Run 
`source .env.sample`

## Running the Application
From this point, you can start the application with `flask run` (You may need to run `python manga_recommendation_clean_data.py` first, I tried to fix this so you don't need to)

Do not be alarmed if start up on the application takes a minute, the application will install the nltk stopwords package on initial startup and this can take a few seconds. This does not happen on consecutive start ups.

You will be brought to a screen with three buttons: Predict Rating, Find New Manga, and Search for Manga

Predict Rating allows a user to manually input titles separated by commas, or submit a csv file containing a column of titles (An example file is in the project called test_titles_predict_route.csv). The application will load the model after scraping for data on the titles and give predictive ratings based on the current model version.

Find New Manga allows a user to search for the newest released manga on Manganato. Users can search for 5, 10, or the whole first page of newest titles and generate ratings for each. Users can also select combinations of genres they wish to search for.

Search for Manga allows users to search for singular titles in the database that have already been predicted to see previous model version ratings.

On the home page, there is also a button in the top corner that will take users to the model info page. Here, I discuss different aspects of the project and how the pipeline works. This area needs some more work and additional graphs, but will give different metrics on the newest model version.

Users can also train their own models after creating their data set by running `python manga_recommendation_clean_data.py` and then running 'python manga_model.py'

Creating the data set should only take a few minutes depending on the size of the data set, model selection will take ~15 minutes.

## Possible Issues
There is a chance after startup that you will get a 403 error when attempting to go to the application, clear your cookies and this should fix the issue.
If there is an error in the Find New Manga route, I recommend trying to search for a smaller number of manga. There are currently issues with certain special characters that will cause issues for the application. Work is being done to fix this.
If you are training your own model and notice that not every title is in the scraped data sheet, I recommend running `python test_missing_data.py`, this file was created to check differences in csvs at different points of the scraping process.

## Future Work
I am going to revamp the model info page with more graphs and better information.
There are a few bugs I've noticed with certain special characters in titles that have been causing errors.
Clean up the database, it's a little random and contains some duplicates, the bugs which caused them should not be an issue any more but they're still there.
