# manga_rating_prediction
I keep a spreadsheet of manga that I read and assign a rating on a 1-10 scale. This model is used to predict what kind of rating I would give a manga before reading it

This is a major work in progress but the goal is that I will be able to scrape new manga for the variables in the model and predict if I would actually enjoy reading it.

This project was started after realizing that my rating system appeared to follow a normal distribution

This current version is chosen through automated selection through a variety of different feature selection and model selection methods. The final model is chosen based on the best accuracy score and r squared

Presently some future ideas are:
Automate model selection : DONE
Select new variables for use in model (Most likely done through scraping): In progress
Create a scraper based on what's already created to scrape data for manga I have not read : NEEDS TESTING
Create a sqllite database so that any data that has already been rated can be retrieved again unless the model has been retrained : DONE
Revamp scraper to get more data from wikipedia : DONE
Revamp scraper to gett all author information : NO

Updated Ideas:
Create react application to host model: IN progress
revamp wiki scraper again, some areas not the best though it's a lot better: Could still be better
add ensembles for model selection

I am currently not happy with the model, but this will hopefully change in the future.

There is a bug where the predict endpoint would fail if the model selected is a neural network; however, i do not see the nn ever being the best
