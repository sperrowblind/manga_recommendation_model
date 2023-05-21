from featurewiz import featurewiz
import pandas as pd
from pandas import DataFrame
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV as GSV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import pickle

from keras import backend as K
from keras import layers, models, regularizers

from manga_pipeline.manga_scraper import connect_db
from manga_pipeline.manga_data_transform import get_word_count_df

## This file will be used to find the best model for use in recommendation

## There are 5 ways that predictors are chosen:
# Through use of featurewiz
# Through the creation of a heatmap and removing highly correlated variables
# Through use of lasso
# Through manual selection
# And using the original data frame

## There are 6 different types of models for each predictor method:
# decision tree
# random forest
# neural network
# logistic regression
# gradient boosting
# and through knn means

def get_data() -> DataFrame:
    with connect_db() as con:
        df = pd.read_sql_query("SELECT * FROM model_data", con)
        df.drop(['index'],axis=1,inplace=True)
    return df

def featurewiz_get_predictors(df: DataFrame) -> DataFrame:
    target = 'Rating'

    features, train = featurewiz(df, target, corr_limit=0.7, verbose=0,
        sep=",", header=0, test_data="", feature_engg="", category_encoders="")

    if 'Rating' not in features:
        features.append('Rating')
    if 'Title' not in features:
        features.append('Title')

    return df[features]

def heatmap_get_predictors(df: DataFrame) -> DataFrame:
    corr_matrix = df.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]

    if 'Rating' in to_drop:
        to_drop.remove('Rating')
    if 'Title' in to_drop:
        to_drop.remove('Title')

    new_df = df.drop(to_drop, axis=1)
    return new_df

def lasso_get_predictors(df: DataFrame) -> DataFrame:
    y = df['Rating']
    df_new = df.drop(['Rating', 'Title'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df_new, y, test_size=0.2, random_state=2340)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    selected_features = lasso.coef_ != 0
    selected_feature_indices = [i for i, x in enumerate(selected_features) if x]
    selected_feature_indices = df.columns[selected_feature_indices].tolist()
    selected_feature_indices.append('Rating')
    selected_feature_indices.append('Title')
    selected_df = df[selected_feature_indices]

    return selected_df


def best_decision_tree(x_train_decision: DataFrame
    , y_train_decision: DataFrame
    , x_test_decision: DataFrame
    , y_test_decision: DataFrame
    , feature_or_heatmap: str
    ):

    x_train_decision = x_train_decision.drop(['Title', 'Rating'], axis=1)
    x_test_titles = x_test_decision[['Title', 'Rating']]
    x_test_decision = x_test_decision.drop(['Title', 'Rating'], axis=1)

    print(f'Starting decision_tree for {feature_or_heatmap}')

    dtc = DecisionTreeClassifier(random_state=2340)
    param_grid = {
        'max_depth': [5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 7, 9, 11],
        'min_samples_leaf': [1, 2, 3, 4, 5]
    }

    grid_search_decision = GSV(dtc, param_grid, cv=5, n_jobs=-1)

    grid_search_decision.fit(x_train_decision, y_train_decision)

    print('Best hyperparameters:', grid_search_decision.best_params_)
    print('Best accuracy:', grid_search_decision.best_score_)

    best_model = grid_search_decision.best_estimator_

    y_pred = best_model.predict(x_test_decision)
    test_predictions = x_test_titles[['Title', 'Rating']]
    test_predictions.reset_index(inplace=True, drop=True)
    df_predictions = pd.DataFrame(y_pred, columns = ['predicted_rating'])
    df_predictions.reset_index()
    test_predictions = test_predictions.join(df_predictions)
    #test_predictions.to_csv(f'{feature_or_heatmap}_decision_tree_predictions.csv')

    accuracy = accuracy_score(y_test_decision, y_pred)
    print('Test accuracy:', accuracy)
    print('')

    return best_model

def best_random_forest(x_train_forest: DataFrame
    , y_train_forest: DataFrame
    , x_test_forest: DataFrame
    , y_test_forest: DataFrame
    , feature_or_heatmap: str
    ):

    x_train_forest = x_train_forest.drop(['Title', 'Rating'], axis=1)
    x_test_titles = x_test_forest[['Title', 'Rating']]
    x_test_forest = x_test_forest.drop(['Title', 'Rating'], axis=1)

    print(f'Starting random_forest for {feature_or_heatmap}')
    param_grid = {
        'n_estimators': [5, 10, 15],
        'max_depth': [5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 7, 9, 11],
        'min_samples_leaf': [1, 2, 3, 4, 5]
    }

    rfc = RandomForestClassifier(random_state=2340)

    grid_search_forest = GSV(rfc, param_grid, cv=5)

    grid_search_forest.fit(x_train_forest, y_train_forest)

    print('Best hyperparameters:', grid_search_forest.best_params_)
    print('Best accuracy:', grid_search_forest.best_score_)

    best_model = grid_search_forest.best_estimator_

    y_pred = best_model.predict(x_test_forest)
    test_predictions = x_test_titles[['Title', 'Rating']]
    test_predictions.reset_index(inplace=True, drop=True)
    df_predictions = pd.DataFrame(y_pred, columns = ['predicted_rating'])
    df_predictions.reset_index()
    test_predictions = test_predictions.join(df_predictions)
    #test_predictions.to_csv(f'{feature_or_heatmap}_random_forest_predictions.csv')

    accuracy = accuracy_score(y_test_forest, y_pred)
    print('Test accuracy:', accuracy)
    print('')

    return best_model

def custom_activation(x):
    # Set the output probability of the first neuron to 0
    x = K.concatenate([K.zeros_like(x[:, :1]), x[:, 1:]], axis=-1)
    # Normalize the remaining probabilities
    return K.softmax(x)

def best_nn(x_train_nn: DataFrame
    , y_train_nn: DataFrame
    , x_test_nn: DataFrame
    , y_test_nn: DataFrame
    , feature_or_heatmap: str
    ):

    x_train_nn = x_train_nn.drop(['Title', 'Rating'], axis=1)
    x_test_titles = x_test_nn[['Title', 'Rating']]
    x_test_nn = x_test_nn.drop(['Title', 'Rating'], axis=1)

    print(f'Starting neural_network for {feature_or_heatmap}')

    """model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_dim=x_train_nn.shape[1]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(11, activation=custom_activation)
    ])"""

    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=x_train_nn.shape[1], kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(11, activation=custom_activation)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train_nn, y_train_nn, epochs=100, batch_size=x_train_nn.shape[1], validation_data=(x_test_nn, y_test_nn), verbose=0)

    y_pred = model.predict(x_test_nn)

    y_pred_int = np.round(y_pred * 8)

    y_pred_max = np.argmax(y_pred_int, axis=1) + 1

    test_predictions = x_test_titles[['Title', 'Rating']]
    test_predictions.reset_index(inplace=True, drop=True)
    df_predictions = pd.DataFrame(y_pred_max)
    df_predictions.reset_index()
    test_predictions = test_predictions.join(df_predictions)
    #test_predictions.to_csv(f'{feature_or_heatmap}_nn_predictions.csv')

    accuracy = accuracy_score(y_test_nn, y_pred_max)
    print('Test accuracy:', accuracy)
    print('')

    return model

def best_logistic_regression(x_train_logistic: DataFrame
    , y_train_logistic: DataFrame
    , x_test_logistic: DataFrame
    , y_test_logistic: DataFrame
    , feature_or_heatmap: str
    ):

    x_train_logistic = x_train_logistic.drop(['Title', 'Rating'], axis=1)
    x_test_titles = x_test_logistic[['Title', 'Rating']]
    x_test_logistic = x_test_logistic.drop(['Title', 'Rating'], axis=1)

    print(f'Starting logistic regression for {feature_or_heatmap}')
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }

    model = LogisticRegression(max_iter=1000, random_state=2340)

    grid_search_logistic = GSV(model, param_grid, cv=5, n_jobs=-1)

    grid_search_logistic.fit(x_train_logistic, y_train_logistic)

    print('Best hyperparameters:', grid_search_logistic.best_params_)
    print('Best accuracy:', grid_search_logistic.best_score_)

    best_model = grid_search_logistic.best_estimator_

    y_pred = best_model.predict(x_test_logistic)
    test_predictions = x_test_titles[['Title', 'Rating']]
    test_predictions.reset_index(inplace=True, drop=True)
    df_predictions = pd.DataFrame(y_pred, columns = ['predicted_rating'])
    df_predictions.reset_index()
    test_predictions = test_predictions.join(df_predictions)
    #test_predictions.to_csv(f'{feature_or_heatmap}_logistic_predictions.csv')

    accuracy = accuracy_score(y_test_logistic, y_pred)
    print('Test accuracy:', accuracy)
    print('')

    return best_model

def best_gradient_boosted(x_train_gradient: DataFrame
    , y_train_gradient: DataFrame
    , x_test_gradient: DataFrame
    , y_test_gradient: DataFrame
    , feature_or_heatmap: str
    ):

    x_train_gradient = x_train_gradient.drop(['Title', 'Rating'], axis=1)
    x_test_titles = x_test_gradient[['Title', 'Rating']]
    x_test_gradient = x_test_gradient.drop(['Title', 'Rating'], axis=1)

    print(f'Starting gradient_boosting for {feature_or_heatmap}')
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [2, 3, 4]
    }

    model = GradientBoostingClassifier(random_state=2340)

    grid_search_gradient = GSV(model, param_grid, cv=5, n_jobs=-1)

    grid_search_gradient.fit(x_train_gradient, y_train_gradient)

    print('Best hyperparameters:', grid_search_gradient.best_params_)
    print('Best accuracy:', grid_search_gradient.best_score_)

    best_model = grid_search_gradient.best_estimator_

    y_pred = best_model.predict(x_test_gradient)
    test_predictions = x_test_titles[['Title', 'Rating']]
    test_predictions.reset_index(inplace=True, drop=True)
    df_predictions = pd.DataFrame(y_pred, columns = ['predicted_rating'])
    df_predictions.reset_index()
    test_predictions = test_predictions.join(df_predictions)
    #test_predictions.to_csv(f'{feature_or_heatmap}_gradient_predictions.csv')

    accuracy = accuracy_score(y_test_gradient, y_pred)
    print('Test accuracy:', accuracy)
    print('')

    return best_model

def best_svc(x_train_svc: DataFrame
    , y_train_svc: DataFrame
    , x_test_svc: DataFrame
    , y_test_svc: DataFrame
    , feature_or_heatmap: str
    ):

    x_train_svc = x_train_svc.drop(['Title', 'Rating'], axis=1)
    x_test_titles = x_test_svc[['Title', 'Rating']]
    x_test_svc = x_test_svc.drop(['Title', 'Rating'], axis=1)

    print(f'Starting SCV for {feature_or_heatmap}')
    param_grid = {
        'C': [1],
        'kernel': ['linear'],
        'gamma': ['scale']
    }

    model = SVC(random_state=2340)

    grid_search_svc = GSV(model, param_grid, cv=5)

    grid_search_svc.fit(x_train_svc, y_train_svc)

    print('Best hyperparameters:', grid_search_svc.best_params_)
    print('Best accuracy:', grid_search_svc.best_score_)

    best_model = grid_search_svc.best_estimator_

    y_pred = best_model.predict(x_test_svc)
    test_predictions = x_test_titles[['Title', 'Rating']]
    test_predictions.reset_index(inplace=True, drop=True)
    df_predictions = pd.DataFrame(y_pred, columns = ['predicted_rating'])
    df_predictions.reset_index()
    test_predictions = test_predictions.join(df_predictions)
    #test_predictions.to_csv(f'{feature_or_heatmap}_svc_predictions.csv')

    accuracy = accuracy_score(y_test_svc, y_pred)
    print('Test accuracy:', accuracy)
    print('')

    return best_model

def best_knn(x_train_knn: DataFrame
    , y_train_knn: DataFrame
    , x_test_knn: DataFrame
    , y_test_knn: DataFrame
    , feature_or_heatmap: str
    ):

    x_train_knn = x_train_knn.drop(['Title', 'Rating'], axis=1)
    x_test_titles = x_test_knn[['Title', 'Rating']]
    x_test_knn = x_test_knn.drop(['Title', 'Rating'], axis=1)

    print(f'Starting KNN for {feature_or_heatmap}')
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

    model = KNeighborsClassifier()

    grid_search_knn = GSV(model, param_grid, cv=5)

    grid_search_knn.fit(x_train_knn, y_train_knn)

    print('Best hyperparameters:', grid_search_knn.best_params_)
    print('Best accuracy:', grid_search_knn.best_score_)

    best_model = grid_search_knn.best_estimator_

    y_pred = best_model.predict(x_test_knn)
    test_predictions = x_test_titles[['Title', 'Rating']]
    test_predictions.reset_index(inplace=True, drop=True)
    df_predictions = pd.DataFrame(y_pred, columns = ['predicted_rating'])
    df_predictions.reset_index()
    test_predictions = test_predictions.join(df_predictions)
    #test_predictions.to_csv(f'{feature_or_heatmap}_knn_predictions.csv')

    accuracy = accuracy_score(y_test_knn, y_pred)
    print('Test accuracy:', accuracy)
    print('')

    return best_model

def find_best_model(df: DataFrame, which_frame: str):

    x = df
    y = df['Rating']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2340)

    try:
        decision_tree = best_decision_tree(x_train, y_train, x_test, y_test, which_frame)
    except:
        decision_tree = None
    try:
        random_forest = best_random_forest(x_train, y_train, x_test, y_test, which_frame)
    except:
        random_forest = None
    try:
        neural_network = best_nn(x_train, y_train, x_test, y_test, which_frame)
    except:
        neural_network = None
    try:
        logistic = best_logistic_regression(x_train, y_train, x_test, y_test, which_frame)
    except:
        logistic = None
    try:
        gradient = best_gradient_boosted(x_train, y_train, x_test, y_test, which_frame)
    except:
        gradient = None
    #svc = best_svc(x_train, y_train, x_test, y_test, which_frame)
    try:
        knn = best_knn(x_train, y_train, x_test, y_test, which_frame)
    except:
        knn = None

    model_dict = {'decision': decision_tree, 'forest': random_forest, 'nn': neural_network, 'logistic': logistic, 'gradient': gradient, 'knn': knn}
    performance_dict = {}
    x_test = x_test.drop(['Title', 'Rating'], axis=1)
    for model_name, model in model_dict.items():
        if model == None:
            continue
        y_pred = model.predict(x_test)
        if model_name == 'nn':
            y_pred_int = np.round(y_pred * 8)
            y_pred_max = np.argmax(y_pred_int, axis=1) + 1
            score = accuracy_score(y_test, y_pred_max)
        else:
            score = accuracy_score(y_test, y_pred)
        performance_dict[model_name] = score

    best_model_name = max(performance_dict, key=performance_dict.get)
    print(f'best model for {which_frame}: {best_model_name}')
    best_model = model_dict[best_model_name]
    return best_model

if __name__ == '__main__':

    df = get_data()

    word_df = df[['Title', 'description', 'Rating']]
    word_df = get_word_count_df(word_df)

    df_with_nltk = get_word_count_df(df)

    df.drop(['description'], axis=1, inplace=True)

    featurewiz_df = featurewiz_get_predictors(df)
    heatmap_df = heatmap_get_predictors(df)
    lasso_df = lasso_get_predictors(df)

    word_featurewiz_df = featurewiz_get_predictors(word_df)
    word_heatmap_df = heatmap_get_predictors(word_df)
    word_lasso_df = lasso_get_predictors(word_df)

    df_nltk_featurewiz = featurewiz_get_predictors(df_with_nltk)
    df_nltk_heatmap = heatmap_get_predictors(df_with_nltk)

    word_model = find_best_model(word_df, 'word_nltk')
    word_featurewiz_model = find_best_model(word_featurewiz_df, 'word_featurewiz')
    word_heatmap_model = find_best_model(word_heatmap_df, 'word_heatmap')
    word_lasso_model = find_best_model(word_lasso_df, 'word_lasso')
    nltk_featurewiz_model = find_best_model(df_nltk_featurewiz, 'nltk_featurewiz')
    nltk_heatmap_model = find_best_model(df_nltk_heatmap, 'nltk_heatmap')
    #nltk_lasso_model = find_best_model(df_nltk_lasso, 'nltk_lasso')
    featurewiz_model = find_best_model(featurewiz_df, 'feature')
    heatmap_model = find_best_model(heatmap_df, 'heatmap')
    lasso_model = find_best_model(lasso_df, 'lasso')
    original_model = find_best_model(df, 'original')

    x = df_with_nltk
    y = df_with_nltk['Rating']
    x = x.drop(['Rating', 'Title'], axis=1)

    models = [featurewiz_model, heatmap_model, lasso_model, original_model, word_model, word_featurewiz_model, word_heatmap_model, word_lasso_model, nltk_featurewiz_model, nltk_heatmap_model]
    dfs =    [featurewiz_df,    heatmap_df,    lasso_df,    df,             word_df,    word_featurewiz_df,    word_heatmap_df,    word_lasso_df,    df_nltk_featurewiz,    df_nltk_heatmap]
    for model in dfs:
        model.drop(['Rating', 'Title'], axis=1, inplace=True)

    best_model = None
    best_r2 = -float('inf')
    best_model_df = None

    if not os.path.exists('csvs'):
        os.makedirs('csvs')

    for model in range(0,len(models)):
        x_train, x_test, y_train, y_test = train_test_split(x[dfs[model].columns], y, test_size=0.1, random_state=2340)
        y_pred = models[model].predict(x_test)

        df_predictions = pd.DataFrame(y_pred, columns = ['predicted_rating'])
        df_predictions.to_csv(os.path.join('csvs', f'{str(models[model])}_predictions.csv'))

        r2 = r2_score(y_test, y_pred)
        print(r2)

        if r2 > best_r2:
            print(f'better model at index {model} which is a {models[model]}, with an r2 of {r2}')
            best_model = models[model]
            best_model_df = dfs[model]
            best_r2 = r2

    best_model_df.to_csv(os.path.join('csvs', 'model_df.csv'))
    with open('final_model_2.pkl', 'wb') as f:
        pickle.dump(best_model, f)
