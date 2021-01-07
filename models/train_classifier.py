###################
# ML Pipeline - to define and train model on the given dataset and save the model as a pickle file, execute:
# 'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'

# GridSearchCV can be used by setting 'grid=True' in build_model and evaluate_model
# script functionality can be tested on a smaller sample of the dataset by setting 'reduced_dataset=True' in load_data
###################

import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
import pickle

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath, reduced_dataset=False):
    '''
    loads pre-processed and cleaned data from a SQLite database
    :param database_filepath (str): path to the SQLite database
    :return:
        X (array): input vector with text messages
        Y (array): output vector with message categories
        category_names (list): labels of the output message categories
    '''
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponseMessageData', engine)

    # test basic functionality on a smaller dataset
    if reduced_dataset:
        print('using reduced dataset for testing basic script functionality')
        df = df.sample(n=2000)

    # get input/output vectors and output vector labels
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    '''
    process text input in four steps: normalization, tokenization, removing stop words, lemmatization
    :param text (str): input text
    :return: clean_tokens (list): list of clean tokens from text
    '''
    # normalize and remove special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # remove english stop words from tokenized list of words
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # lemmatize remaining tokens including verbs
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(t).strip() for t in tokens]
    lemmed = [lemmatizer.lemmatize(t, pos='v').strip() for t in lemmed]

    clean_tokens = lemmed

    return clean_tokens


def build_model(grid=False):
    '''
    builds machine learning pipeline to classify messages into given categories
    :param grid (bool): whether to use GridSearchCV for parameter optimization
    :return: model: ML pipeline
    '''
    # build ML pipeline with MultiOutputClassifier and tokenize function
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    # show pipeline parameters for optimization
    # print(pipeline.get_params())

    # use grid search for parameter optimization
    if grid:
        parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'tfidf__use_idf': (True, False),
            'clf__estimator__max_features': ('auto', 'sqrt'),
            'clf__estimator__n_estimators': [10, 20]
        }
        model = GridSearchCV(pipeline, param_grid=parameters, cv=3)
        print('Model built using GridSearchCV for optimization')
    else:
        pipe_params = {'clf__estimator__max_features': 'auto',
                       'clf__estimator__n_estimators': 10,
                       'tfidf__use_idf': False,
                       'vect__ngram_range': (1, 2)}
        model = pipeline.set_params(**pipe_params)
        print('Model built using pipeline (tuned parameters selected from former optimization)')

    return model


def evaluate_model(model, X_test, Y_test, category_names, grid=False):
    '''
    runs the prediction on a test dataset and evaluates it against the target values
    :param model: trained ML model used for prediction
    :param X_test: test dataset
    :param Y_test: target values for test dataset
    :param category_names (list): labels of the output message categories
    :return:
    '''
    # make prediction
    Y_pred = model.predict(X_test)

    # display parameters determined by GridSearch
    if grid:
        print(f'optimized parameters determined by Gridsearch:\n{model.best_params_}')
    else:
        print(f'pipeline parameters used:\n{model.get_params()}')

    # evaluate prediction against Y_test
    show_report(category_names, Y_test, Y_pred)


def show_report(category_names, Y_test, Y_pred):
    '''
    evaluates predicted values against target values using sklearn's classification_report
    :param category_names (list): labels of the output message categories
    :param Y_test: target values for test dataset
    :param Y_pred: predicted values for test dataset
    :return: prints classification report for each column in Y
    '''
    # show classification report
    for i in range(len(category_names)):
        print ('results for category ({}, {}):'.format(i, category_names[i]))
        # for sklearn < v.0.20.0:
        rep = classification_report(y_true=np.array(Y_test)[:, i], y_pred=np.array(Y_pred)[:, i])
        # for sklearn > v.0.20.0:
        # rep = classification_report(y_true=Y_test[:, i], y_pred=Y_pred[:, i], output_dict=True)
        print(rep)


def save_model(model, model_filepath):
    '''
    save trained model as pickle file
    :param model: trained ML model
    :param model_filepath (str): name and path for the model to be saved (i.e.path/to/modelname.pkl)
    :return: None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath, reduced_dataset=False)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(grid=False)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names, grid=False)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()