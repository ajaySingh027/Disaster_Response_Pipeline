# import libraries
import pandas as pd
import numpy as np
import sys

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' +database_filepath)
    df = pd.read_sql_table('DisasterRespTabl', engine)
    X = df['message'].values
    """ Y_values = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
            'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter',
            'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
            'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers',
            'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 
            'other_weather', 'direct_report'] """
    # Y = df[Y_values].values
    Y = df.columns[4:].values
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", "", text.lower())
    
    #tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens



def build_model():
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #For reference:
    ## RandomForestClassifier(random_state=100)
    ## OneVsRestClassifier(LinearSVC())

     # create grid search object
    parameters = {
                    'clf__min_samples_split': [2, 3, 4],
                    'clf__estimator__learning_rate': [0.001, 0.01, 0.1],
                    'tfidf__smooth_idf': [True, False]
                    }
    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.prodict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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