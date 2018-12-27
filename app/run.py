import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterTable', engine)

# load model
model = joblib.load("../models/DisPipeline_model.pic")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Calculate message count by genre with related status
    genre_related = df[df['related']==1].groupby('genre').count()['message']
    genre_not_related = df[df['related']==0].groupby('genre').count()['message']
    genre_names = list(genre_related.index)
    
    # Each Message's category wise count
    categories = df.columns[4:].tolist()
    messgs_recv = df.iloc[:, 4:].sum().tolist()

    # Messages containing weather related text graphs
    df_weather = df[['weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold','other_weather']].mean()
    weather_mean = df_weather
    weather_names = df_weather.index


    # create visuals
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_related,
                    name='Related'
                ),

                Bar(
                    x=genre_names,
                    y=genre_not_related,
                    name='Not Related'
                )
            ],

            'layout': {
                'title': 'Distribution of Messages by Genres with Related status',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },


        {
            'data': [
                Histogram(
                    x = categories,
                    y = messgs_recv,
                    histfunc='sum',
                    marker=dict(color='green')
                )
            ],

            'layout': {
                'title': 'Frequency distribution of Categories of Messages',
                'yaxis': {
                    'title': "Frequency of Message Category"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },


        {
            'data': [
                Bar(
                    x = weather_names,
                    y = weather_mean,
                    marker= dict(color='dimgray')
                )
            ],

            'layout': {
                'title': 'Weather related Messages received',
                'yaxis': {
                    'title': "% of all messages"
                },
                'xaxis': {
                    'title': "Type of Message"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='localhost', port=3001, debug=True)


if __name__ == '__main__':
    main()