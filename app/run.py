import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib as jb
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
df = pd.read_sql_table('DisasterResponseMessageData', engine)

# load model
model = jb.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals

    # genre overview
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # categories overview
    category_names = list(df.columns[4:])
    category_sums = df[category_names].sum().sort_values(ascending=False)

    # number of categories the messages are in
    cat_df = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    # get number of categories the messages are categorized into
    cat_number = cat_df.sum(axis=1)
    # get share of how many messages are categorized into how many categories
    cat_share = 100 * cat_number.value_counts() / len(cat_number)
    share = pd.DataFrame({'number_of_categories': cat_share.index, 'messages_share': cat_share})
    # bin the results
    share['bins'] = pd.cut(share['number_of_categories'],
                           bins=[-1, 0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
                           labels=['0 categories', '1 categories', '2 categories', '3 categories',
                                   '4 categories', '5 categories', '6-10 categories', '11-15 categories',
                                   '16-20 categories', '21-25 categories', '26-30 categories'])
    share_bins = share.groupby('bins').sum()['messages_share']

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker={'color': 'rgb(46, 108, 141)'}
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres Across Given Data',
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
                Bar(
                    y=category_sums,
                    x=category_sums.index,
                    marker={'color': 'rgb(87, 144, 175)'}
                )
            ],

            'layout': {
                'title': 'Message Classifications - Messages can be Assigned to Multiple Categories',
                'yaxis': {
                    'title': "Number of Messages in Category"
                },
                'xaxis': {
                    'title': "Message Categories",
                    'tickangle': 45,
                    'automargin': True
                }
            }
        },
        {
            'data': [{
                'values': share_bins,
                'labels': share_bins.index,
                'type': 'pie',
                'name': 'messages in...',
                'hoverinfo': 'label+name',
                'textinfo': 'None',
                'hole': .4,
                'marker': {
                    'colors': [
                        'rgb(56, 109, 138)',
                        'rgb(89, 162, 159)',
                        'rgb(106, 146, 167)',
                        'rgb(99, 153, 151)',
                        'rgb(155, 182, 197)',
                        'rgb(85, 92, 126)',
                        'rgb(205, 218, 226)',
                        'rgb(56, 61, 84)',
                        'rgb(108, 145, 143)',
                        'rgb(113, 122, 168)',
                        'rgb(123, 132, 131)'
                    ]
                },
            }
            ],
            'layout': {
                'title': 'Number of Categories a Message is Assigned to'
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()