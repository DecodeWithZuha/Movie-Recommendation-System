from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

movies_data = pd.read_excel('/Users/apple/Desktop/chatbot/movies.xlsx')

movies_data.head()

movies_data.shape

selected_features = ['genres', 'keywords', 'original_language', 'original_title', 'production_companies',
                     'title', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].astype(str)
    movies_data[feature] = movies_data[feature].fillna(' ')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data[
    'original_language'] + ' ' + movies_data['original_title'] + ' ' + movies_data[
                        'production_companies'] + ' ' + movies_data['title'] + ' ' + movies_data['cast'] + ' ' + \
                     movies_data['director']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    action = req['queryResult']['action']
    parameters = req['queryResult']['parameters']

    if action == 'suggestMovie':
        movie_name = parameters['movieName']

        list_of_all_titles = movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        close_match = find_close_match[0]
        index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]

        similarity_score = list(enumerate(similarity[index_of_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        suggested_movies = []
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            suggested_movies.append(title_from_index)

        response = "Movies suggested for you are: " + ', '.join(suggested_movies[:10])
    else:
        response = "Unknown action."

    res = {'fulfillmentText': response}
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True)
