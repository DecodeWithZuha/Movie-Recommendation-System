import numpy as np
import pandas as pd
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies_data = pd.read_csv('movies.csv')

selected_features =['budget' ,'genres' , 'keywords', 'original_language' , 'original_title', 'popularity', 'production_companies',
                    'production_countries', 'release_date', 'revenue', 'tagline', 'title', 'cast', 'director']
    
for feature in selected_features:
    movies_data[feature] = movies_data[feature].astype(str)
    movies_data[feature] = movies_data[feature].fillna(' ')

combined_features = movies_data['budget']+' '+movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['original_language']+' '+movies_data['original_title']+' '+movies_data['popularity']+' '+movies_data['production_companies']+' '+movies_data['production_countries']+' '+movies_data['release_date']+' '+movies_data['revenue']+' '+movies_data['tagline']+' '+movies_data['title']+' '+movies_data['cast']+' '+movies_data['director']

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

#------------------------------------------------------------------------------------------------------------------

movie_genres = input('Enter the genre: ')

list_of_all_genres = movies_data['genres'].tolist()

find_close_match = difflib.get_close_matches(movie_genres, list_of_all_genres)
#print(find_close_match)

close_match = find_close_match[0]
#print(close_match)

index_of_movie = movies_data[movies_data.genres == close_match]['index'].values[0]
#print(index_of_movie)

similarity_score =  list(enumerate(similarity[index_of_movie]))

sorted_similar_movies = sorted(similarity_score, key= lambda x:x[1], reverse=True)

print('Movies suggested for you are:')
i=1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index== index]['title'].values[0]
    if(i<=10):
        print(i,'-',title_from_index)
        i+=1
