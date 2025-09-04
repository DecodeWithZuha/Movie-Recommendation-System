import numpy as np
import pandas as pd
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies_data = pd.read_excel('/Users/apple/Desktop/chatbot/movies.xlsx')

selected_features =['budget' ,'genres' , 'keywords', 'original_language' , 'original_title', 'popularity', 'production_companies',
                    'production_countries', 'release_date', 'revenue', 'tagline', 'title', 'cast', 'director']
    
for feature in selected_features:
    movies_data[feature] = movies_data[feature].astype(str)
    movies_data[feature] = movies_data[feature].fillna(' ')

combined_features = movies_data['budget']+' '+movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['original_language']+' '+movies_data['original_title']+' '+movies_data['popularity']+' '+movies_data['production_companies']+' '+movies_data['production_countries']+' '+movies_data['release_date']+' '+movies_data['revenue']+' '+movies_data['tagline']+' '+movies_data['title']+' '+movies_data['cast']+' '+movies_data['director']

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

