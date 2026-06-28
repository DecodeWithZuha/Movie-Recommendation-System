# this movie recommendation system will extract movies on the basis of
# genres(eg: action, comedy, fiction, etc)✅
# release-date(eg: which movies were released on 20th oct 2004, etc)
# title(eg: recommend movies like avatar, kal ho na ho, my name is khan)✅
# language(en, hi, fr, ko)
# keywords()
# budget()
# popularity(eg: what is the most popular movie of 2020)
# production companies()
# production countries()
# revenue()
# tagline()
# cast(eg:salman khan , johnny depp, etc)
# director()


#Kuch b user sy input lena hy... uski similarities dhoondhni hain aur list deni.. by default 5 batyega lkn agr user ny ziyada poochi toh ziyada b dega



from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

movies_data = pd.read_csv('movies.csv')

#movies_data.head() #prints first 5 rows of dataset

#movies_data.shape # prints the total no. of rows and columns of dataset which is (4803, 24)

#Feature /column selection for recommendation
selected_features =['genres' , 'keywords', 'original_language' , 'original_title', 'production_companies',
                    'title', 'cast', 'director'] # 8... title and orignal title may be similar

#print(selected_features)

# cleaning data... filling the empty spaces/null values with null strings for only the selected features
for feature in selected_features:
    movies_data[feature] = movies_data[feature].astype(str)
    movies_data[feature] = movies_data[feature].fillna(' ')

# comibining the selected features and turning then into strings first
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['original_language']+' '+movies_data['original_title']+' '+movies_data['production_companies']+' '+movies_data['title']+' '+movies_data['cast']+' '+movies_data['director']

#print(combined_features)

# Converting the textdata into feature vector
vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features) #converts data into numbers

#print(feature_vectors)

#--------------------COSINE SIMILARITY ALGORITHM-----------------
# Getting the similarity scores
similarity = cosine_similarity(feature_vectors)

#print(similarity)

#print(similarity.shape) #(4803,4803) 1st is index no. and second is similarities..... aik movie poori 4803 movies sy compare ki jyegi.. thats why equal rows and columns hain

#--------------------GETTING INPUT FROM THE USER------------------
movie_name = input('Enter the movie name: ')



#creating a list of all the movie names

list_of_all_titles = movies_data['title'].tolist()
#print(list_of_all_titles)

#finding the close match to the movie name given by the user
#to get best match we are using difflib

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
#print(find_close_match)

# giving 3 for iron man but we need 1 so
close_match = find_close_match[0]
#print(close_match)

#---------------------Finding Index of Movie------------------------
index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]
#print(index_of_movie)

#------Getting  A list of similar movies based on index no. --------
similarity_score =  list(enumerate(similarity[index_of_movie])) # similar movies will have high similarity score.... enumerate is used to carry out a loop in a list

#print(similarity_score) #eg (4787, 0.04970329462060189) 1st shows the index of the movie.. and the 2nd shows the similarty score with the user entered movie


#Now sorting the data from highest to lowest so that we could recommend similar movies based on similarity score

sorted_similar_movies = sorted(similarity_score, key= lambda x:x[1], reverse=True) #sorted will arrange in ascending order... so reverse is true to get data in decending order... x[1] means 2nd element in ordered pair eg.(3, 90).. x[0]=3 and x[1]=90 where x is whole ordered pair (3, 90)
#print(sorted_similar_movies)

#printing the name of similar movies based on index
print('Movies suggested for you are:')
i=1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index== index]['title'].values[0]
    if(i<=10):
        print(i,'-',title_from_index)
        i+=1
