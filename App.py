#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import string

# Load the data
data = pd.read_csv("netflixData.csv")

# Data cleaning and preprocessing
data = data[["Title", "Description", "Genres", "Content Type"]]
data.dropna(inplace=True)

# Using NLTK's SnowballStemmer and stopwords
stemmer = nltk.SnowballStemmer("english")
stopwords_set = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopwords_set]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    return " ".join(text)

data["Genres"] = data["Genres"].apply(clean_text)
data["Description"] = data["Description"].apply(clean_text)

# Creating the TF-IDF matrix for the cleaned "Genres" column
tfidf_vectorizer = text.TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(data["Genres"])

# Calculating cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Creating a Streamlit app
st.title("Netflix Movie Recommendation")

# Add an input widget for user to enter movie title
movie_title = st.text_input("Enter a movie title:")

# Function for movie recommendation
def netflix_recommendation(title, similarity=similarity_matrix):
    if title not in data["Title"].values:
        return "Title not found in the dataset."

    indices = pd.Series(data.index, index=data['Title']).drop_duplicates()
    idx = indices[title]
    similarity_scores = list(enumerate(similarity[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]  # Excluding the title itself
    movie_indices = [i[0] for i in similarity_scores]
    return data['Title'].iloc[movie_indices]

# Display movie recommendations if a title is entered
if movie_title:
    recommendations = netflix_recommendation(movie_title)
    st.header("Recommended Movies:")
    for idx, rec_movie in enumerate(recommendations, start=1):
        st.write(f"{idx}. {rec_movie}")

