# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# Importing the dataset
small_links = pd.read_csv('./data_sets/nlp/movies/links_small.csv')
movie_dataset = pd. read_csv('./data_sets/nlp/movies/movies_metadata.csv')

# filling mulls
small_links = small_links[small_links['tmdbId'].notnull()]['tmdbId'].astype('int')

# dropping bad formatted data
movie_dataset = movie_dataset.drop([19730, 29503, 35587])

movie_dataset['id'] = movie_dataset['id'].astype('int')
small_movie_dataset = movie_dataset[movie_dataset['id'].isin(small_links)]
small_movie_dataset.shape

# base on description
small_movie_dataset['tagline'] = small_movie_dataset['tagline'].fillna('')
small_movie_dataset['description'] = small_movie_dataset['overview'] + small_movie_dataset['tagline']
small_movie_dataset['description'] = small_movie_dataset['description'].fillna('')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(small_movie_dataset['description'])

tfidf_matrix.shape

"""
Cosine Similarity
I will be using the Cosine Similarity to calculate a numeric quantity that denotes the similarity between
two movies. Mathematically, it is defined as follows:

cosine(x,y)=x.y⊺/||x||.||y|| 
Since we have used the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine
Similarity Score. Therefore, we will use sklearn's linear_kernel instead of cosine_similarities since it
is much faster.
"""

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

small_movie_dataset.loc[1]

# the index is reseted to par with the 'cosine_sim' list
small_movie_dataset = small_movie_dataset.reset_index()

titles = small_movie_dataset['title']
indices = pd.Series(small_movie_dataset.index, index=small_movie_dataset['title'])


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


""" """

get_recommendations('The Godfather').head(10)