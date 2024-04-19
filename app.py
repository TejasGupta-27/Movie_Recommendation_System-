import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, KNNBasic, CoClustering, BaselineOnly
from surprise import Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data from CSV files
movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

# Create Surprise Reader object
reader = Reader(rating_scale=(1, 5))

# Create Surprise Dataset
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset = data.build_full_trainset()

# Initialize collaborative filtering models
svd = SVD()
knn = KNNBasic()
baseline = BaselineOnly()
co_clustering = CoClustering()

# Train collaborative filtering models
svd.fit(trainset)
knn.fit(trainset)
baseline.fit(trainset)
co_clustering.fit(trainset)

# Merge movie genres into a single string
movies_df['genres'] = movies_df['genres'].fillna('')
movies_df['genres'] = movies_df['genres'].apply(lambda x: ' '.join(x.split('|')))

# Initialize TF-IDF Vectorizer for content-based recommendation
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

# Compute cosine similarity matrix for content-based recommendation
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get collaborative filtering recommendations for a user
def get_collaborative_filtering_recommendations(user_id):
    watched_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
    all_movie_ids = set(movies_df['movieId'])
    unwatched_movies = list(all_movie_ids - watched_movies)

    predictions = []
    for model in [svd, knn, baseline, co_clustering]:
        model_predictions = [(user_id, movie_id, model.predict(user_id, movie_id).est) for movie_id in unwatched_movies]
        predictions.extend(model_predictions)

    combined_preds = {}
    for user_id, movie_id, est in predictions:
        if movie_id not in combined_preds:
            combined_preds[movie_id] = [est]
        else:
            combined_preds[movie_id].append(est)

    for movie_id in combined_preds:
        combined_preds[movie_id] = np.mean(combined_preds[movie_id])

    sorted_recommendations = sorted(combined_preds.items(), key=lambda x: x[1], reverse=True)
    top_10_recommendations = sorted_recommendations[:10]

    return top_10_recommendations

# Function to get content-based recommendations for a movie
def get_content_based_recommendations(movie_id, cosine_sim=cosine_sim):
    idx = movies_df[movies_df['movieId'] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

# Streamlit UI
st.title("Movie Recommendation System")

# Option to select recommendation type
recommendation_type = st.selectbox("Select Recommendation Type", ("Collaborative Filtering", "Content-Based"))

if recommendation_type == "Collaborative Filtering":
    # Collaborative Filtering
    user_id = st.number_input("Enter User ID", min_value=1, max_value=610, value=1, step=1)

    if st.button("Get Recommendations"):
        recommendations = get_collaborative_filtering_recommendations(user_id)
        st.write(f"Top 10 Movie Recommendations for User {user_id}:")
        for movie_id, _ in recommendations:
            movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
            st.write(f"- {movie_title}")

elif recommendation_type == "Content-Based":
    # Content-Based
    movie_id = st.number_input("Enter Movie ID", min_value=1, max_value=movies_df['movieId'].max(), value=1, step=1)

    if st.button("Get Recommendations"):
        similar_movies = get_content_based_recommendations(movie_id)
        st.write(f"Movies Similar to Movie with ID: {movie_id}")
        for movie_title in similar_movies:
            st.write(f"- {movie_title}")
