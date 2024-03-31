# -*- coding: utf-8 -*-
"""project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CQJ9pqhIkbiABh70ViP4Cx6NNzweQq7i
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the datasets
links = pd.read_csv('links.csv')
movies = pd.read_csv('movies.csv')

tags = pd.read_csv('tags.csv')

#read rating file
ratings = pd.read_csv('ratings.csv')

ratings.head()

final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.head()

final_dataset.fillna(0,inplace=True)
final_dataset.head()

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

f,ax = plt.subplots(1,1,figsize=(16,4))
# ratings['rating'].plot(kind='hist')
plt.scatter(no_user_voted.index,no_user_voted,color='lightseagreen')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()

movies.head()

# Merge movies dataset with tags dataset
movies_with_tags = pd.merge(movies, tags, on='movieId', how='left')

# Merge the result with links dataset
movies_combined = pd.merge(movies_with_tags, links, on='movieId', how='left')

# Data Cleaning and Preprocessing
# Drop unnecessary columns
movies_combined.drop(['imdbId', 'tmdbId'], axis=1, inplace=True)

# Fill missing values with appropriate values or drop them
# For instance, for 'tag' column, we may fill missing values with an empty string
movies_combined['tag'].fillna('', inplace=True)

# You might need to handle missing values in other columns as well based on your analysis requirements

# Now, the dataset is preprocessed and ready for further analysis

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the final dataset
normalized_final_dataset = scaler.fit_transform(final_dataset)

# Convert the normalized dataset back to a DataFrame
normalized_final_df = pd.DataFrame(normalized_final_dataset, index=final_dataset.index, columns=final_dataset.columns)

# Display the normalized dataset
normalized_final_df.head()

# Fill missing values with zero
final_dataset_filled = final_dataset.fillna(0)

# Normalize the final dataset
normalized_final_dataset = scaler.fit_transform(final_dataset_filled)

# Convert the normalized dataset back to a DataFrame
normalized_final_df = pd.DataFrame(normalized_final_dataset, index=final_dataset_filled.index, columns=final_dataset_filled.columns)

# Display the normalized dataset
normalized_final_df.head()
