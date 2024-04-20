Movie Recommendation System
Welcome to the Movie Recommendation System project! This system provides personalized movie recommendations using collaborative filtering, content-based filtering, and traditional machine learning models.

Overview
The Movie Recommendation System is built to help users discover movies they might enjoy based on their preferences and similarities between movies. It employs various recommendation techniques, including collaborative filtering, content-based filtering, and traditional machine learning models trained on movie metadata.

Features
Multiple Recommendation Techniques: The system utilizes collaborative filtering, content-based filtering, and traditional machine learning models for recommendation.
Model Evaluation: Performance metrics such as RMSE and MAE are used to evaluate and compare the recommendation models.
Interactive Interface: The system is built using Streamlit, offering an interactive and intuitive user experience.
Installation
To run the Movie Recommendation System locally, follow these steps:

Clone the repository:
bash
Copy code
git clone https://github.com/your-username/movie-recommendation-system.git
Navigate to the project directory:
bash
Copy code
cd movie-recommendation-system
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:
bash
Copy code
streamlit run app.py
Access the app via the provided URL in your browser.
Usage
Select the recommendation type: Collaborative Filtering, Content-Based Filtering, or Traditional ML Models.
Provide the required inputs (e.g., user ID, movie ID).
Click the button to get recommendations.
Data
The MovieLens dataset is used for this project, containing movie ratings and metadata. The dataset files (movies.csv, ratings.csv) are included in the repository.

Traditional ML Models
In addition to collaborative filtering and content-based filtering, the system incorporates traditional machine learning models trained on movie metadata. These models analyze features such as movie genres and titles to provide recommendations.

Model Comparison
Performance of all recommendation techniques is evaluated using metrics such as RMSE and MAE. The results are compared to identify the most effective approach for generating movie recommendations.

Contributing
Contributions to the Movie Recommendation System project are welcome! If you'd like to contribute, please follow the steps outlined in the CONTRIBUTING.md file.

License
This project is licensed under the MIT License. See the LICENSE file for details.