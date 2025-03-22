#load required data
import warnings
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the user-based collaborative filtering model (Top 20 products for a user)
with open("user_final_rating.pkl", "rb") as f:
    user_final_rating = pickle.load(f)

user_final_rating_df = pd.DataFrame(user_final_rating)

# Load the sentiment analysis model
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Load reviews dataset
with open("processed_reviews.pkl", "rb") as f:
    df_reviews = pickle.load(f)

# Load the vectorizers
with open("word_vectorizer.pkl", "rb") as f:
    word_vectorizer = pickle.load(f)

with open("char_vectorizer.pkl", "rb") as f:
    char_vectorizer = pickle.load(f)


def check_user_exists(username):
    """Check if a given username exists in the dataset."""
    return username in df_reviews["reviews_username"].unique()

# Function to get top 20 recommended products for a user
def get_top_20_products(username):
    return user_final_rating_df.loc[username].sort_values(ascending=False)[0:20].index.tolist()

def get_top_5_products(username):
    top_20_products = get_top_20_products(username)
    # Filter reviews for the top 20 recommended products
    filtered_reviews = df_reviews[df_reviews["name"].isin(top_20_products)]


    # Extract the text reviews for sentiment prediction
    X_reviews = filtered_reviews['reviews_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))


    # Transform the reviews into feature vectors
    word_features = word_vectorizer.transform(X_reviews)
    char_features = char_vectorizer.transform(X_reviews)

    # Combine word and char features
    test_features = hstack([char_features, word_features])

    # Predict sentiment
    filtered_reviews["predicted_sentiment"] = xgb_model.predict(test_features)

    # Keep only products with **positive sentiment** (if applicable)
    final_filtered_reviews = filtered_reviews[filtered_reviews["predicted_sentiment"] == 1]

    # Select the **top 5 products** considering ratings and sentiment
    final_recommendations = (
        final_filtered_reviews.groupby("name")
        .agg(avg_rating=("reviews_rating", "mean"), sentiment_count=("predicted_sentiment", "count"))
        .sort_values(["sentiment_count", "avg_rating"], ascending=[False, False])
        .head(5)
    )

    top_5_products = final_recommendations.index.tolist()

    return top_5_products