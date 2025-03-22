# Sentiment-Based Product Recommendation System

## Project Overview
Ebuss is an emerging e-commerce company competing with established players like Amazon and Flipkart. To enhance user experience and drive sales, Ebuss aims to build a sentiment-based product recommendation system. This system utilizes user reviews and ratings to recommend products more effectively by leveraging machine learning models for sentiment analysis and recommendation algorithms.

## Project Goals
1. **Sentiment Analysis:** Analyze product reviews after text preprocessing and build a machine learning model to determine user sentiments for multiple products.
2. **Recommendation System:** Implement and evaluate user-based and item-based recommendation models.
3. **Sentiment-Enhanced Recommendations:** Improve recommendations by filtering the top five products based on sentiment analysis.
4. **Deployment:** Develop and deploy an end-to-end web application using Flask and make it publicly accessible via Heroku.

## Key Components
### 1. Data Sourcing and Sentiment Analysis
- **Exploratory Data Analysis (EDA):** Understand data distributions, trends, and insights.
- **Data Cleaning:** Handle missing values, duplicates, and inconsistent data.
- **Text Preprocessing:** Tokenization, stopword removal, lemmatization, etc.
- **Feature Extraction:** Utilize Bag-of-Words (BoW), TF-IDF, or word embeddings.
- **Model Training:** Train at least three of the following models and select the best one:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Naive Bayes
- **Class Imbalance & Hyperparameter Tuning:** If needed, apply techniques to balance the dataset and optimize models.

### 2. Building the Recommendation System
- **User-Based Collaborative Filtering:** Recommend products based on user similarities.
- **Item-Based Collaborative Filtering:** Recommend products based on item similarities.
- **Model Evaluation:** Compare the two recommendation systems and select the most effective one.

### 3. Enhancing Recommendations with Sentiment Analysis
- Apply the trained sentiment analysis model to refine recommendations.
- Recommend 20 products using the collaborative filtering model and filter the top 5 based on sentiment analysis.

### 4. Web Application & Deployment
- **User Interface Features:**
  - Input an existing username.
  - Submit the username via a button "Get Recommendations".
  - Display five recommended products based on sentiment analysis.
- **Deployment:** Deploy the application using Flask and make it publicly accessible via Heroku.

## Launching the app
Access the web app at (https://ebuss-product-reco-62ec2e58ca0f.herokuapp.com/)

## Deployment on Heroku
The app is deployed at below link on Heroku
https://ebuss-product-reco-62ec2e58ca0f.herokuapp.com/   ```

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Flask, XGBoost, Heroku
- **Machine Learning Models:** Logistic Regression, Random Forest, XGBoost
- **Deployment Platform:** Heroku

## Contributors
- Abhishek Mulay

## License
