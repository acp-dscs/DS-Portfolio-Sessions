"""
Task 21 Capstone Project 2 Pseudocode File
Task 21.1 Capstone Project - Natural Language Processing Applications file (sentiment_analysis.py)

This program performs sentiment analysis using the spaCy library.
By loading the en_core_web_sm spaCy model to enable natural language processing tasks.
Using the Amazon product review dataset (amazon_product_reviews.csv).
The Dataset contains a list of over 34,000 consumer reviews for Amazon products.
Necessary text cleaning and removal of stopwords has been undertaken for data analysis.
"""
# Initialise variables:
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
# Section 1: Implement sentiment analysis model using spaCy.
# Dataset will load 'amazon_product_reviews.csv' with 34,000 rows of data.
# For testing on smaller 5,000 row Dataset load 'amazon_product_reviews_5000.csv'. 
amazon_dataframe = pd.read_csv('amazon_product_reviews.csv', low_memory=False) 
nlp = spacy.load('en_core_web_sm') # Replace 'sm' with 'md' for more accurate results.
spacy_text_blob = SpacyTextBlob(nlp)
nlp.add_pipe('spacytextblob')
# Section 2: Function to preprocess text data from imported CSV file.
def preprocess_text(data):
    # Select the 'reviews.text' column and remove missing values.
    clean_data = data.dropna(subset=['reviews.text'])
    # Clean text by removing stopwords and other techniques.
    clean_data['reviews.text'] = clean_data['reviews.text'].apply(clean_text)
    return clean_data
# Text cleaning function 'clean_text'.
def clean_text(text):
    # Tokenize the text using spaCy.
    doc = nlp(text)
    # Remove stopwords and non-alphabetic words, using lower() and strip().
    clean_tokens = [token.text.lower().strip() for token in doc if not token.is_stop and token.is_alpha]
    # Join the tokens back into clean text output.
    clean_text = ' '.join(clean_tokens)
    return clean_text
# Section 2: Preprocess text data.
cleaned_data = preprocess_text(amazon_dataframe)
reviews_data = cleaned_data['reviews.text'] # Select the 'reviews.text' column from the cleaned data.
# Section 3: Function for sentiment analysis using spaCy and TextBlob.
def predict_sentiment(review):
    doc = nlp(review)  
    # Analyse sentiment using the polarity attribute '.blob' and the sentiment attribute '.sentiment'.
    polarity = doc._.blob.polarity
    sentiment = doc._.blob.sentiment
    # Classify sentiment based on polarity and sentiment.
    if polarity > 0:
        sentiment_label = "Positive"
    elif polarity < 0:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"  
    return {
        'polarity': polarity,
        'sentiment': sentiment,
        'sentiment_label': sentiment_label      
    }
# Section 4: Test model on sample product reviews.
def test_sentiment_analysis(reviews_data):
    # Choose two product reviews for similarity comparison
    my_review_of_choice1 = reviews_data['reviews.text'][20]
    my_review_of_choice2 = reviews_data['reviews.text'][70]
    # Get index numbers
    index_choice1 = reviews_data.index[reviews_data['reviews.text'] == my_review_of_choice1][0]
    index_choice2 = reviews_data.index[reviews_data['reviews.text'] == my_review_of_choice2][0]
    # Analyse and predict sentiment
    sentiment1 = predict_sentiment(my_review_of_choice1)
    sentiment2 = predict_sentiment(my_review_of_choice2)
    # Compare similarity between two reviews
    similarity_score = nlp(my_review_of_choice1).similarity(nlp(my_review_of_choice2))
    # Print sentiment and polarity outcomes of model.
    print("\nSentiment Analysis Outcomes:\n") # Index values for specific rows are +2 due to column title and starting from 0.
    # First review in question.
    print(f"First Test Under Review (Index Row {index_choice1+2}): {my_review_of_choice1}")
    print(f"\nReview of Index Row Number {index_choice1+2} in file 'amazon_product_reviews.csv' Sentiment: ")
    print(f"{sentiment1['sentiment_label']}")
    print(f"{sentiment1['sentiment']}")
    # Second review in question.
    print(f"\nSecond Test Under Review (Index Row {index_choice2+2}): {my_review_of_choice2}")
    print(f"\nReview of Index Row Number {index_choice2+2} in file 'amazon_product_reviews.csv' Sentiment: ")
    print(f"{sentiment2['sentiment_label']}")
    print(f"{sentiment2['sentiment']}")
    # Print similarity score of the two customer reviews in question.
    print(f"\nSimilarity score between the two reviews in question is: {similarity_score}\n")
# Test model on sample product reviews.
test_sentiment_analysis(cleaned_data)
# Program ends