# Capstone Project (Sentiment Analysis) - NLP Applications

## Description

This project, named "Capstone Project (Sentiment Analysis) - NLP Applications," focuses on sentiment analysis using natural language processing (NLP) techniques.
The project utilises the spaCy library to implement a sentiment analysis model. 
It works with the Amazon product review dataset, consisting of over 34,000 consumer reviews, to analyse and classify sentiments as positive, negative, or neutral.
The importance of this project lies in its ability to automate sentiment analysis, providing insights into customer opinions and feedback.
[Link to sentiment_analysis.py](https://github.com/acp-dscs/finalCapstone/blob/main/sentiment_analysis.py)

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Credits](#credits)

## Installation

**To install and run this project locally, follow these steps:**

- Download the Amazon product review dataset from Kaggle.

- The Amazon Consumer Reviews data can be downloaded from Kaggle:
  ⁻	(Source): Datafiniti's Product Database
  ⁻	(https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products?resource=download)

- Save the dataset as a CSV file named amazon_product_reviews.csv.

- Install the required dependencies by running: pip install pandas spacy spacytextblob

## Usage

- Open the sentiment_analysis.py file in your preferred Python environment, the program was built using the IDE Visual Studio Code.

- Ensure the necessary Python packages are installed (pandas, spacy, spacytextblob).

- Set the index values in the code to identify the Consumer Review rows that you wish to compare.

- Run the script to perform sentiment analysis on the Amazon Product Reviews dataset.

- The script includes a function to preprocess text data, a function for sentiment analysis, and a test function on sample reviews.

- View the sentiment analysis outcomes, including sentiment labels, polarity, and similarity scores.

**Screenshots of the expected program output format.**


![Amazon Reviews Screenshot](https://github.com/acp-dscs/finalCapstone/raw/main/amazon_reviews.png)
![Amazon Reviews Screenshot](https://github.com/acp-dscs/finalCapstone/blob/main/amazon_reviews_output.png)


## Credits

This project was created by Anthony Pieri whilst undertaking the HyperionDev Data Science Bootcamp.
Anthony Pieri, produced and was responsible for the code, documentation, and testing.

