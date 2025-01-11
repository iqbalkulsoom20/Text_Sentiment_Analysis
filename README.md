# Text_Sentiment_Analysis
This project focuses on sentiment analysis of movie reviews using the IMDb dataset. The dataset consists of 50,000 movie reviews labeled as positive or negative. The main goal of this project is to develop models that can accurately classify the sentiment of movie reviews.
## Dataset

| Label | Number of Samples|
| :-----: | :---: | 
| Positive | 25000| 
| Negative | 25000| 

Dataset link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data?select=IMDB+Dataset.csv
## Description
The project involves the following steps:

## Text Preprocessing:
Prepare text data by tokenizing, removing stopwords, punctuation, and applying lemmatization to normalize words into their root forms.

## Feature Engineering:
Convert processed text into numerical format using TF-IDF, which assigns importance to words based on their frequency in reviews and the overall dataset.

## Model Training:
Train a Logistic Regression model using labeled data to predict sentiments (positive or negative) for new text inputs.

## Model Evaluation:
Assess model performance using metrics like accuracy, precision, recall, F1-score, and visualize results with a confusion matrix.
## Requirements

To run this project, the following dependencies are required:
- joblib==1.4.2
- nltk==3.9.1
- numpy==2.2.1
- pandas==2.2.3
- scikit-learn==1.6.0
