import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time
import textstat
from textblob import TextBlob
#Import statsmodel
import statsmodels.api as sm
#Import packages for data modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

class features:
        
    def calculate_sentiment(self,title):
        """
        Function to calculate the sentiment polarity of news title
        """
        return TextBlob(title).sentiment.polarity


    def get_readability_score(self,text):
        """
        Function to calculate the readability score for the text
        """
        return textstat.flesch_reading_ease(text)


    def get_baseline_features(self,news_df):
        """
        Function to get baseline model features
        """
        news_df['readability_score'] = news_df.text.apply(self.get_readability_score)
        news_df["word_count"] = news_df["text"].apply(lambda x: len(x.split()))
        news_df['sentiment_score'] = news_df.title.apply(self.calculate_sentiment)
        X = news_df[["readability_score","word_count","sentiment_score"]]
        y = news_df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
        return X_train, X_test, y_train, y_test


    def get_pipeline_features_labels(self,news_df):
        """
        Function to create pipeline features and labels
        """
        X = news_df[["text"]]
        y = news_df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
        return X_train, X_test, y_train, y_test
    