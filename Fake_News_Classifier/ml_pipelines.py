import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time
import textstat
from textblob import TextBlob
import statsmodels.api as sm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Input, Dropout
from keras import Sequential
from preprocess import Preprocesser

class pipelines:
    
    def create_svc_pipeline(self):
        """
        Function to create SVC classifier pipeline
        """
        pipeline = Pipeline([
                        ('vec', CountVectorizer(analyzer='word',       
                             min_df=10,                      # minimum reqd occurences of a word 
                             stop_words='english',           # remove stop words
                             lowercase=True,                 # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}')),  
                        ('tfidf', TfidfTransformer()),
                        ('classifier', LinearSVC())
                    ])
        return pipeline



    def create_nvb_pipeline(self):
        """
        Function to create Multinomial Naive-Bayes classifier Pipeline
        """
        pipeline = Pipeline([
        ('vec', CountVectorizer(analyzer='word',       
                             min_df=10,                      # minimum reqd occurences of a word 
                             stop_words='english',           # remove stop words
                             lowercase=True,                 # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}')),  
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB()),  
        ])
        return pipeline