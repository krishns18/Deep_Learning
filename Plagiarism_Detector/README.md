# Plagiarism Detector Deployment

This repository contains code and associated files for deploying a plagiarism detector using AWS SageMaker, Scikit-learn
and PyTorch

## Project Overview

In this project,I built a plagiarism detector that examines a text file and performs binary classification; labeling 
that file as either *plagiarized* or *not*, depending on how similar that text file is to a provided source text. 
Detecting plagiarism is an active area of research; the task is non-trivial and the differences between paraphrased 
answers and original work are often not so obvious.

This project is broken down into three main notebooks:

**Notebook 1: Data Exploration**
* Load in the corpus of plagiarism text data.
* Explore the existing data features and the data distribution.

**Notebook 2: Feature Engineering**

* Clean and pre-process the text data.
* Defined features for comparing the similarity of an answer text and a source text, and extract similarity features.
* Selected "good" features, by analyzing the correlations between different features.
* Created train/test `.csv` files that hold the relevant features and class labels for train/test data points.

**Notebook 3: Train and Deploy Model in SageMaker**

* Upload train/test feature data to S3.
* Define a binary classification model and a training script.
* Trained the model and deployed it using SageMaker.
* Evaluated my deployed classifier.


**Model Training**

Following are the models built to achieve 100% test accuracy results:

1. Multinomial Naive-Bayes
2. SVC
3. Fully Connected Feed Forward Models

| Model | Test Accuracy |
| --- | --- |
| Multinomial Naive-Bayes | 60% |
| SVC | 96% |
| Fully Connected Feed Forward Model (1 hidden layer 20 nodes, dropout-0.3, adam optimizer, learning rate-0.001) | 60% |
| Fully Connected  Feed Forward Model (2 hidden layer 20 nodes, dropout-0.3, adam optimizer, learning rate-0.001) | 60% |
| Fully Connected  Feed Forward Model (1 hidden layer 20 nodes, dropout-0.2, adam optimizer, learning rate-0.01) | 100% |
