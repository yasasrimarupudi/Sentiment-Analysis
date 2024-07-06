# Sentiment-Analysis
Overview
This project implements sentiment analysis using LSTM (Long Short-Term Memory) networks to classify the sentiment (positive, negative, neutral) of text data.

Dataset
The sentiment analysis model is trained on the [Name of Dataset] dataset, which consists of [brief description of dataset].
Dataset Source: [https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment]
LONGSHORT-TERMMEMORYNETWORKS: LSTM stands for Long Short Term 
Memory Networks. It is a variant of Recurrent Neural Networks. RNN's are usually used with 
sequential data such as text and audio. Usually, while computing an embedding matrix, the 
meaning of every word and its calculations (which are called hidden states) are stored. If the 
reference of a word, let’s say a word is used after 100 words in a text, then all these calculations 
RNNs cannot store in its memory. That’s why RNNs are not capable of learning these longterm dependencies
Preprocessing
Data cleaning: Removing noise, handling missing values.
Tokenization: Breaking text into tokens.
Lemmatization/Stemming: Normalizing tokens.
Vectorization: Converting text data into numerical vectors.
1) Supervised Learning:
In supervised learning, you train a model on a labeled dataset where each text sample is 
associated with its corresponding sentiment label (e.g., positive, negative, neutral). The 
model learns to recognize patterns in the text and their relation to sentiment labels. Common 
algorithms used for sentiment analysis in supervised learning include:
Support Vector Machines (SVM)
Naive Bayes Classifier
Logistic Regression
Neural Networks (e.g., LSTM, GRU)
2) Unsupervised Learning:
Unsupervised learning doesn't require labeled data for training. Instead, it aims to discover 
patterns and structures in the data without explicit guidance. In sentiment analysis, 
unsupervised techniques might involve clustering similar sentiment expressions or using 
lexicons to score sentiment words and aggregating them to determine the overall sentiment.
Summary:
In this machine learning project, we will build a binary text classifier that 
classifies the sentiment of the tweets into positive and negative. We will 
obtain more than 94% accuracy on validation. This interesting project helps 
businesses across the domains understand customers' sentiments/feelings 
towards their brands.
