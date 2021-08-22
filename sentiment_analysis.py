# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import nltk
import seaborn as sns
import re

# make a title
st.title("Sentiment Analysis of COVID-19 in Indonesia in Web Application")
st.write("### This project is developed with **Python** + **Streamlit** by using COVID-19 Tweet dataset from **Kaggle**")
st.write("### Link Dataset : [Click Here!](https://www.kaggle.com/dionisiusdh/covid19-indonesian-twitter-sentiment)")

# import data
df = pd.read_csv('TRANSLATED-covid-sentiment.csv')

st.header("COVID-19 DataFrame")
st.write(df)

# import clean data
st.header("COVID-19 DataFrame + Cleaning")
st.write("### You can check the process at : [Click Here!](https://colab.research.google.com/drive/1TxBVVkz7pYMSEc1EyyGh_F6d_oqw99wb?usp=sharing)")
df_clean = pd.read_csv('df_clean.csv')
st.write(df_clean)

# frequency plot
st.header("Frequency Plot")
st.write("### Nb : Make sure to see it in **Dark Mode**")
from PIL import Image
img = Image.open('freqplot.png')
st.image(img)

# create a BoW model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=5)
X = cv.fit_transform(df_clean['tweet_clean']).toarray()
y = df_clean.iloc[:, -1].values

# # splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# create a naive bayes model
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X, y)
y_pred = classifier.predict(X_test)
#
# # #show metrics evaluation
# st.header("Metrics Evaluation")
# from sklearn.metrics import accuracy_score
# acc = round(accuracy_score(y_test, y_pred),2)
# st.write(str(round(acc*100,2)) + ' %')

# prediction

st.title("Sentiment Prediction")

words = st.text_input("Masukkan kalimat yang akan dianalisis sentimennya")

if words:
    if len(words.split()) < 5:
        st.write("Error! Silakan masukkan kalimat dengan jumlah kata lebih dari 5")
    else:
        lst = []
        lst.append(words)
        dt = pd.DataFrame()
        dt['text'] = lst
        cv_pred = TfidfVectorizer(max_features=5)
        word_pred = cv_pred.fit_transform(dt['text']).toarray()
        res = classifier.predict(word_pred)
        lst_res = []
        lst_res.append(res)
        dt['label'] = lst_res
        st.write(dt)





