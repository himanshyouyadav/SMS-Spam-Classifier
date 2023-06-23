import streamlit as st
import pickle
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

import string 
from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    alpha_numeric = []
    for t in text:
        if t.isalnum(): #Removing Special Character
            if t not in stopwords.words('english') and t not in string.punctuation: #StopWords
                alpha_numeric.append(ps.stem(t)) #Stemming
    
    return " ".join(alpha_numeric)

input_sms = st.text_input("Enter the message")

if st.button("Predict"):

    #1. Preprocess
    transformed_sms = transform_text(input_sms)

    #2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    #3. Predict 
    result = model.predict(vector_input)[0]

    #4. Display

    if result==1:
        st.header("Spam")
    else:
        st.header("Not a Spam")