import re
from nltk.corpus import stopwords
import nltk
import streamlit as st

@st.cache_resource
def get_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    return set(stopwords.words("english"))

stop_words = get_stopwords()
custom_stopwords = stop_words.union({"bookmyshow", "username", "email", "event", "account"})

def text_cleaner(text):
    text = re.sub(r"http\S+", "", text)                # remove URLs
    text = re.sub(r"\S+@\S+", "", text)                # remove emails
    text = re.sub(r"[^a-zA-Z\s]", "", text)            # remove numbers/punct
    text = text.lower()
    text = " ".join(word for word in text.split() if word not in custom_stopwords)
    return text
