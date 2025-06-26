import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
custom_stopwords = stop_words.union({"bookmyshow", "username", "email", "event", "account"})

def text_cleaner(text):
    text = re.sub(r"http\S+", "", text)                # remove URLs
    text = re.sub(r"\S+@\S+", "", text)                # remove emails
    text = re.sub(r"[^a-zA-Z\s]", "", text)            # remove numbers/punct
    text = text.lower()
    text = " ".join(word for word in text.split() if word not in custom_stopwords)
    return text
