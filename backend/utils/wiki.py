import re
import string
import yake
import wikipedia
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st

@st.cache_resource
def ensure_nltk_data():
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    return True

ensure_nltk_data()



def extract_unique_keywords(paragraphs_with_keywords):
    """Extract and clean unique keywords from paragraph-keyword pairs."""
    all_keywords = set()
    stop_words = set(stopwords.words("english"))

    for _, keywords in paragraphs_with_keywords:
        for word in keywords:
            cleaned = word.lower().strip(string.punctuation)
            if (cleaned and cleaned not in stop_words and len(cleaned) > 2):
                all_keywords.add(cleaned.title())  # Title-case for Wikipedia

    return sorted(all_keywords)


def fetch_wikipedia_summary(keyword):
    """Get a short summary of a keyword from Wikipedia, with fallback for disambiguation."""
    try:
        return wikipedia.summary(keyword, sentences=2)
    except wikipedia.DisambiguationError as e:
        for option in e.options:
            try:
                return wikipedia.summary(option, sentences=2)
            except Exception:
                continue
        return f"[Disambiguation] {keyword} is ambiguous. Top suggestions: {', '.join(e.options[:3])}"
    except wikipedia.PageError:
        return f"[Not Found] No information found for '{keyword}'."
    except Exception as e:
        return f"[Error] {str(e)}"


def lookup_keywords_explanation(paragraphs_with_keywords):
    """
    Main function: Extract unique keywords, search on Wikipedia,
    return a dictionary mapping keyword -> explanation.
    """
    unique_keywords = extract_unique_keywords(paragraphs_with_keywords)
    keyword_info = {}

    for keyword in unique_keywords:
        keyword_info[keyword] = fetch_wikipedia_summary(keyword)

    return keyword_info
