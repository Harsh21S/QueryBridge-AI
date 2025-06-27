import re
from keybert import KeyBERT
from collections import defaultdict
from backend.rag.generate_answer_gemini import generate_answer_gemini
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
import numpy as np
import streamlit as st

# Text cleaning function
@st.cache_resource
def get_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    return set(stopwords.words("english"))

stop_words = get_stopwords()
custom_stopwords = stop_words.union({"bookmyshow", "username", "email", "event", "account"})
model = SentenceTransformer('all-MiniLM-L6-v2')



def text_cleaner(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return " ".join(word for word in text.split() if word not in custom_stopwords)

def extract_section_keywords(text):
    sections = re.split(r'\n\s*\n+', text)
    kw_model = KeyBERT()
    results = []

    for sec in sections:
        sec = sec.strip()
        if len(sec) < 50:
            continue
        keywords = kw_model.extract_keywords(sec, stop_words='english', top_n=3)
        topic = ", ".join([kw[0] for kw in keywords])
        results.append((topic.title(), sec))

    return results


def enhance_sections_by_topic(sectioned, similarity_threshold=0.7):
    # Step 1: Collect all unique topic labels and embeddings
    topic_to_paragraphs = defaultdict(list)
    for topic, content in sectioned:
        topic_to_paragraphs[topic].append(content)
    
    topics = list(topic_to_paragraphs.keys())
    embeddings = model.encode(topics, convert_to_tensor=True)

    # Step 2: Group similar topics
    merged_groups = []
    used = set()

    for i, topic_i in enumerate(topics):
        if i in used:
            continue
        group = [topic_i]
        for j in range(i + 1, len(topics)):
            if j in used:
                continue
            sim = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            if sim >= similarity_threshold:
                group.append(topics[j])
                used.add(j)
        used.add(i)
        merged_groups.append(group)

    # Step 3: Combine content and generate summaries
    enhanced = {}
    for group in merged_groups:
        merged_topic = ", ".join(group)
        combined_text = "\n".join(
            paragraph for topic in group for paragraph in topic_to_paragraphs[topic]
        )

        prompt = f"Rewrite and enhance the following text into a coherent summary for the topic(s): '{merged_topic}':\n\n{combined_text}"
        try:
            summary = generate_answer_gemini(prompt, [combined_text])
        except Exception as e:
            summary = f"❌ Error generating summary: {e}"
        
        enhanced[merged_topic] = summary

    return enhanced

# def extract_topics(text: str):
#     # Split long text into meaningful paragraphs
#     paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    
#     # Clean each paragraph
#     cleaned_paragraphs = [text_cleaner(p) for p in paragraphs if len(p.split()) > 5]

#     # Fit BERTopic
#     topic_model = BERTopic()
#     topics, _ = topic_model.fit_transform(cleaned_paragraphs)

#     # Generate frequency table and visual chart
#     freq_df = topic_model.get_topic_info()
#     visual_html = topic_model.visualize_barchart(top_n_topics=10).to_html()

#     # Return for further use
#     return freq_df[["Topic", "Name", "Count"]], visual_html, topics, cleaned_paragraphs