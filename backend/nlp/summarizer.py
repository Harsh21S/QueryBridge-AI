from transformers import pipeline, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import streamlit as st
import os

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

@st.cache_resource
def ensure_nltk():
    nltk_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(os.path.join(nltk_path, "tokenizers", "punkt")):
        nltk.download("punkt", download_dir=nltk_path)
    nltk.data.path.append(nltk_path)
    return True

ensure_nltk()

summarizer = load_summarizer()
tokenizer = load_tokenizer()

def split_into_chunks(text, max_tokens=1024):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_length = 0

    for sent in sentences:
        sent_len = len(tokenizer.encode(sent, add_special_tokens=False))
        if current_length + sent_len <= max_tokens:
            current_chunk.append(sent)
            current_length += sent_len
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_length = sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"[INFO] Split into {len(chunks)} chunks.")
    return chunks

def summarize_text(text, length="Medium"):
    target_words = {"Short": 80, "Medium": 180, "Long": 350}[length]
    chunks = split_into_chunks(text)
    
    if len(chunks) == 1:
        # Directly summarize single chunk
        print("[INFO] Only one chunk – summarizing directly.")
        result = summarizer(
            chunks[0],
            max_length=int(target_words * 1.5),
            min_length=int(target_words * 0.6),
            do_sample=False
        )
        return result[0]["summary_text"]

    # Step 1: Summarize each chunk individually
    intermediate = []
    for i, chunk in enumerate(chunks):
        print(f"[INFO] Summarizing chunk {i+1}/{len(chunks)}...")
        result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        intermediate.append(result[0]["summary_text"])

    merged_summary = " ".join(intermediate)

    # Step 2: Final summarization to target length
    print(f"[INFO] Final summarization from {len(merged_summary.split())} words...")
    max_tokens = int(target_words * 1.5)

    final_result = summarizer(
        merged_summary, 
        max_length=max_tokens, 
        min_length=max(30, int(target_words * 0.6)), 
        do_sample=False
    )

    return final_result[0]["summary_text"]


