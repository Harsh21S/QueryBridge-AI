import re
import nltk
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import textstat

nltk.download('punkt')


class DocumentSimplifier:
    def __init__(self, model_name="csebuetnlp/mT5_multilingual_XLSum"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            "summarization", 
            model=model_name, 
            tokenizer=model_name, 
            device=self.device
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"https?://\S+", "", text)
        return text.strip()

    def simplify_text_from_string(self, raw_text):
        cleaned = self.clean_text(raw_text)

        try:
            print("[INFO] Running simplification on entire text...")
            result = self.pipeline(
                cleaned,
                max_length=150,
                min_length=40,
                do_sample=False
            )
            simplified = result[0]["summary_text"].strip()
            return simplified
        except Exception as e:
            print(f"[ERROR] Simplification failed: {e}")
            return raw_text  # fallback

    def evaluate_similarity(self, original, simplified):
        emb1 = self.embedding_model.encode(original, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(simplified, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2)[0][0])

    def readability_score(self, text):
        return textstat.flesch_reading_ease(text)
