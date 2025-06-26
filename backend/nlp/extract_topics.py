from bertopic import BERTopic
from backend.nlp.text_cleaner import text_cleaner


def extract_topics(text: str):
    # Split long text into meaningful paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    
    # Clean each paragraph
    cleaned_paragraphs = [text_cleaner(p) for p in paragraphs if len(p.split()) > 5]

    # Fit BERTopic
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(cleaned_paragraphs)

    # Generate frequency table and visual chart
    freq_df = topic_model.get_topic_info()
    visual_html = topic_model.visualize_barchart(top_n_topics=10).to_html()

    # Return for further use
    return freq_df[["Topic", "Name", "Count"]], visual_html, topics, cleaned_paragraphs
