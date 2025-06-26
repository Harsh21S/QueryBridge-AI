from backend.rag.generate_answer_gemini import generate_answer_gemini  # or use any summarizer

def enhance_sections(paragraphs, topics):
    from collections import defaultdict

    # Group paragraphs by topic
    topic_groups = defaultdict(list)
    for para, topic in zip(paragraphs, topics):
        if topic != -1:  # skip outliers
            topic_groups[topic].append(para)

    enhanced_sections = {}
    for topic, chunks in topic_groups.items():
        combined_text = " ".join(chunks)
        prompt = f"Rewrite and enhance the following text into a coherent summary for topic {topic}:\n\n{combined_text}"
        try:
            summary = generate_answer_gemini(prompt, [combined_text])
        except Exception as e:
            summary = f"❌ Error generating summary: {e}"
        enhanced_sections[topic] = summary

    return enhanced_sections
