import re

def highlight_query(text, query):
    for word in query.strip().split():
        text = re.sub(f"(?i)\\b({re.escape(word)})\\b", r"**\1**", text)
    return text
