import streamlit as st
import os
from backend.nlp.extract_topics import extract_topics
from backend.nlp.topic_modeling import enhance_sections_by_topic, extract_section_keywords
from backend.document_parser import parse_file, chunk_text
from backend.embedding import get_embeddings, embed_query
from backend.search import search
from backend.utils.highlight_query import highlight_query
from backend.rag.generate_answer_gemini import generate_answer_gemini


# Session states
if "selected_task" not in st.session_state:
    st.session_state["selected_task"] = None

st.set_page_config(page_title="Ask EDX++ Analyzer", layout="wide")
st.markdown("""
    <style>
    .task-icon-button {
        width: 50px;
        overflow: hidden;
        white-space: nowrap;
        transition: all 0.3s ease-in-out;
    }
    .task-icon-button:hover {
        width: 180px;
        background-color: #444;
    }
    .task-icon-label {
        display: inline-block;
        margin-left: 10px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Ask EDX++ – Smart PDF Analyzer")

# --- PDF Upload ---
st.subheader("📄 Upload Your PDF")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf", "txt", "docx"], accept_multiple_files=False)

if uploaded_file:
    raw_text = parse_file(uploaded_file, uploaded_file.name)
    st.session_state["doc_text"] = raw_text
    st.success("✅ File uploaded and text extracted!")
else:
    st.info("Please upload a file to begin.")

# --- Query Input ---
query = st.text_input("🔍 Ask a question about your document")

# --- Conditional NLP Task Buttons (visible only when no query) ---
if not query:
    st.markdown("---")
    st.subheader("🧠 Select an NLP Task")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📌 Topic Detection"):
            st.session_state["selected_task"] = "topic"

    with col2:
        if st.button("✂️ Summarization"):
            st.session_state["selected_task"] = "summary"

    with col3:
        if st.button("📚 Simplification"):
            st.session_state["selected_task"] = "simplify"

    col4, col5, col6 = st.columns(3)

    with col4:
        if st.button("🔑 Keyword Extraction"):
            st.session_state["selected_task"] = "keywords"

    with col5:
        if st.button("📖 Keyword Explanation"):
            st.session_state["selected_task"] = "wiki"

    with col6:
        if st.button("🧾 Bullet Summary"):
            st.session_state["selected_task"] = "bullets"

    if st.button("✨ Highlight Key Sentences"):
        st.session_state["selected_task"] = "highlight"

# --- NLP Sidebar Tasks (show only when query is present) ---
if query:
    with st.sidebar:
        st.markdown("### 🧠 NLP Tasks")
        if st.button("📌 Topic", key="topic_btn"):
            st.session_state["selected_task"] = "topic"
        if st.button("✂️ Summary", key="summary_btn"):
            st.session_state["selected_task"] = "summary"
        if st.button("📚 Simplify", key="simplify_btn"):
            st.session_state["selected_task"] = "simplify"
        if st.button("🔑 Keywords", key="keywords_btn"):
            st.session_state["selected_task"] = "keywords"
        if st.button("📖 Wiki", key="wiki_btn"):
            st.session_state["selected_task"] = "wiki"
        if st.button("🧾 Bullets", key="bullets_btn"):
            st.session_state["selected_task"] = "bullets"
        if st.button("✨ Highlight", key="highlight_btn"):
            st.session_state["selected_task"] = "highlight"

# --- Document Search + RAG ---
if uploaded_file and query:
    if "doc_text" in st.session_state:
        all_chunks = chunk_text(st.session_state["doc_text"])
        doc_embeddings = get_embeddings(all_chunks)
        query_embedding = embed_query(query)
        results = search(query_embedding, doc_embeddings, top_k=5)

        st.subheader("📌 Top Matching Document Chunks")
        for res in results:
            score = res['score']
            idx = res['corpus_id']
            chunk = all_chunks[idx]
            highlighted = highlight_query(chunk, query)

            st.markdown(f"**Score:** `{score:.2f}`")
            st.markdown(highlighted)

        st.subheader("💡 RAG-style Answer (Generated with Gemini)")
        top_chunks = [all_chunks[r['corpus_id']] for r in results]

        if st.button("🧠 Generate Answer with Gemini"):
            with st.spinner("🔍 Generating answer using Gemini..."):
                answer = generate_answer_gemini(query, top_chunks)
                st.success("✅ Answer generated!")
                st.markdown(answer)

# --- Conditional NLP Task Execution ---
if uploaded_file and st.session_state["selected_task"]:
    st.markdown("---")
    task = st.session_state["selected_task"]
    st.subheader(f"🔍 Performing Task: {task.title()}")

    if task == "topic":
            st.info("Running topic detection with BERTopic...")
            section_topics = extract_section_keywords(st.session_state["doc_text"])
            st.write("### 📌 Section-wise Topics")
            for topic, section in section_topics:
                st.markdown(f"#### 🧩 {topic}")
                st.markdown(section)

            section_html = "<h2>📌 Section-wise Topics</h2>"
            for topic, section in section_topics:
                section_html += f"<h3>{topic}</h3><p>{section}</p>"

            from backend.utils.pdf_utils import convert_html_to_pdf
            section_pdf = convert_html_to_pdf(section_html)

            st.download_button(
                label="📥 Download Topics PDF",
                data=section_pdf,
                file_name="section_topics.pdf",
                mime="application/pdf"
            )

            if st.button("✨ Enhance Sections"):
                with st.spinner("Combining and enhancing topic-based sections..."):
                    enhanced_sections = enhance_sections_by_topic(section_topics)
                    for topic, summary in enhanced_sections.items():
                        st.markdown(f"### 🧠 {topic}")
                        st.markdown(summary)

                # Generate HTML after spinner ends
                enhanced_html = "<h2>🧠 Enhanced Summaries</h2>"
                for topic, summary in enhanced_sections.items():
                    enhanced_html += f"<h3>{topic}</h3><p>{summary}</p>"

                enhanced_pdf = convert_html_to_pdf(enhanced_html)

                st.download_button(
                    label="📥 Download Enhanced PDF",
                    data=enhanced_pdf,
                    file_name="enhanced_sections.pdf",
                    mime="application/pdf"
                )




    elif task == "summary":
        st.info("Generating summary using Hugging Face BART pipeline...")
        length = st.radio("Select summary length:", ["Short", "Medium", "Long"], index=1, horizontal=True)

        from backend.nlp.summarizer import summarize_text

        summary = summarize_text(st.session_state["doc_text"], length)

        st.subheader("📝 Summary")
        st.markdown(summary)
        st.download_button("💾 Download Summary", summary, file_name="summary.txt")

    elif task == "simplify":
        st.info("Running simplification using transformer model...")
        from backend.nlp.simplify import DocumentSimplifier
        simplifier = DocumentSimplifier()

        with st.spinner("🔄 Simplifying the document..."):
            simplified_text = simplifier.simplify_text_from_string(st.session_state["doc_text"])

        st.subheader("📚 Simplified Document")
        st.markdown(simplified_text)

        st.download_button(
            label="💾 Download Simplified Text",
            data=simplified_text,
            file_name="simplified_document.txt",
            mime="text/plain"
        )

        similarity = simplifier.evaluate_similarity(st.session_state["doc_text"], simplified_text)
        readability = simplifier.readability_score(simplified_text)

        st.markdown(f"**🔎 Semantic Similarity with Original:** `{similarity:.2f}`")
        st.markdown(f"**📖 Flesch Reading Ease Score:** `{readability:.2f}`")


    elif task == "keywords":
        st.info("Extracting paragraphs and keywords...")

        from backend.nlp.keywords import extract_keywords
        from backend.utils.pdf_utils import generate_keyword_pdf
        from backend.utils.extract_paragraphs import extract_paragraphs_from_pdf

        if uploaded_file:
            with st.spinner("📄 Parsing document and identifying paragraphs..."):
                # In-memory file object → passed directly to PyMuPDF
                paragraphs = extract_paragraphs_from_pdf(uploaded_file)

            paragraphs_with_keywords = []

            st.subheader("🧾 Paragraphs and Their Keywords")
            for i, para in enumerate(paragraphs):
                if not para.strip() or len(para.split()) < 5:
                    continue  # Skip empty or too-short paragraphs

                keywords = extract_keywords(para)
                paragraphs_with_keywords.append((para, keywords))

                st.markdown(f"### 📄 Paragraph {i + 1}")
                st.markdown(para)
                st.markdown(f"🔑 **Keywords**: `{', '.join(keywords)}`")
                st.markdown("---")

            # PDF Output
            with st.spinner("📥 Generating customized PDF..."):
                from tempfile import NamedTemporaryFile
                temp_pdf = NamedTemporaryFile(delete=False, suffix=".pdf")
                pdf_path = generate_keyword_pdf(paragraphs_with_keywords, output_path=temp_pdf.name)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Download Keywords Annotated PDF",
                    data=f,
                    file_name="keywords_paragraphs.pdf",
                    mime="application/pdf"
                )


    elif task == "wiki":
        st.info("Looking up keyword explanations...")
        st.write("[TODO] Wikipedia definitions here.")

    elif task == "bullets":
        st.info("Generating bullet summary...")
        st.write("[TODO] Bullet points here.")

    elif task == "highlight":
        st.info("Highlighting key sentences...")
        st.write("[TODO] Highlighted sentences here.")
