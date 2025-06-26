import streamlit as st
from backend.document_parser import parse_file, chunk_text
from backend.embedding import get_embeddings, embed_query
from backend.search import search
from backend.utils.highlight_query import highlight_query
from backend.rag.generate_answer_gemini import generate_answer_gemini

st.set_page_config(page_title="Ask EDX Clone", layout="wide")
st.title("📄 Ask EDX Clone – Document Search + RAG")

uploaded_files = st.file_uploader("Upload document(s)", type=["txt", "pdf", "docx"], accept_multiple_files=True)
query = st.text_input("🔍 Ask a question about your documents")

if uploaded_files and query:
    all_chunks = []
    file_refs = []

    with st.spinner("🔄 Processing files..."):
        for file in uploaded_files:
            text = parse_file(file, file.name)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            file_refs.extend([file.name] * len(chunks))

        doc_embeddings = get_embeddings(all_chunks)
        query_embedding = embed_query(query)
        results = search(query_embedding, doc_embeddings, top_k=5)

    st.subheader("📌 Top Matching Document Chunks")
    for res in results:
        score = res['score']
        idx = res['corpus_id']
        chunk = all_chunks[idx]
        highlighted = highlight_query(chunk, query)

        st.markdown(f"**Source:** `{file_refs[idx]}` — **Score:** `{score:.2f}`")
        st.markdown(highlighted)

    st.subheader("💡 RAG-style Answer (Generated with Gemini)")
    top_chunks = [all_chunks[r['corpus_id']] for r in results]

    if st.button("🧠 Generate Answer with Gemini"):
        with st.spinner("🔍 Generating answer using Gemini..."):
            answer = generate_answer_gemini(query, top_chunks)
            st.success("✅ Answer generated!")
            st.markdown(answer)
