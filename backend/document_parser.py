# backend/document_parser.py
import fitz  # PyMuPDF
import docx
import os

def read_txt(file):
    return file.read().decode("utf-8")

def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_file(file, filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".txt":
        return read_txt(file)
    elif ext == ".pdf":
        return read_pdf(file)
    elif ext == ".docx":
        return read_docx(file)
    else:
        raise ValueError("Unsupported file format")

def chunk_text(text, chunk_size=300):
    paragraphs = text.split("\n")
    chunks, chunk = [], ""
    for para in paragraphs:
        if len(chunk) + len(para) < chunk_size:
            chunk += para + " "
        else:
            chunks.append(chunk.strip())
            chunk = para + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks
