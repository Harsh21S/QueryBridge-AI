from xhtml2pdf import pisa
from fpdf import FPDF
import io

def convert_html_to_pdf(html_content: str) -> bytes:
    pdf_bytes_io = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html_content), dest=pdf_bytes_io)
    return pdf_bytes_io.getvalue()


def generate_keyword_pdf(paragraphs_with_keywords, output_path="output.pdf"):
    from fpdf import FPDF
    import unicodedata

    def clean_text(text):
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for i, (para, keywords) in enumerate(paragraphs_with_keywords):
        pdf.multi_cell(0, 10, f"Paragraph {i + 1}:\n{clean_text(para)}\n")
        pdf.set_font("Arial", style="B", size=11)
        pdf.multi_cell(0, 10, f"Keywords: {clean_text(', '.join(keywords))}\n\n")
        pdf.set_font("Arial", style="", size=12)

    pdf.output(output_path)
    return output_path