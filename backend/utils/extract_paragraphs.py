import fitz  # PyMuPDF

def extract_paragraphs_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_lines = []

    for page in doc:
        lines = page.get_text("text").split("\n")
        all_lines.extend(lines + ["<PAGEBREAK>"])  # to track logical breaks

    paragraphs = []
    current_para = []

    for line in all_lines:
        stripped = line.strip()
        if stripped == "" or stripped == "<PAGEBREAK>":
            if current_para and current_para[-1] != "":
                current_para.append("")  # track blank line
        else:
            if current_para and current_para[-1] == "":
                # if a blank line just ended, start new paragraph
                paragraphs.append(" ".join(current_para[:-1]))
                current_para = [stripped]
            else:
                current_para.append(stripped)

    if current_para:
        paragraphs.append(" ".join(current_para))

    return paragraphs
