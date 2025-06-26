import fitz  # PyMuPDF

def extract_paragraphs_from_pdf(file_obj):
    file_obj.seek(0)  # 🔁 Reset stream to beginning
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    all_lines = []

    for page in doc:
        lines = page.get_text("text").split("\n")
        all_lines.extend(lines + ["<PAGEBREAK>"])

    paragraphs = []
    current_para = []

    for line in all_lines:
        stripped = line.strip()
        if stripped == "" or stripped == "<PAGEBREAK>":
            if current_para and current_para[-1] != "":
                current_para.append("")
        else:
            if current_para and current_para[-1] == "":
                paragraphs.append(" ".join(current_para[:-1]))
                current_para = [stripped]
            else:
                current_para.append(stripped)

    if current_para:
        paragraphs.append(" ".join(current_para))

    return paragraphs
