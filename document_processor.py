import io
import PyPDF2
import docx

def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file_bytes):
    return file_bytes.decode('utf-8', errors='ignore')

def process_document(filename, file_bytes):
    ext = filename.split('.')[-1].lower()
    if ext == 'pdf':
        text = extract_text_from_pdf(file_bytes)
    elif ext == 'docx':
        text = extract_text_from_docx(file_bytes)
    elif ext == 'txt':
        text = extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_length:
            break
        start += (chunk_size - overlap)
    return chunks
