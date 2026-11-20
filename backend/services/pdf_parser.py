from services.timing import timed_stage
from io import StringIO

import pymupdf


@timed_stage("PDF text extraction with PyMuPDF")
def extract_text_from_pdf(pdf_path: str, request_id: str = "unknown"):

    document = pymupdf.open(pdf_path)
    text_buffer = StringIO()
    try:
        for page in document:
            text_buffer.write(page.get_text())
        return text_buffer.getvalue()
    finally:
        document.close()
