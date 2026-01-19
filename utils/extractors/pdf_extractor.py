import fitz


def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise RuntimeError(f'Failed to open PDF: {e}')

    text_parts = []
    for _, page in enumerate(doc):
        try:
            text = page.get_text().strip()
            if text:
                text_parts.append(text)
        except Exception:
            continue

    doc.close()

    full_text = ' '.join(text_parts).strip()
    return full_text if full_text else None
