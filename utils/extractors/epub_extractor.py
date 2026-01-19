import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


def extract_text_from_epub(file_path):
    try:
        book = epub.read_epub(file_path, {'ignore_ncx': True})
    except Exception as e:
        raise RuntimeError(f'Failed to read EPUB: {e}')

    text_parts = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            # Удаляем скрипты и стили
            for script in soup(['script', 'style']):
                script.decompose()
            text = soup.get_text(separator=' ', strip=True)
            if text:
                text_parts.append(text)

    full_text = ' '.join(text_parts).strip()
    return full_text if full_text else None
