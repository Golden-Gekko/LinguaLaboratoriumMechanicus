from lxml import etree


def extract_text_from_fb2(file_path):
    parser = etree.XMLParser(recover=True, encoding='utf-8')

    with open(file_path, 'rb') as f:
        content_bytes = f.read()
    # Определение кодировки - utf-8 или cp1251. Без этого падала при cp1251
    first_line = content_bytes.split(b'\n', 1)[0].decode('latin-1', errors='ignore')
    if 'windows-1251' in first_line.lower() or 'cp1251' in first_line.lower():
        content = content_bytes.decode('cp1251')
    else:
        try:
            content = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            content = content_bytes.decode('cp1251')

    tree = etree.fromstring(content.encode(), parser)

    text = ' '.join(
        tree.xpath(
            '//fb2:p//text()',
            namespaces={'fb2': 'http://www.gribuser.ru/xml/fictionbook/2.0'}
        )
    ).strip()

    return text if text else None
