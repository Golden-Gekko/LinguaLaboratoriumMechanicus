import argparse
from collections import Counter
from datetime import datetime
import json
import os
import sys
from tqdm import tqdm

from extractors import (
    extract_text_from_epub, extract_text_from_fb2, extract_text_from_pdf)

EXTENSION_TO_EXTRACTOR = {}
EXTENSION_TO_EXTRACTOR['.fb2'] = extract_text_from_fb2
EXTENSION_TO_EXTRACTOR['.epub'] = extract_text_from_epub
EXTENSION_TO_EXTRACTOR['.pdf'] = extract_text_from_pdf
LOG_DIR = 'logs'


def process_book_files(input_dir, output_dir=None, allowed_ext=None):
    error_logs = []
    counter = Counter()
    ext_counter = Counter()

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'JSON')
    os.makedirs(output_dir, exist_ok=True)

    if allowed_ext is None:
        allowed_extensions = set(EXTENSION_TO_EXTRACTOR.keys())
    else:
        allowed_extensions = []
        for ext in allowed_ext:
            ext = ext.lower()
            if not ext.startswith('.'):
                ext = '.' + ext
            if EXTENSION_TO_EXTRACTOR.get(ext, None):
                allowed_extensions.append(ext)
            else:
                print(
                    f'Warning: неподдерживаемое расширение: "{ext}"'
                )
        allowed_extensions = set(allowed_extensions)

    files_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            _, ext = os.path.splitext(file.lower())
            if ext in allowed_extensions:
                files_list.append((root, file))

    pbar = tqdm(total=len(files_list), desc='Обработка')

    for root, file in files_list:
        file_path = os.path.join(root, file)
        _, ext = os.path.splitext(file.lower())
        extractor = EXTENSION_TO_EXTRACTOR.get(ext)

        try:
            text = extractor(file_path)
            if text is None:
                error_logs.append(f'Warning: Не найден текст в "{file_path}"')
                counter['warnings'] += 1
                pbar.update(1)
                continue

            output_file_path = os.path.join(
                output_dir,
                f'{os.path.splitext(file)[0]}.json'
            )

            data = {
                'parent_folder': os.path.basename(root),
                'file_name': file,
                'text': text
            }

            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            counter['processed'] += 1
            ext_counter[ext] += 1

        except Exception as e:
            counter['errors'] += 1
            error_logs.append(f'Ошибка при обработке "{file_path}": {e}')

        pbar.update(1)

    pbar.close()

    if error_logs:
        os.makedirs(LOG_DIR, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = os.path.join(LOG_DIR, f'log_{timestamp}.txt')
        print('\n--- Log ---')
        with open(log_file_path, 'w', encoding='utf-8') as file:
            for log in error_logs:
                print(log)
                file.write(log + '\n')

    msg = '; '.join(f'{key.capitalize()}: {counter[key]}' for key in counter)
    print('-' * len(msg), msg, '-' * len(msg), sep='\n')

    if ext_counter:
        ext_msg = 'Обработано файлов: ' + ', '.join(
            f'{ext}: {count}' for ext, count in sorted(ext_counter.items()))
        print(ext_msg)


def main():
    parser = argparse.ArgumentParser(
        description='Извлечение текста и сохранение в .json.'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Директория с файлами для обработки'
    )
    parser.add_argument(
        '--output',
        help='Выходная директория (по умолчанию: ./JSON)'
    )
    parser.add_argument(
        '--extensions',
        nargs='*',
        help=(
            'Список обрабатываемых расширений (вида --extensions fb2 epub).'
            ' При отсутствии ключа обрабатывает все поддерживаемые файлы.'
        )
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(
            f'Error: директория "{args.input}" не существует.',
            file=sys.stderr
        )
        sys.exit(1)

    process_book_files(
        input_dir=args.input,
        output_dir=args.output,
        allowed_ext=args.extensions
    )


if __name__ == '__main__':
    main()
