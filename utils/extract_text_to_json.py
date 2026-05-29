import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from extractors import (
    extract_text_from_epub, extract_text_from_fb2, extract_text_from_pdf)

EXTENSION_TO_EXTRACTOR = {
    '.fb2': extract_text_from_fb2,
    '.epub': extract_text_from_epub,
    '.pdf': extract_text_from_pdf
}

LOG_DIR = Path('logs')


def process_book_files(
        input_dir: Path,
        output_dir: Path | None = None,
        allowed_ext: list[str] = None
) -> None:
    error_logs = []
    counter = Counter()
    ext_counter = Counter()

    if output_dir is None:
        output_dir = Path(__file__).parent / 'JSON'
    output_dir.mkdir(parents=True, exist_ok=True)

    if allowed_ext is None:
        allowed_extensions = set(EXTENSION_TO_EXTRACTOR.keys())
    else:
        allowed_extensions = set()
        for ext in allowed_ext:
            ext = ext.lower()
            if not ext.startswith('.'):
                ext = f'.{ext}'
            if ext in EXTENSION_TO_EXTRACTOR:
                allowed_extensions.add(ext)
            else:
                print(f'Warning: неподдерживаемое расширение: "{ext}"')

    files_list = [
        f for f in input_dir.rglob('*')
        if f.is_file() and f.suffix.lower() in allowed_extensions
    ]

    for file_path in tqdm(files_list, desc='Обработка'):
        ext = file_path.suffix.lower()
        extractor = EXTENSION_TO_EXTRACTOR.get(ext)

        try:
            text = extractor(file_path)
            if text is None:
                error_logs.append(f'Warning: Не найден текст в "{file_path}"')
                counter['warnings'] += 1
                continue

            output_file_path = output_dir / f'{file_path.stem}.json'

            data = {
                'parent_folder': file_path.parent.name,
                'file_name': file_path.name,
                'text': text
            }

            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            counter['processed'] += 1
            ext_counter[ext] += 1

        except Exception as e:
            counter['errors'] += 1
            error_logs.append(f'Ошибка при обработке "{file_path}": {e}')

    if error_logs:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = LOG_DIR / f'log_{timestamp}.txt'

        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write('\n'.join(error_logs) + '\n')

        if len(error_logs) < 10:
            print('\n--- Log ---')
            for log in error_logs:
                print(log)
        else:
            print(f'Лог ошибок сохранён в: {log_file_path}')


    if counter:
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

    if not Path(args.input).is_dir():
        print(f'Error: директория "{args.input}" не существует.', file=sys.stderr)
        sys.exit(1)

    process_book_files(
        input_dir=Path(args.input),
        output_dir=Path(args.output) if args.output else None,
        allowed_ext=args.extensions
    )


if __name__ == '__main__':
    main()
