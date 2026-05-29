import sys
from collections import Counter
from pathlib import Path


def analyze_file_extensions(root_dir: Path) -> Counter:
    ext_counter = Counter()

    for file_path in root_dir.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext:
                ext_counter[ext] += 1
            else:
                ext_counter['(no extension)'] += 1

    return ext_counter


def main():
    if len(sys.argv) != 2:
        print(
            'Использование: python analyze_extensions.py <path_to_directory>')
        sys.exit(1)

    target_dir = Path(sys.argv[1])

    if not target_dir.is_dir():
        print(f'Error: {target_dir} не является директорией.')
        sys.exit(1)

    extensions = analyze_file_extensions(target_dir)

    print('Количество файлов разных расширений:')
    for ext, count in sorted(extensions.items()):
        print(f'{ext:<10} : {count:>5}')


if __name__ == '__main__':
    main()
