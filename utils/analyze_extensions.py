from collections import Counter
import os
import sys


def analyze_file_extensions(root_dir):
    ext_counter = Counter()

    for _, _, filenames in os.walk(root_dir):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
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

    target_dir = sys.argv[1]

    if not os.path.isdir(target_dir):
        print(f'Error: {target_dir} не является директорией.')
        sys.exit(1)

    extensions = analyze_file_extensions(target_dir)

    print('Количество файлов разных расширений:')
    for ext, count in sorted(extensions.items()):
        print(f'{ext:<10} : {count:>5}')


if __name__ == '__main__':
    main()
