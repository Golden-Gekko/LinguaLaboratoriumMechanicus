import argparse
from pathlib import Path

from transformers import AutoTokenizer


def extend_tokenizer(base_path: str | Path, out_path: str | Path) -> int:
    base_path = Path(base_path)
    out_path = Path(out_path)

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    added = tokenizer.add_tokens(
        ['<|user|>', '<|assistant|>'],
        special_tokens=True)
    out_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_path)

    print(f'Базовый токенизатор: {base_path}')
    print(f'Добавлено токенов: {added}')
    print(f'Размер словаря: {len(tokenizer)}')
    print(f'Сохранено в: {out_path.resolve()}')
    return len(tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Расширить токенизатор токенами ролей для чата',
    )
    parser.add_argument(
        '--base',
        type=str,
        default='tokenizer/tokenizer_config')
    parser.add_argument(
        '--out',
        type=str,
        default='tokenizer/tokenizer_chat_config')
    args = parser.parse_args()
    extend_tokenizer(args.base, args.out)
