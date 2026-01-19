import json
from pathlib import Path

from tokenizers import Tokenizer, processors
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


def extract_texts_from_json_dir(data_dir: str | Path) -> list[str]:
    data_path = Path(data_dir)
    texts = []

    if not data_path.is_dir():
        raise FileNotFoundError(f'Директория "{data_dir}" не существует')

    json_files = list(data_path.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(
            f'В директории "{data_dir}" не найдено .json файлов')

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'text' in data:
                    text = data['text']
                    if isinstance(text, str):
                        texts.append(text.strip())
                else:
                    print(f'Файл {json_file.name} не содержит поля "text"')
        except Exception as e:
            print(f'Не удалось прочитать файл {json_file.name}: {e}')
            continue

    return texts


def train_byte_level_bpe_tokenizer(
        texts: list[str],
        vocab_size: int = 50257,
        min_frequency: int = 2,
        save_dir: str | Path = 'tokenizer_config'
) -> PreTrainedTokenizerFast:
    if not texts:
        raise ValueError('Список текстов пуст.')

    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=['<|endoftext|>', '<pad>'],
        show_progress=True,
    )

    print('Обучение токенизатора...', '-' * 35, sep='\n')
    tokenizer.train_from_iterator(texts, trainer=trainer)
    print('-' * 35, 'Обучение токенизатора завершено.', sep='\n')

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = ByteLevelDecoder()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token='<|endoftext|>',
        eos_token='<|endoftext|>',
        pad_token='<pad>',
    )
    hf_tokenizer.save_pretrained(save_dir)

    return hf_tokenizer


def main(data_dir, vocab_size, min_frequency, save_dir):
    try:
        texts = extract_texts_from_json_dir(data_dir)
        tokenizer = train_byte_level_bpe_tokenizer(
            texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            save_dir=save_dir,
        )

        test_text = 'Привет, мир! Это тест токенизатора.'
        print(f'\nТестирование токенизатора на тексте: "{test_text}"')
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        print(f'Закодировано: "{encoded}"')
        print(f'Декодировано: "{decoded}"')

        print(f'Кол-во символов: {len(test_text)}')
        print(f'Кол-во токенов : {len(encoded)}')

        print(f'Реальный размер словаря: {tokenizer.vocab_size}')

        print('Токенизатор успешно создан и сохранён.')
    except Exception as e:
        print(f'Ошибка при создании токенизатора: {e}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Обучение Byte-Level BPE токенизатора на JSON-данных')
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Путь к директории с .json файлами')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=50257,
        help='Размер словаря (по умолчанию 50257)')
    parser.add_argument(
        '--min_frequency',
        type=int,
        default=2,
        help='Минимальная частота токена для включения в словарь')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='tokenizer_config',
        help='Директория для сохранения токенизатора')

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        save_dir=args.save_dir,
    )
