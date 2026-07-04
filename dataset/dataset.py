import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class W40kDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            json_path: str | Path,
            max_length: int = 1024,
            force_reprocess=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')

        self.processed_dir = Path(json_path) / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        self.data_file = self.processed_dir / 'tokenized_data.npy'

        if force_reprocess or not self.data_file.exists():
            self._preprocess_data(json_path)

        self.token_blocks = np.load(self.data_file, mmap_mode='r')

    def _preprocess_data(self, json_dir_path):
        all_token_ids = []

        json_files = list(Path(json_dir_path).glob('*.json'))
        print(f'Найдено {len(json_files)} JSON файлов')

        for json_path in tqdm(json_files, desc='Обработка файлов'):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data['text']

                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                tokens.append(self.sep_token_id)
                all_token_ids.extend(tokens)

        total_tokens = len(all_token_ids)
        num_blocks = total_tokens // self.max_length
        print(f'Всего в корпусе данных {total_tokens:,} токенов')

        blocks = []
        for i in tqdm(range(num_blocks), desc='Создание блоков данных'):
            start = i * self.max_length
            end = start + self.max_length
            blocks.append(all_token_ids[start:end])

        np.save(self.data_file, np.array(blocks, dtype=np.int32))

    def __len__(self):
        return len(self.token_blocks)

    def __getitem__(self, idx):
        block = self.token_blocks[idx]
        return (
            torch.tensor(block[:-1], dtype=torch.long),
            torch.tensor(block[1:], dtype=torch.long))


def main(
        tokenizer_path: str,
        json_path: str,
        max_length: int,
        force_reprocess: bool,
        batch_size: int):
    print('-' * 60, 'Тестирование W40kDataset', '-' * 60, sep='\n')
    print(f'JSON путь:      {json_path}')
    print(f'Токенизатор:    {tokenizer_path}')
    print(f'Контекст:       {max_length}')
    print(f'Переобработка:  {force_reprocess}')
    print(f'Батч:           {batch_size}')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f'Размер словаря: {tokenizer.vocab_size}')
    print('-' * 60)

    dataset = W40kDataset(
        tokenizer=tokenizer,
        json_path=json_path,
        max_length=max_length,
        force_reprocess=force_reprocess
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f'Батч {batch_idx}:')
        print(f'  inputs:  {inputs.shape}  (dtype: {inputs.dtype})')
        print(f'  targets: {targets.shape}  (dtype: {targets.dtype})')

        print(
            '  Проверка смещения: все inputs[0, 1:] == targets[0, :-1]?',
            torch.all(inputs[0, 1:] == targets[0, :-1]).item()
        )

        input_text = tokenizer.decode(inputs[0][:20].tolist())
        target_text = tokenizer.decode(targets[0][:20].tolist())

        print(f'  Пример input:  {repr(input_text)}')
        print(f'  Пример target: {repr(target_text)}')

        if batch_idx == 1:
            break

    print('\nТестирование завершено')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Предобработка и тестирование Датасета')
    parser.add_argument(
        '--json_path',
        type=str,
        default='./json_data',
        help='Путь к JSON файлам')
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default='./my_tokenizer',
        help='Путь к токенизатору')
    parser.add_argument(
        '--max_length',
        type=int,
        default=1024,
        help='Длина контекста')
    parser.add_argument(
        '--force_reprocess',
        action='store_true',
        help='Переобработать данные')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Размер батча для теста')

    args = parser.parse_args()

    main(
        tokenizer_path=args.tokenizer_path,
        json_path=args.json_path,
        max_length=args.max_length,
        force_reprocess=args.force_reprocess,
        batch_size=args.batch_size
    )
