import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def format_messages(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg['role']
        content = msg['content'].strip()
        parts.append(f'<|{role}|>\n{content}')
    return '\n'.join(parts)


class ChatQADataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            json_path: str | Path,
            max_length: int = 1024,
            force_reprocess=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.processed_dir = Path(json_path) / 'processed'
        self.processed_dir.mkdir(exist_ok=True)

        self.inputs_file = self.processed_dir / 'input_blocks.npy'
        self.targets_file = self.processed_dir / 'target_blocks.npy'

        if force_reprocess or not self.inputs_file.exists():
            self._preprocess_data(json_path)

        self.input_blocks = np.load(self.inputs_file, mmap_mode='r')
        self.target_blocks = np.load(self.targets_file, mmap_mode='r')

    def _build_sequence(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        input_ids: list[int] = []
        train_positions: set[int] = set()

        for msg in messages:
            role = msg['role']
            header_ids = self.tokenizer.encode(
                f'<|{role}|>\n', add_special_tokens=False,
            )
            content_ids = self.tokenizer.encode(
                msg['content'].strip(), add_special_tokens=False,
            )

            content_start = len(input_ids) + len(header_ids)
            input_ids.extend(header_ids)
            input_ids.extend(content_ids)

            if role == 'assistant':
                for i in range(content_start - 1, content_start + len(content_ids) - 1):
                    train_positions.add(i)

        input_ids.append(self.tokenizer.eos_token_id)
        if messages and messages[-1]['role'] == 'assistant':
            train_positions.add(len(input_ids) - 2)

        labels = [-100] * len(input_ids)
        for i in train_positions:
            if i + 1 < len(input_ids):
                labels[i] = input_ids[i + 1]

        return input_ids, labels

    def _preprocess_data(self, json_dir_path):
        json_files = sorted(Path(json_dir_path).glob('*.json'))
        print(f'Найдено {len(json_files)} JSON файлов')

        input_blocks: list[list[int]] = []
        target_blocks: list[list[int]] = []
        skipped = 0

        for json_path in tqdm(json_files, desc='Обработка файлов'):
            if json_path.name == 'example.json':
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                dialogs = json.load(f)

            for dialog in dialogs:
                input_ids, labels = self._build_sequence(dialog['messages'])

                if len(input_ids) > self.max_length:
                    skipped += 1
                    continue

                pad_len = self.max_length - len(input_ids)
                ids = input_ids + [self.pad_token_id] * pad_len
                labs = labels + [-100] * pad_len
                input_blocks.append(ids[:-1])
                target_blocks.append(labs[:-1])

        if not input_blocks:
            raise ValueError(
                f'Нет диалогов в пределах max_length={self.max_length}. '
                f'Пропущено: {skipped}',
            )

        print(f'Загружено диалогов: {len(input_blocks)}')
        if skipped:
            print(f'Пропущено длинных диалогов: {skipped}')

        np.save(self.inputs_file, np.array(input_blocks, dtype=np.int32))
        np.save(self.targets_file, np.array(target_blocks, dtype=np.int32))

    def __len__(self):
        return len(self.input_blocks)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_blocks[idx], dtype=torch.long),
            torch.tensor(self.target_blocks[idx], dtype=torch.long),
        )


def main(
        tokenizer_path: str,
        json_path: str,
        max_length: int,
        force_reprocess: bool,
        batch_size: int):
    print('-' * 60, 'Тестирование ChatQADataset', '-' * 60, sep='\n')
    print(f'JSON путь:      {json_path}')
    print(f'Токенизатор:    {tokenizer_path}')
    print(f'Контекст:       {max_length}')
    print(f'Переобработка:  {force_reprocess}')
    print(f'Батч:           {batch_size}')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f'Размер словаря: {tokenizer.vocab_size}')
    print('-' * 60)

    dataset = ChatQADataset(
        tokenizer=tokenizer,
        json_path=json_path,
        max_length=max_length,
        force_reprocess=force_reprocess,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f'Батч {batch_idx}:')
        print(f'  inputs:  {inputs.shape}  (dtype: {inputs.dtype})')
        print(f'  targets: {targets.shape}  (dtype: {targets.dtype})')

        trained = (targets[0] != -100).sum().item()
        total = targets[0].numel()
        print(f'  Токенов с лоссом: {trained}/{total}')

        input_text = tokenizer.decode(inputs[0][:80].tolist())
        print(f'  Пример input:  {repr(input_text)}')

        if batch_idx == 1:
            break

    print('\nТестирование завершено')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Предобработка и тестирование ChatQADataset')
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
        batch_size=args.batch_size,
    )
