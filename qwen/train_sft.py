import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.optimization import get_cosine_schedule_with_warmup

from chat_dataset import QwenChatDataset
from training import (
    load_checkpoint,
    load_tokenizer,
    save_checkpoint,
    train_one_epoch,
)


@dataclass
class SftConfig:
    model_id: str = 'Qwen/Qwen3-1.7B-Base'
    cpt_checkpoint: str = 'checkpoints_qwen/cpt/epoch_02'
    json_data_dir: str = 'dataset/json_data/qa_data'
    save_dir: str = 'checkpoints_qwen/sft'
    max_context_length: int = 2048
    batch_size: int = 4
    lr: float = 1e-5
    max_epochs: int = 3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    force_reprocess: bool = False
    warmup_steps: int = 100
    grad_clip: float = 1.0
    eval_questions: tuple[str, ...] = (
        'Что такое Гибельный шторм и как он повлиял на ход войны?',
        'Что такое Эра Раздора и как она повлияла на межзвездные путешествия человечества?',
    )
    eval_max_new_tokens: int = 120


@torch.no_grad()
def run_eval_sft(
    model,
    tokenizer: PreTrainedTokenizerBase,
    cfg: SftConfig,
) -> None:
    model.eval()
    print('--- Проверка генерацией (SFT, greedy) ---')
    for question in cfg.eval_questions:
        messages = [{'role': 'user', 'content': question}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors='pt',
            enable_thinking=False,
        ).to(cfg.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.eval_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = inputs['input_ids'].shape[1]
        text = tokenizer.decode(
            output_ids[0, prompt_len:],
            skip_special_tokens=True,
        )
        print(f'Вопрос: {question}')
        print(f'Ответ: {text}')


def train(cfg: SftConfig) -> None:
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(cfg.cpt_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'CPT-чекпоинт не найден: {ckpt_path}')

    tokenizer = load_tokenizer(cfg.model_id)
    dataset = QwenChatDataset(
        tokenizer=tokenizer,
        json_path=cfg.json_data_dir,
        max_length=cfg.max_context_length,
        force_reprocess=cfg.force_reprocess,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    model = load_checkpoint(cfg.cpt_checkpoint, cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(dataloader) * cfg.max_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps,
    )

    for epoch in range(cfg.max_epochs):
        start = time.perf_counter()
        losses, _ = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.device,
            grad_clip=cfg.grad_clip,
            ignore_index=-100,
        )
        elapsed = time.perf_counter() - start

        avg_loss = sum(losses) / len(losses)
        cur_lr = scheduler.get_last_lr()[0]

        print(f'\n=== Эпоха {epoch + 1}/{cfg.max_epochs} [{elapsed:.1f}s] ===')
        print(f'AvgLoss: {avg_loss:.4f}  LR: {cur_lr:.2e}')

        run_eval_sft(model, tokenizer, cfg)
        save_checkpoint(model, tokenizer, save_dir, loss=avg_loss, epoch=epoch + 1)

    print('SFT-обучение завершено.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFT-дообучение Qwen на Q&A')
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--cpt_checkpoint', type=str)
    parser.add_argument('--json_data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--max_context_length', type=int)
    parser.add_argument('--force_reprocess', action='store_true')
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--grad_clip', type=float)
    args = parser.parse_args()

    cfg = SftConfig()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    train(cfg)
