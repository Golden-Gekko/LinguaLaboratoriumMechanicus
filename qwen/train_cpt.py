import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.optimization import get_cosine_schedule_with_warmup

from corpus_dataset import QwenCorpusDataset
from training import (
    find_latest_checkpoint,
    load_base_model,
    load_checkpoint,
    load_tokenizer,
    load_training_state,
    prune_checkpoints,
    save_checkpoint,
    train_one_epoch,
)


@dataclass
class CptConfig:
    model_id: str = 'Qwen/Qwen3-1.7B-Base'
    json_data_dir: str = 'dataset/json_data'
    save_dir: str = 'checkpoints_qwen/cpt'
    max_context_length: int = 2048
    batch_size: int = 4
    lr: float = 2e-5
    max_epochs: int = 2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    force_reprocess: bool = False
    warmup_steps: int = 100
    grad_clip: float = 1.0
    checkpoint_every_steps: int = 1000
    keep_checkpoints: int = 3
    resume_from: str | None = None
    eval_prompts: tuple[str, ...] = (
        'В 31-м тысячелетии Империум',
        'Гибельный шторм — это',
    )
    eval_max_new_tokens: int = 120
    eval_temperature: float = 0.8
    eval_top_k: int = 40


@torch.no_grad()
def run_eval_cpt(
    model,
    tokenizer: PreTrainedTokenizerBase,
    cfg: CptConfig,
) -> None:
    model.eval()
    print('--- Проверка генерацией (CPT) ---')
    for prompt in cfg.eval_prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(cfg.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.eval_max_new_tokens,
            do_sample=True,
            temperature=cfg.eval_temperature,
            top_k=cfg.eval_top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(
            output_ids[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        print(f'Промпт: {prompt}')
        print(f'Ответ: {text}')


def _resolve_resume_dir(cfg: CptConfig, save_dir: Path) -> Path | None:
    if cfg.resume_from is None:
        return None
    if cfg.resume_from == 'auto':
        resume_dir = find_latest_checkpoint(save_dir)
        if resume_dir is None:
            print('Resume auto: чекпоинт не найден, начинаем с нуля.')
        return resume_dir
    resume_dir = Path(cfg.resume_from)
    if not resume_dir.exists():
        raise FileNotFoundError(f'Чекпоинт для resume не найден: {resume_dir}')
    return resume_dir


def train(cfg: CptConfig) -> None:
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg.model_id)
    dataset = QwenCorpusDataset(
        tokenizer=tokenizer,
        json_path=cfg.json_data_dir,
        max_length=cfg.max_context_length,
        force_reprocess=cfg.force_reprocess,
    )
    dataloader_len = len(DataLoader(dataset, batch_size=cfg.batch_size))
    total_steps = dataloader_len * cfg.max_epochs

    resume_dir = _resolve_resume_dir(cfg, save_dir)
    start_epoch = 0
    start_batch = 0
    global_step = 0

    if resume_dir is not None:
        print(f'Resume из чекпоинта: {resume_dir}')
        model = load_checkpoint(resume_dir, cfg.device)
        optimizer = AdamW(model.parameters(), lr=cfg.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps,
        )
        state = load_training_state(resume_dir, optimizer, scheduler, cfg.device)
        if 'optimizer' not in state:
            print(
                'Предупреждение: в чекпоинте нет optimizer state. '
                'Эпоха начнётся заново, веса модели загружены.'
            )
        else:
            global_step = state.get('global_step', 0)
            start_epoch = global_step // dataloader_len
            start_batch = global_step % dataloader_len
            print(
                f'Продолжаем: global_step={global_step}, '
                f'эпоха={start_epoch + 1}, батч={start_batch}'
            )
    else:
        model = load_base_model(cfg.model_id, cfg.device)
        optimizer = AdamW(model.parameters(), lr=cfg.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps,
        )

    def on_checkpoint(step: int, loss: float) -> None:
        ckpt_path = save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            save_dir=save_dir,
            loss=loss,
            step=step,
            global_step=step,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        prune_checkpoints(save_dir, cfg.keep_checkpoints)
        print(f'[step {step}] Чекпоинт сохранён: {ckpt_path} (loss={loss:.4f})')

    for epoch in range(start_epoch, cfg.max_epochs):
        generator = torch.Generator()
        generator.manual_seed(epoch)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=generator,
        )

        epoch_start_batch = start_batch if epoch == start_epoch else 0
        start = time.perf_counter()
        losses, global_step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.device,
            grad_clip=cfg.grad_clip,
            start_batch=epoch_start_batch,
            global_step=global_step,
            checkpoint_every_steps=cfg.checkpoint_every_steps,
            on_checkpoint=on_checkpoint,
        )
        elapsed = time.perf_counter() - start

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        cur_lr = scheduler.get_last_lr()[0]

        print(f'\n=== Эпоха {epoch + 1}/{cfg.max_epochs} [{elapsed:.1f}s] ===')
        print(f'AvgLoss: {avg_loss:.4f}  LR: {cur_lr:.2e}  global_step={global_step}')

        run_eval_cpt(model, tokenizer, cfg)
        save_checkpoint(
            model,
            tokenizer,
            save_dir,
            loss=avg_loss,
            epoch=epoch + 1,
            global_step=global_step,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    print('CPT-обучение завершено.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CPT-дообучение Qwen на корпусе')
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--json_data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--max_context_length', type=int)
    parser.add_argument('--force_reprocess', action='store_true')
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--grad_clip', type=float)
    parser.add_argument('--checkpoint_every_steps', type=int)
    parser.add_argument('--keep_checkpoints', type=int)
    parser.add_argument('--resume_from', type=str)
    args = parser.parse_args()

    cfg = CptConfig()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    train(cfg)
