import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from dataset.dataset import W40kDataset
from llm import LinguaLaboratoriumMechanicus
from llm.generation import generate


@dataclass
class Config:
    tokenizer_path: str = 'tokenizer/tokenizer_config'
    json_data_dir: str = 'dataset/json_data'
    save_dir: str = 'checkpoints'
    vocab_size: int = 50257
    emb_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    max_context_length: int = 1024
    dropout: float = 0.1
    qkv_bias: bool = False
    batch_size: int = 4
    lr: float = 3e-4
    max_epochs: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_prompts: tuple[str, ...] = (
        'В 31-м тысячелетии Империум',
        'Гибельный шторм — это',
    )
    eval_max_new_tokens: int = 100
    eval_temperature: float = 0.8
    eval_top_k: int = 40
    force_reprocess: bool = False
    warmup_steps: int = 500
    grad_clip: float = 1.0


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    vocab_size: int,
    device: str,
    grad_clip: float,
) -> list[float]:
    model.train()
    losses: list[float] = []

    for inputs, targets in tqdm(dataloader, desc='Train', leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    return losses


def run_eval(model, tokenizer, cfg: Config) -> None:
    model.eval()
    print(f'--- Проверка генерацией ---')
    for prompt in cfg.eval_prompts:
        output = generate(
            model, tokenizer, prompt,
            max_new_tokens=cfg.eval_max_new_tokens,
            temperature=cfg.eval_temperature,
            top_k=cfg.eval_top_k,
            device=cfg.device,
        )
        print(f'Вопрос: {prompt}. Нагенерировано: {output}')


def train(cfg: Config) -> None:
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)

    dataset = W40kDataset(
        tokenizer=tokenizer,
        json_path=cfg.json_data_dir,
        max_length=cfg.max_context_length,
        force_reprocess=cfg.force_reprocess,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = LinguaLaboratoriumMechanicus(
        vocab_size=cfg.vocab_size,
        emb_dim=cfg.emb_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_context_length=cfg.max_context_length,
        dropout=cfg.dropout,
        qkv_bias=cfg.qkv_bias,
    ).to(cfg.device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=len(dataloader) * cfg.max_epochs,
    )

    for epoch in range(cfg.max_epochs):
        start = time.perf_counter()
        losses = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            vocab_size=cfg.vocab_size,
            device=cfg.device,
            grad_clip=cfg.grad_clip)
        elapsed = time.perf_counter() - start

        cur_lr = scheduler.get_last_lr()[0]
        avg_loss = sum(losses) / len(losses)

        torch.save(losses, save_dir / f'loss_epoch{epoch + 1:02d}.pt')
        print(f'\n=== Эпоха {epoch + 1}/{cfg.max_epochs} [{elapsed:.1f}s] ===')
        print(f'AvgLoss: {avg_loss:.4f}  LR: {cur_lr:.2e}')

        run_eval(model, tokenizer, cfg)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_loss,
        }, save_dir / f'checkpoint_epoch{epoch + 1:02d}.pt')

    print('Обучение завершено.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--emb_dim', type=int)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--n_heads', type=int)
    parser.add_argument('--max_context_length', type=int)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--json_data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--force_reprocess', action='store_true')
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--grad_clip', type=float)
    args = parser.parse_args()

    cfg = Config()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    train(cfg)
