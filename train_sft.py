import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset.chat_dataset import ChatQADataset, format_messages
from llm import LinguaLaboratoriumMechanicus


@dataclass
class Config:
    tokenizer_path: str = 'tokenizer/tokenizer_chat_config'
    json_data_dir: str = 'Warhammer_40k/parsed/lor/qa_data'
    pretrained_checkpoint: str = 'checkpoints/checkpoint_epoch10.pt'
    save_dir: str = 'checkpoints_sft'
    emb_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    max_context_length: int = 1024
    dropout: float = 0.1
    qkv_bias: bool = False
    batch_size: int = 4
    lr: float = 1e-5
    max_epochs: int = 15
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_questions: tuple[str, ...] = (
        # 'Кто такой Император?',
        # 'Что такое Эра Раздора и как она повлияла на межзвездные путешествия человечества?',
        'Кто такой Малькадор и какова была его роль при Императоре?',
    )
    eval_max_new_tokens: int = 80
    eval_temperature: float = 0.4
    eval_top_k: int = 40
    force_reprocess: bool = False
    warmup_steps: int = 150
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0


def resize_vocab(
    model: LinguaLaboratoriumMechanicus,
    new_vocab_size: int,
) -> LinguaLaboratoriumMechanicus:
    old_vocab_size = model.vocab_size
    if new_vocab_size == old_vocab_size:
        return model

    emb_dim = model.emb_dim
    old_emb = model.token_emb.weight.data
    old_head = model.out_head.weight.data

    model.token_emb = nn.Embedding(new_vocab_size, emb_dim)
    model.out_head = nn.Linear(emb_dim, new_vocab_size, bias=False)

    model.token_emb.weight.data[:old_vocab_size] = old_emb
    model.token_emb.weight.data[old_vocab_size:] = old_emb.mean(dim=0)
    model.out_head.weight.data[:old_vocab_size] = old_head
    model.out_head.weight.data[old_vocab_size:] = old_head.mean(dim=0)
    model.vocab_size = new_vocab_size

    return model


def load_pretrained(
    cfg: Config,
    vocab_size: int,
) -> LinguaLaboratoriumMechanicus:
    ckpt_path = Path(cfg.pretrained_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Чекпоинт не найден: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    old_state = ckpt['model_state_dict']
    old_vocab_size = old_state['token_emb.weight'].shape[0]

    model = LinguaLaboratoriumMechanicus(
        vocab_size=old_vocab_size,
        emb_dim=cfg.emb_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_context_length=cfg.max_context_length,
        dropout=cfg.dropout,
        qkv_bias=cfg.qkv_bias,
    )
    model.load_state_dict(old_state, strict=True)
    return resize_vocab(model, vocab_size)

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
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=-100,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    return losses


@torch.no_grad()
def run_eval(model, tokenizer, cfg: Config) -> None:
    from llm.generation import generate

    model.eval()
    mode = 'greedy' if cfg.eval_temperature < 1e-5 else f'T={cfg.eval_temperature}'
    print(f'--- Проверка генерацией (чат, {mode}) ---')
    for question in cfg.eval_questions:
        prompt = format_messages([{'role': 'user', 'content': question}])
        prompt += '\n<|assistant|>\n'
        output = generate(
            model, tokenizer, prompt,
            max_new_tokens=cfg.eval_max_new_tokens,
            temperature=cfg.eval_temperature,
            top_k=cfg.eval_top_k,
            device=cfg.device,
        )
        print(f'Вопрос: {question}')
        print(f'Ответ: {output}')


def train(cfg: Config) -> None:
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    vocab_size = len(tokenizer)

    dataset = ChatQADataset(
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

    model = load_pretrained(cfg, vocab_size).to(cfg.device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(dataloader) * cfg.max_epochs
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - cfg.warmup_steps),
        eta_min=cfg.lr * cfg.min_lr_ratio,
    )
    scheduler = SequentialLR(
        optimizer, [warmup, cosine], milestones=[cfg.warmup_steps],
    )

    for epoch in range(cfg.max_epochs):
        start = time.perf_counter()
        losses = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            vocab_size=vocab_size,
            device=cfg.device,
            grad_clip=cfg.grad_clip,
        )
        elapsed = time.perf_counter() - start

        avg_loss = sum(losses) / len(losses)
        cur_lr = scheduler.get_last_lr()[0]

        print(f'\n=== Эпоха {epoch + 1}/{cfg.max_epochs} [{elapsed:.1f}s] ===')
        print(f'AvgLoss: {avg_loss:.4f}  LR: {cur_lr:.2e}')

        run_eval(model, tokenizer, cfg)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_loss,
            'vocab_size': vocab_size,
        }, save_dir / f'checkpoint_epoch{epoch + 1:02d}.pt')

    print('SFT-обучение завершено.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFT-дообучение чат-модели')
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--json_data_dir', type=str)
    parser.add_argument('--pretrained_checkpoint', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--max_context_length', type=int)
    parser.add_argument('--force_reprocess', action='store_true')
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--min_lr_ratio', type=float)
    parser.add_argument('--grad_clip', type=float)
    parser.add_argument('--eval_temperature', type=float)
    args = parser.parse_args()

    cfg = Config()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    train(cfg)
