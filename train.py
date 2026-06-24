import argparse
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from dataset.dataset import W40kDataset
from llm import LinguaLaboratoriumMechanicus
from llm.configuration_llm import LinguaLaboratoriumMechanicusConfig
from llm.generation import generate
from llm.modeling_llm import LLMForCausalLM


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
    qvk_bias: bool = False
    batch_size: int = 4
    lr: float = 3e-4
    max_epochs: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_prompts: tuple[str] = ('Кто такой Император?', )
    eval_max_new_tokens: int = 100
    eval_temperature: float = 0.8
    eval_top_k: int = 40
    force_reprocess: bool = False
    warmup_steps: int = 500
    grad_clip: float = 1.0
    hub_repo_id: str | None = None


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


def _push_to_hub(model, cfg: Config) -> None:
    print(f'\n\n=== Загрузка на HF: {cfg.hub_repo_id} ===')

    hf_config = LinguaLaboratoriumMechanicusConfig(
        vocab_size=cfg.vocab_size,
        emb_dim=cfg.emb_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_context_length=cfg.max_context_length,
        dropout=cfg.dropout,
        qvk_bias=cfg.qvk_bias,
    )
    hf_model = LLMForCausalLM(hf_config)
    hf_model.load_state_dict(model.state_dict(), strict=True)

    tmp = Path(tempfile.mkdtemp())
    LinguaLaboratoriumMechanicusConfig.register_for_auto_class()
    hf_model.register_for_auto_class('AutoModelForCausalLM')
    hf_config.save_pretrained(tmp)
    hf_model.save_pretrained(tmp)

    shutil.copytree('llm', tmp / 'llm', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

    tk_path = Path(cfg.tokenizer_path)
    for f in ['tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json']:
        src = tk_path / f
        if src.exists():
            shutil.copy2(src, tmp / f)

    api = HfApi()
    api.create_repo(cfg.hub_repo_id, exist_ok=True)
    api.upload_folder(repo_id=cfg.hub_repo_id, folder_path=str(tmp))
    shutil.rmtree(tmp)
    print(f'Готово: https://huggingface.co/{cfg.hub_repo_id}')


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
        qvk_bias=cfg.qvk_bias,
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

    if cfg.hub_repo_id:
        _push_to_hub(model, cfg)


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
    parser.add_argument('--hub-repo-id', type=str)
    parser.add_argument('--warmup-steps', type=int)
    parser.add_argument('--grad-clip', type=float)
    args = parser.parse_args()

    cfg = Config()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    train(cfg)
