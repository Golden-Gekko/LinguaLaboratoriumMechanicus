import shutil
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_id: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    return model.to(device)


def load_checkpoint(checkpoint_dir: str | Path, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
    )
    return model.to(device)


def save_checkpoint(
    model,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Path,
    loss: float,
    step: int | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    optimizer: AdamW | None = None,
    scheduler=None,
) -> Path:
    if step is not None:
        ckpt_name = f'step_{step:06d}'
    elif epoch is not None:
        ckpt_name = f'epoch_{epoch:02d}'
    else:
        raise ValueError('Укажите step или epoch для save_checkpoint')

    final_dir = save_dir / ckpt_name
    tmp_dir = save_dir / f'{ckpt_name}.tmp'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)

    state: dict = {'loss': loss}
    if epoch is not None:
        state['epoch'] = epoch
    if global_step is not None:
        state['global_step'] = global_step
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()

    torch.save(state, tmp_dir / 'training_state.pt')

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    return final_dir


def load_training_state(
    checkpoint_dir: str | Path,
    optimizer: AdamW,
    scheduler,
    device: str,
) -> dict:
    state_path = Path(checkpoint_dir) / 'training_state.pt'
    state = torch.load(state_path, map_location=device, weights_only=False)
    if 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    if 'scheduler' in state:
        scheduler.load_state_dict(state['scheduler'])
    return state


def prune_checkpoints(save_dir: Path, keep_last: int = 3) -> None:
    step_dirs = sorted(
        (
            p for p in save_dir.glob('step_*')
            if p.is_dir() and not p.name.endswith('.tmp')
        ),
        key=lambda p: int(p.name.split('_')[1]),
    )
    for old_dir in step_dirs[:-keep_last]:
        shutil.rmtree(old_dir)


def find_latest_checkpoint(save_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for checkpoint_dir in save_dir.glob('step_*'):
        if not checkpoint_dir.is_dir() or checkpoint_dir.name.endswith('.tmp'):
            continue
        state_path = checkpoint_dir / 'training_state.pt'
        if not state_path.exists():
            continue
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        if 'optimizer' not in state:
            continue
        step = int(checkpoint_dir.name.split('_')[1])
        candidates.append((step, checkpoint_dir))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: str,
    grad_clip: float,
    ignore_index: int | None = None,
    *,
    start_batch: int = 0,
    global_step: int = 0,
    checkpoint_every_steps: int | None = None,
    on_checkpoint: Callable[[int, float], None] | None = None,
) -> tuple[list[float], int]:
    model.train()
    losses: list[float] = []

    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc='Train', leave=False)):
        if batch_idx < start_batch:
            continue

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(input_ids=inputs)
        logits = outputs.logits

        kwargs = {}
        if ignore_index is not None:
            kwargs['ignore_index'] = ignore_index

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            **kwargs,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        global_step += 1
        losses.append(loss.item())

        if (
            on_checkpoint is not None
            and checkpoint_every_steps
            and global_step % checkpoint_every_steps == 0
        ):
            recent_loss = sum(losses[-100:]) / min(len(losses), 100)
            on_checkpoint(global_step, recent_loss)

    return losses, global_step
