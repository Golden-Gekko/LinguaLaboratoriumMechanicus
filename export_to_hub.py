import argparse
import shutil
from pathlib import Path

import torch

from llm import LinguaLaboratoriumMechanicus
from llm.configuration_llm import LinguaLaboratoriumMechanicusConfig
from llm.modeling_llm import LLMForCausalLM

EMB_DIM = 768
N_LAYERS = 12
N_HEADS = 12
MAX_CONTEXT = 1024
DROPOUT = 0.1
QKV_BIAS = False


def main(checkpoint: Path, tokenizer_path: Path, out_dir: Path) -> None:
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = ckpt['model_state_dict']
    vocab_size = ckpt.get('vocab_size') or state['token_emb.weight'].shape[0]

    model = LinguaLaboratoriumMechanicus(
        vocab_size=vocab_size, emb_dim=EMB_DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
        max_context_length=MAX_CONTEXT, dropout=DROPOUT, qkv_bias=QKV_BIAS)
    model.load_state_dict(state, strict=True)

    hf_config = LinguaLaboratoriumMechanicusConfig(
        vocab_size=vocab_size, emb_dim=EMB_DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
        max_context_length=MAX_CONTEXT, dropout=DROPOUT, qkv_bias=QKV_BIAS)

    hf_model = LLMForCausalLM(hf_config)
    hf_model.load_state_dict(model.state_dict(), strict=True)

    LinguaLaboratoriumMechanicusConfig.register_for_auto_class()
    hf_model.register_for_auto_class('AutoModelForCausalLM')
    hf_config.save_pretrained(out_dir)
    hf_model.save_pretrained(out_dir)

    shutil.copytree('llm', out_dir / 'llm', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

    for name in ('tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json'):
        src = tokenizer_path / name
        if not src.exists():
            raise FileNotFoundError(f'Нет файла токенизатора: {src}')
        shutil.copy2(src, out_dir / name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Подготовка файлов для загрузки на HF Hub')
    parser.add_argument('--checkpoint', required=True, help='Путь к .pt чекпоинту')
    parser.add_argument('--tokenizer_path', required=True, help='Директория токенизатора')
    parser.add_argument('--out_dir', default='hf_export', help='Куда сохранить HF-папку')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    main(Path(args.checkpoint), Path(args.tokenizer_path), out_dir)
