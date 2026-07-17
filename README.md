# Lingua Laboratorium Mechanicus

Пайплайн обучения собственной языковой модели с нуля.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-astral-DE5FE9.svg)](https://docs.astral.sh/uv/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU%2032GB%2B%20%7C%20Container%20Toolkit-76B900.svg?logo=nvidia&logoColor=white)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Установка

```bash
git clone https://github.com/Golden-Gekko/LinguaLaboratoriumMechanicus
cd LinguaLaboratoriumMechanicus
uv sync --all-groups
```

Флаг `--all-groups` подтягивает dev-зависимости, нужные для извлечения текста из книг (`pymupdf`, `ebooklib`, `beautifulsoup4` и др.).

## Этап 1. Continued Pre-Training (CPT)

### 1. Извлечение текста из книг

Скрипт рекурсивно обходит входную директорию, извлекает текст из поддерживаемых форматов и сохраняет по одному JSON на файл.

Поддерживаемые форматы: `.fb2`, `.epub`, `.pdf`

Формат выходного JSON:

```json
{
  "parent_folder": "название_папки",
  "file_name": "книга.fb2",
  "text": "полный текст книги..."
}
```

```bash
uv run python utils/extract_text_to_json.py \
  --input path/to/books \
  --output dataset/json_data
```

| Параметр | Обязательный | По умолчанию | Описание |
|----------|:------------:|--------------|----------|
| `--input` | да | - | Директория с исходными файлами |
| `--output` | нет | `utils/JSON` | Куда сохранять JSON |
| `--extensions` | нет | все поддерживаемые | Фильтр расширений, например: `--extensions fb2 epub` |

Ошибки и предупреждения пишутся в `logs/log_<timestamp>.txt`.

### 2. Обучение токенизатора

Byte-Level BPE токенизатор (Hugging Face `tokenizers` + `PreTrainedTokenizerFast`). Специальные токены: `<|endoftext|>` (BOS/EOS) и `<pad>`.

```bash
uv run python tokenizer/train_tokenizer.py \
  --data_dir dataset/json_data \
  --vocab_size 50257 \
  --min_frequency 2 \
  --save_dir tokenizer/tokenizer_config
```

| Параметр | Обязательный | По умолчанию | Описание |
|----------|:------------:|--------------|----------|
| `--data_dir` | да | - | Директория с JSON из шага 1 |
| `--vocab_size` | нет | `50257` | Целевой размер словаря |
| `--min_frequency` | нет | `2` | Минимальная частота токена для включения в словарь |
| `--save_dir` | нет | `tokenizer_config` | Куда сохранить токенизатор |

### 3. Создание датасета

Скрипт токенизирует все JSON, склеивает в единый поток токенов (c `<|endoftext|>` между источниками), режет на блоки фиксированной длины и сохраняет в `path/to/json/processed/`.

```bash
uv run python dataset/dataset.py \
  --json_path dataset/json_data \
  --tokenizer_path tokenizer/tokenizer_config \
  --max_length 1024 \
  --batch_size 4
```

| Параметр | Обязательный | По умолчанию | Описание |
|----------|:------------:|--------------|----------|
| `--json_path` | нет | `./json_data` | Директория с JSON |
| `--tokenizer_path` | нет | `./my_tokenizer` | Путь к обученному токенизатору |
| `--max_length` | нет | `1024` | Длина контекста (блока) |
| `--force_reprocess` | нет | `false` | Пересоздать `processed/` даже если кэш есть |
| `--batch_size` | нет | `4` | Размер батча для тестового прогона |

### 4. Обучение модели (CPT)

```bash
uv run python train.py \
  --tokenizer_path tokenizer/tokenizer_config \
  --json_data_dir dataset/json_data \
  --save_dir checkpoints \
  --batch_size 4 \
  --max_epochs 10
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--tokenizer_path` | `tokenizer/tokenizer_config` | Путь к токенизатору |
| `--json_data_dir` | `dataset/json_data` | Директория с JSON |
| `--save_dir` | `checkpoints` | Куда сохранять чекпоинты и лоссы |
| `--batch_size` | `4` | Размер батча |
| `--lr` | `3e-4` | Learning rate |
| `--max_epochs` | `10` | Число эпох |
| `--emb_dim` | `768` | Размерность эмбеддингов |
| `--n_layers` | `12` | Число слоёв Transformer |
| `--n_heads` | `12` | Число голов внимания |
| `--max_context_length` | `1024` | Длина контекста (должна совпадать с датасетом) |
| `--warmup_steps` | `500` | Шаги линейного разогрева LR |
| `--grad_clip` | `1.0` | Максимальная норма градиента |
| `--force_reprocess` | `false` | Пересоздать кэш датасета |

После каждой эпохи сохраняется `checkpoints/checkpoint_epochNN.pt` и запускается тестовая генерация.

## Этап 2. Supervised Fine-Tuning (SFT)

Дообучение CPT-модели на диалогах в формате вопрос–ответ. Токенизатор расширяется чат-токенами ролей, лосс считается только на ответах ассистента.

### Формат данных для SFT

Формат исходных данных для датасета: один или несколько `.json` файлов структуры:

```json
[
  {
    "messages": [
      {"role": "user", "content": "Кто такой Император?"},
      {"role": "assistant", "content": "Император Человечества - ..."}
    ]
  }
]
```

### 1. Расширение токенизатора чат-токенами

Добавляет специальные токены `<|user|>` и `<|assistant|>` к базовому токенизатору.

```bash
uv run python tokenizer/extend_chat_tokens.py \
  --base tokenizer/tokenizer_config \
  --out tokenizer/tokenizer_chat_config
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--base` | `tokenizer/tokenizer_config` | Базовый токенизатор после CPT |
| `--out` | `tokenizer/tokenizer_chat_config` | Куда сохранить расширенный токенизатор |

### 2. Создание чат-датасета

Токенизирует диалоги в формате `<|role|>\n{content}`, маскирует лосс на репликах пользователя и сохраняет блоки в `path/to/qa/json/processed/`.

```bash
uv run python dataset/chat_dataset.py \
  --json_path parsed/qa_data \
  --tokenizer_path tokenizer/tokenizer_chat_config \
  --max_length 1024 \
  --batch_size 4
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--json_path` | `./json_data` | Директория с JSON-диалогами |
| `--tokenizer_path` | `./my_tokenizer` | Расширенный чат-токенизатор |
| `--max_length` | `1024` | Максимальная длина диалога в токенах |
| `--force_reprocess` | `false` | Пересоздать `processed/` |
| `--batch_size` | `4` | Размер батча для тестового прогона |

Диалоги длиннее `max_length` пропускаются.

### 3. Запуск SFT-обучения

Загружает CPT-чекпоинт, расширяет embedding-слой под новый размер словаря и дообучает на чат-данных.

```bash
uv run python train_sft.py \
  --tokenizer_path tokenizer/tokenizer_chat_config \
  --json_data_dir dataset/json_data/qa_data \
  --pretrained_checkpoint checkpoints/checkpoint_epoch10.pt \
  --save_dir checkpoints_sft \
  --batch_size 4 \
  --max_epochs 15
```

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--tokenizer_path` | `tokenizer/tokenizer_chat_config` | Расширенный токенизатор |
| `--json_data_dir` | `dataset/json_data/qa_data` | Директория с JSON-диалогами |
| `--pretrained_checkpoint` | `checkpoints/checkpoint_epoch10.pt` | CPT-чекпоинт для инициализации весов |
| `--save_dir` | `checkpoints_sft` | Куда сохранять SFT-чекпоинты |
| `--batch_size` | `4` | Размер батча |
| `--lr` | `1e-5` | Learning rate (ниже, чем на CPT) |
| `--max_epochs` | `15` | Число эпох |
| `--max_context_length` | `1024` | Длина контекста |
| `--warmup_steps` | `150` | Шаги линейного разогрева LR |
| `--min_lr_ratio` | `0.1` | Минимальный LR как доля от начального (cosine decay) |
| `--grad_clip` | `1.0` | Максимальная норма градиента |
| `--force_reprocess` | `false` | Пересоздать кэш датасета |
| `--eval_temperature` | `0.4` | Температура при тестовой генерации после эпохи |

После каждой эпохи - тестовая генерация ответов на вопросы из `eval_questions` (задаются в коде `train_sft.py`).

## Загрузка моделей на Hugging Face Hub

Для авторизации надо создать токен: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) с правом write.

```bash
uvx hf auth login --token hf_xxxx
```

### Подготовка и загрузка модели

`export_to_hub.py` готовит HF-папку из `.pt`. Загрузка - через `hf upload`.

```bash
# CPT
uv run python export_to_hub.py \
  --checkpoint checkpoints/<ИМЯ ЧЕКПОИНТА>.pt \
  --tokenizer_path tokenizer/tokenizer_config \
  --out_dir hf_export
uvx hf upload <ВАШ НИКНЕЙМ>/<ВАШ РЕПОЗИТОРИЙ> hf_export

# SFT (instruct)
uv run python export_to_hub.py \
  --checkpoint checkpoints_sft/<ИМЯ ЧЕКПОИНТА>.pt \
  --tokenizer_path tokenizer/tokenizer_chat_config \
  --out_dir hf_export_instruct
uvx hf upload <ВАШ НИКНЕЙМ>/<ВАШ РЕПОЗИТОРИЙ> hf_export_instruct
```

| Параметр | Обязательный | Описание |
|----------|:------------:|----------|
| `--checkpoint` | да | Путь к `.pt` чекпоинту |
| `--tokenizer_path` | да | Директория токенизатора |
| `--out_dir` | нет (`hf_export`) | Куда сохранить HF-папку |
