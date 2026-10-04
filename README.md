# rujev

Датасеты и обучение моделей, выбирающих ответ за один проход:
`state → question → options → label`. Ответ должен следовать из контекста и вопроса.

## UI и backend

```bash
uv sync --extra model-mac --extra ui
uv run --extra model-mac --extra ui jev-model ui
```

Откройте [localhost:8501](http://localhost:8501). Выберите сохранённый run, стадию и устройство,
нажмите «Загрузить модель». Введите контекст, вопрос и варианты ответа (по одному на строку),
затем нажмите «Получить результат». Интерфейс покажет выбранный ответ, вероятности всех вариантов
и время предсказания. Можно импортировать пример из JSON или заполнить форму строкой подготовленного
датасета. Если у примера есть `label`, результат сравнивается с меткой.

Загрузка модели означает загрузку обученного run из локального каталога `runs/` вместе с его
базовой моделью. Backend использует общий `JevModel.predictor`: поддерживаются `pointer`,
`frozen_head` и `zero_shot` (для последнего нужен сервер из его конфига). В памяти остаётся одна
модель; кнопка «Выгрузить» освобождает её. При переключении старая модель выгружается до загрузки новой.

Другой каталог runs и порт:

```bash
uv run --extra model-mac --extra ui jev-model ui --runs-dir /path/to/runs --port 8502
```

Frontend и FastAPI backend работают на одном порту. API доступен в `/api`, его описание —
в [localhost:8501/docs](http://localhost:8501/docs). По умолчанию сервер слушает только `127.0.0.1`;
это локальный инструмент без авторизации. Для CUDA установите extra `model` вместо `model-mac`.

Основные запросы:

```bash
curl http://localhost:8501/api/runs
curl -X POST http://localhost:8501/api/model \
  -H 'Content-Type: application/json' \
  -d '{"run":"pointer/mac-a","stage":"ce","device":"auto"}'
curl -X POST http://localhost:8501/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"state":"The answer is blue","question":"Which color?","options":["red","blue"]}'
curl -X DELETE http://localhost:8501/api/model
```

`options` принимает строки или объекты `{"id": "blue", "text": "blue"}` из контракта датасета.
Ответ содержит `prediction`, список `options` с `probability`, сведения о модели и `elapsed_ms`.
В `/api/predict` можно передать `model_id` из ответа загрузки: backend отклонит запрос, если другая
вкладка уже сменила модель. `/api/datasets` и `/api/sample?dataset=sst2&partition=test&index=0`
дают доступ к подготовленным примерам с тем же разбиением, что при обучении и оценке.

Проверка HTTP API, включая загрузку настоящего checkpoint:

```bash
uv run --extra model-mac --extra ui python scripts/smoke_ui.py
uv run --extra model-mac --extra ui python scripts/smoke_ui.py --run-dir runs/pointer/mac-a --device cpu
```

## Локальные эксперименты на Mac M1

Нужен Python 3.12+ для **arm64**. Установка обучения без CUDA, Unsloth и Triton:

```bash
uv sync --extra model-mac
```

CLI автоматически выбирает CUDA, затем Apple GPU (`mps`), затем CPU. Устройство можно задать явно:
`--device mps` или `--device cpu`. MPS используется через PyTorch; vLLM для этих пайплайнов не нужен.
На CPU настройки тоже работают, но обучение декодера будет медленнее.

### Проверка изменений без скачивания модели и датасетов

```bash
uv run --extra model-mac python scripts/smoke_models.py
# Принудительная проверка на CPU:
uv run --extra model-mac python scripts/smoke_models.py --device cpu
```

Скрипт создаёт крошечный случайный Qwen и синтетические примеры с явным ответом в контексте.
Для `frozen_head` и `pointer` он проходит CE, RL, калибровку, загрузку сохранённых весов, оценку
и предсказание. Также проверяет разные количества вариантов ответа и лимит выборки.
Результаты сохраняются в `runs/smoke/<timestamp>/`. Эта проверка проверяет работоспособность кода;
качество случайной модели не имеет смысла. `--steps 10` позволяет увеличить число шагов.

### Эксперимент с предобученной моделью

Подготовьте SST-2, если его ещё нет:

```bash
uv run --extra model-mac jev prepare sst2
```

Оба локальных конфига используют [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
в float32, последовательности до 256 токенов и максимум 128 исходных примеров **на датасет и
раздел**. Первый запуск скачает веса. Лимит применяется после общего разделения данных;
при увеличении лимита сохраняется ранее выбранное подмножество. Калибровка температуры ограничена
диапазоном 0.01–100 и сохраняется только при отсутствии ухудшения validation NLL, чтобы маленькая
выборка не приводила к неустойчивой оценке. Для `frozen_head` лимит стоит
в `[cache]`, для `pointer` — в корне конфига. `0` снимает лимит.

Для быстрых экспериментов с головой, функцией потерь и RL:

```bash
uv run --extra model-mac jev-model train frozen_head -c configs/mac/frozen_head.toml --run-dir runs/frozen_head/mac-a
uv run --extra model-mac jev-model eval runs/frozen_head/mac-a
uv run --extra model-mac jev-model train frozen_head --run-dir runs/frozen_head/mac-a --stage rl
```

Замороженный декодер запускается при построении кэша. Следующие эксперименты с головой переиспользуют
этот кэш. Локальный кэш находится в `cache/frozen_head-mac`; лимит данных входит в его ключ.
При замене данных в `data_dir` укажите новый `cache_dir`, чтобы пересчитать признаки.

Для экспериментов с LoRA и pointer head:

```bash
uv run --extra model-mac jev-model train pointer -c configs/mac/pointer.toml --run-dir runs/pointer/mac-a
uv run --extra model-mac jev-model eval runs/pointer/mac-a
uv run --extra model-mac jev-model train pointer --run-dir runs/pointer/mac-a --stage rl
uv run --extra model-mac jev-model predict runs/pointer/mac-a --state "The answer is blue" --question "Which color?" --option red --option blue
```

Здесь декодер выполняется на каждом шаге: batch size = 1, LoRA rank = 4, checkpointing активаций
экономит память. Это стартовые настройки для M1, в том числе с 8 ГБ памяти; фактическое потребление
зависит от длины примеров и других приложений. Если памяти мало, начните с `frozen_head`.
Для `pointer` можно снизить длину до 128 токенов (`-s max_length=128 -s eval_batch_tokens=128`).
Это может исключить примеры с длинными вопросами или вариантами ответа.

Меняйте параметры и сохраняйте отдельный run для каждого эксперимента:

```bash
uv run --extra model-mac jev-model train pointer -c configs/mac/pointer.toml --run-dir runs/pointer/mac-b -s ce.lr=1e-4 -s ce.steps=80
uv run --extra model-mac jev-model compare runs/pointer/mac-a runs/pointer/mac-b
```

Для корректного сравнения оставляйте одинаковые датасеты, лимиты и длину контекста. Маленькая выборка
помогает проверять реализацию и направление изменений; результат затем нужно проверить на полной выборке.

### Перенос эксперимента на CUDA GPU

```bash
uv sync --extra model
uv run --extra model jev-model config pointer > pointer-gpu.json
uv run --extra model jev-model train pointer -c pointer-gpu.json --run-dir runs/pointer/gpu-a -s datasets=sst2 -s ce.lr=1e-4
```

Полный конфиг использует стандартную модель `unsloth/Qwen3.5-2B-Base`, длину 2048 и данные без локального
лимита. Для `frozen_head` аналогично: `jev-model config frozen_head`. Перенесите проверяемые изменения
и нужные параметры обучения в новый конфиг; подберите размер батча под память GPU.
Для `pointer` на CUDA используются Transformers и PEFT. Изменение размера базовой модели требует нового run:
LoRA и обученная голова привязаны к её архитектуре. Сохранённые веса одной архитектуры можно загружать
на CPU, MPS и CUDA; результаты между устройствами могут немного отличаться.

При необходимости уменьшить сами файлы данных используйте отдельный каталог:

```bash
uv run --extra model-mac jev prepare sst2 --max-samples 1000 --output-dir /tmp/rujev-data-small
uv run --extra model-mac jev-model train pointer -c configs/mac/pointer.toml -s data_dir=/tmp/rujev-data-small
```

`jev prepare --max-samples` ограничивает сохранённые разделы после конвертации; загрузка исходного
датасета при этом всё равно выполняется. Для обычных локальных экспериментов достаточно лимита в конфиге модели.
