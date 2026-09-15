# FlyDurak

[English](README.md) | **Русский**

Экспериментальный проект, в котором опубликованный полный коннектом центральной нервной системы самца дрозофилы **MaleCNS** используется как рекуррентная политика для игры в подкидного дурака один на один.

В обработанном графе — **165 122 нейрона и 10 511 038 направленных связей**. Топология работает как разреженная рекуррентная сеть, а компактные обучаемые блоки превращают её состояние в допустимый карточный ход.

> Это pet/research-проект и демонстрация. Текущие результаты не доказывают, что биологическая топология лучше GRU, LSTM или других стандартных архитектур.

## Быстрый запуск игры

После установки из корня репозитория нужна одна команда:

```powershell
flydurak
```

Лаунчер автоматически:

- выберет `CUDA`, если она доступна в установленном PyTorch, иначе `CPU`;
- найдёт подготовленный граф MaleCNS;
- найдёт checkpoint мухи или автоматически скачает его из GitHub Release (около 374 МБ);
- возьмёт первый свободный порт, начиная с `8765`;
- откроет игру в браузере.

На Windows без активированного окружения можно вызвать ту же команду напрямую:

```powershell
.\.venv\Scripts\flydurak.exe
```

На Linux:

```bash
.venv/bin/flydurak
```

Остановить сервер можно сочетанием `Ctrl+C` в терминале.

### Необязательные параметры

```powershell
flydurak --device cpu
flydurak --device cuda
flydurak --port 9000
flydurak --no-browser
flydurak --no-download
```

Указанный `--port` считается предпочтительным: если он занят, лаунчер автоматически попробует следующие порты.

## Установка

Нужен Python 3.11 или новее. Для игры на ноутбуке отдельная видеокарта не обязательна.

### Windows PowerShell

```powershell
cd C:\путь\к\drosophila-git
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\flydurak.exe
```

Если команда `py` отсутствует, установите Python 3.11+ с [python.org](https://www.python.org/downloads/) и включите пункт добавления Python Launcher/Python в `PATH`.

### Linux / DGX Spark

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[dev]'
python3 -m pytest -q tests
flydurak
```

Для GPU заранее установите подходящую CUDA-сборку PyTorch по [официальной инструкции](https://pytorch.org/get-started/locally/). Сам проект не заменяет системный PyTorch и не использует `--break-system-packages`.

## Какие файлы нужны для игры

Лаунчер ищет граф в таком порядке:

1. `data/processed/malecns-v1.0-soma.pt` — предпочтительный вариант с координатами нейронов;
2. `data/processed/malecns-v1.0.pt`.

Checkpoint ищется сначала в `models/durak-fast-plastic-r32.best.pt`, затем среди основных экспериментальных checkpoint в `runs/`. Если весов нигде нет, `flydurak` автоматически скачает проверенный checkpoint из релиза `v0.1.0`. Файл загружается через временное имя и принимается только после проверки SHA-256.

Свои пути можно указать явно:

```powershell
flydurak `
  --graph ".\data\processed\malecns-v1.0-soma.pt" `
  --checkpoint ".\models\durak-fast-plastic-r32.best.pt"
```

## Подготовка MaleCNS

Большие файлы не требуется скачивать на ноутбук заранее. На сервере команда загрузит публичные таблицы и соберёт разреженный граф локально:

```bash
flysans-prepare all \
  --raw data/raw/malecns-v1.0 \
  --output data/processed/malecns-v1.0.pt \
  --minimum-weight 3
```

Ожидаемый результат:

```text
neurons: 165122
edges:   10511038
```

Чтобы добавить официальные soma-координаты для визуализации мозга:

```bash
flysans-prepare positions \
  --raw data/raw/malecns-v1.0 \
  --graph data/processed/malecns-v1.0.pt \
  --output data/processed/malecns-v1.0-soma.pt
```

## Что находится в репозитории

- полноценная headless-среда подкидного дурака на 36 карт;
- разреженная recurrent-модель на топологии MaleCNS;
- GRU, LSTM, RNN, MLP и topology-control baselines;
- scripted, mixed, search и frozen-policy соперники;
- обучение actor/critic и факторизованная пластичность;
- терминальная и браузерная игра против checkpoint;
- интерактивная карта активности нейронов и синаптических сигналов;
- тесты и инструменты экспериментального исследования.

## Структура

```text
src/flysans/durak.py              правила игры и соперники
src/flysans/durak_train.py        цикл обучения actor/critic
src/flysans/model.py              connectome-модель и baselines
src/flysans/prepare.py            загрузка и обработка MaleCNS
src/flysans/durak_web.py          браузерная игра FastAPI
src/flysans/launch.py             автоматический запуск игры
src/flysans/study.py              сравнительные эксперименты
src/flysans/topology_controls.py  контрольные топологии
tests/                             тесты
docs/                              архитектура, результаты, model card
```

## Ограничения результатов

Имеющиеся показатели получены в исследовательских запусках и подходят для выбора демонстрационного checkpoint. Для научного вывода о преимуществе биологической топологии нужны несколько seed, одинаковые бюджеты параметров и вычислений, held-out соперники, перемешанные контрольные графы и доверительные интервалы.

Дополнительные материалы: [архитектура](docs/ARCHITECTURE.md), [эксперименты](docs/EXPERIMENTS.md), [model card](docs/MODEL_CARD.md), [происхождение данных](data/README.md).

## Лицензия

Код проекта распространяется по [MIT License](LICENSE). Данные MaleCNS в репозитории не распространяются и сохраняют собственные требования к лицензии и атрибуции.
