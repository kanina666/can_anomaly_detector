# can_anomaly_detector

Определяет отсутствующие банки на фото матрицы KUDO для двух
фиксированных позиций камеры (`first_pos`, `second_pos`) и возвращает
решение "есть аномалия", количество обнаруженных банок и подсказку о
месте пропуска.

Обе позиции решаются **разными архитектурами** — это два независимо
откалиброванных механизма, не один переиспользуемый пайплайн:

- **`first_pos`** — `first_pos_v2/`: whole-image ResNet18 **ИЛИ**
  (row12 / row0 zone: ResNet-PatchCore **И** DINOv2-PatchCore, отфильтровано
  HSV-детектором блика). См. `first_pos_v2/model.py` за деталями и
  `CLAUDE.md` (раздел "first_pos: сессия 2026-08-20") за полной историей
  экспериментов, приведших к этой архитектуре.
- **`second_pos`** — per-cell TinyCNN + LightGBM.

## Файлы

| Файл | Назначение |
|---|---|
| `can_anomaly_detector.py` | Точка входа. Импортировать напрямую или запускать как скрипт. |
| `bbox_config.py` | Границы матрицы / зоны исключения, используются для фильтрации детекций YOLO. |
| `config.json` | Все пути к моделям, пороги и параметры калибровки. Для перенастройки редактировать этот файл, а не код. |
| `calib_yolo_first_pos.json` / `calib_yolo_second_pos.json` | Откалиброванные позиции ячеек (row, col) -> пиксельные координаты (x, y) для каждой матрицы. |
| `can_detector_yolov8n.pt` | Чекпойнт YOLOv8n. Используется для `detected_count` на обеих позициях; для `second_pos` также участвует в решении (`yolo_deficit`), для `first_pos` — только информационное поле, в решении не участвует. |
| `percell_classifier.pt` | Классификатор занятости ячейки (банка есть/отсутствует), вход 64x64. Используется только для `second_pos`. |
| `percell_lgbm_second_pos.txt` | Модель LightGBM, принимает финальное решение "аномалия/нет" для `second_pos`. |
| `first_pos_v2/` | Пайплайн `first_pos` целиком: код, векторизованный DINOv2 (см. ниже), мелкие артефакты. См. `first_pos_v2/model.py`. |

### `first_pos_v2/` подробнее

| Путь | Назначение | В git? |
|---|---|---|
| `model.py` | Логика: `check_anomaly(img, device)`, `warm(device)`. | да |
| `dinov2/` | Векторизованная (vendored) инференс-часть DINOv2 ViT-S/14 (только `layers/` + `models/vision_transformer.py`, ~120KB кода) — **без сети**: не тянет код или веса через `torch.hub` в рантайме, в отличие от исходного research-скрипта. | да |
| `artifacts/` | `resnet18_whole_image.pth` (43MB, через Git LFS), пороги (`*_meta.json`, `hsv_safe_thresholds.json`), `checksums_external.json`. | да |
| `artifacts_external/` | 4 PatchCore-банка (row12/row0 × resnet/dino, по 215MB) + 2 pretrained backbone (DINOv2 ViT-S/14 85MB, ImageNet ResNet18 45MB) — **~945MB, не в git**. | **нет** |
| `download_artifacts.py` | Скачивает `artifacts_external/` с Яндекс.Диска, проверяет sha256 по `checksums_external.json`. | да |
| `artifacts_source.json` | Публичная ссылка на папку Яндекс.Диска с внешними артефактами (заполнить перед первым запуском). | да |

**Перед первым запуском** (или после клонирования репозитория с нуля):

```
cd first_pos_v2
python download_artifacts.py
```

Без этого `first_pos_v2.check_anomaly(...)` / `--service` упадёт с понятной
ошибкой `FileNotFoundError`, перечисляющей, каких файлов не хватает.

## Зависимости

Проверено на:

```
python == 3.14
torch == 2.11.0
torchvision == 0.26.0
ultralytics == 8.4.95
opencv-python == 5.0.0
numpy == 2.4.6
lightgbm == 4.6.0
```

`torchvision` нужен только для `first_pos_v2` (resnet18-архитектура для
whole-image классификатора и PatchCore-экстрактора).

## Использование

### Из командной строки

```
python can_anomaly_detector.py photo.jpg first_pos
python can_anomaly_detector.py photo.jpg second_pos
```

Выводит результат в формате JSON в stdout.

### Как библиотека

```python
from can_anomaly_detector import check_anomaly

result = check_anomaly("photo.jpg", "first_pos")
print(result["is_anomaly"], result["probability"])
```

`check_anomaly(image_path, position, device="cpu")` — `position` это
`"first_pos"` или `"second_pos"`. `device` принимает `"cpu"`, `"cuda"`
или номер CUDA-устройства строкой (например `"0"`); на Raspberry Pi
использовать `"cpu"`.

Возвращаемое значение (общая часть для обеих позиций):

```
{
  "is_anomaly": bool,
  "probability": float,           # first_pos: вероятность whole-image классификатора (0-1);
                                   # NB: is_anomaly может быть True при probability < threshold,
                                   # если сработал сигнал row12/row0 (см. "trigger")
  "threshold": float,
  "detected_count": int,          # first_pos: информационное, в решении не участвует
  "expected_count": int,
  "location_confidence": "high" | "low",
  "top3_candidate_locations": [
    {"x": float, "y": float, "row": int, "col": int, "prob_missing": float},  # prob_missing только second_pos
    ...
  ],
  "zone": {"value": "near"|"mid"|"far", "note": str},  # только second_pos, только при низкой уверенности
  "trigger": ["whole_image" | "row12" | "row0", ...]   # только first_pos: какие сигналы сработали
}
```

Для `first_pos`, если сработал только `whole_image` (без `row12`/`row0`),
`top3_candidate_locations` будет пустым и `location_confidence` = `"low"`
— whole-image классификатор не даёт информации о конкретной ячейке,
только вероятность по всему кадру.

### "Тёплый" сервисный режим (без перезагрузки моделей на каждый вызов)

```
python can_anomaly_detector.py --service
```

Загружает все модели один раз (YOLO, percell, first_pos_v2 — включая
один прогрев-проход по каждой из них на фиктивном кадре, см. ниже),
затем построчно читает из stdin `photo.jpg position` и построчно выводит
в stdout JSON-результат. Предназначен для долго работающего процесса
(например, вызываемого из управляющего скрипта), а не для запуска нового
процесса на каждое фото — именно загрузка моделей и есть основная часть
времени одного "холодного" вызова.

**Важно про прогрев**: одной загрузки весов недостаточно. И у YOLO
(`.predict()`), и у моделей `first_pos_v2` первый настоящий forward-проход
после загрузки весов имеет собственный скрытый разовый оверхед
(внутренний прогрев аллокатора/ядер PyTorch/ultralytics) — измерено
~2.5-2.7 секунды дополнительно к загрузке весов, независимо от размера
фото. Если это не прогреть заранее, первый реальный запрос после старта
сервиса рискует не уложиться в лимит 3с сам по себе. `--service` поэтому
прогоняет один фиктивный кадр через каждую модель при старте — этот
оверхед поглощается один раз при запуске сервиса, а не на первом
реальном фото.

### Скорость (замерено на CPU разработческого ноутбука, не Raspberry Pi 5!)

После прогрева `--service`:

| Позиция | Случай | Время |
|---|---|---|
| `first_pos` | чистое фото (DINOv2 не запускается) | ~0.6-0.7с |
| `first_pos` | фото с сигналом row12/row0 (DINOv2 запускается) | ~1.5-1.8с |
| `second_pos` | любое фото | ~0.15-0.2с |

⚠️ Ни разу не измерено на настоящем Raspberry Pi 5. ARM Cortex-A76
исторически заметно слабее CPU ноутбука x86 (см. `PROJECT_METHODOLOGY.md`
в корне исследовательской директории). DINOv2 ViT-S/14 — самая тяжёлая
модель во всей системе; `first_pos` с сигналом row12/row0 (худший случай,
~1.5-1.8с на ноутбуке) — это то, что нужно в первую очередь
перепроверить на целевом железе перед вводом в эксплуатацию.

## Известные ограничения `first_pos_v2`

- **~9 из 521 ложных тревог на row12, крайние колонки (0, 11-15)**:
  причина не установлена; распределения PatchCore-score реальных
  пропусков и этих ложных тревог пересекаются на каждой проверенной оси.
  Семь разных способов почини (перекалибровка, пороги, денсификация
  банка и т.д.) не сработали или портили recall — см. CLAUDE.md. Нужны
  новые размеченные фото этого паттерна, не доработка существующих
  данных/алгоритма.
- **Не проверено на реальном Raspberry Pi 5** — см. раздел "Скорость" выше.
- **Recall/specificity** (честная независимая проверка, `audit_240` +
  `newdate_281`, n=521): recall=100.0%, specificity=96.7%,
  precision=93.0%, accuracy=97.7%.
