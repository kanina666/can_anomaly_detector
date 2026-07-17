# can_anomaly_detector

Определяет отсутствующие банки на фото матрицы KUDO для двух
фиксированных позиций камеры (`first_pos`, `second_pos`) и возвращает
решение "есть аномалия", количество обнаруженных банок и подсказку о
месте пропуска.

## Файлы

| Файл | Назначение |
|---|---|
| `can_anomaly_detector.py` | Точка входа. Импортировать напрямую или запускать как скрипт. |
| `bbox_config.py` | Границы матрицы / зоны исключения, используются для фильтрации детекций YOLO. |
| `config.json` | Все пути к моделям, пороги и параметры калибровки. Для перенастройки редактировать этот файл, а не код. |
| `calib_yolo_first_pos.json` / `calib_yolo_second_pos.json` | Откалиброванные позиции ячеек (row, col) -> пиксельные координаты (x, y) для каждой матрицы. |
| `can_detector_yolov8n.pt` | Чекпойнт YOLOv8n, дообученный на детекцию банок. |
| `percell_classifier.pt` | Классификатор занятости ячейки (банка есть/отсутствует), вход 64x64. |
| `percell_lgbm_second_pos.txt` | Модель LightGBM (gradient boosting), принимает финальное решение "аномалия/нет" для second_pos. Для first_pos решение принимает логрегрессия из `config.json` (`ensemble_logreg`), отдельная модель не нужна — `config.json`, поле `ensemble_type`, указывает, какой механизм используется для какой позиции. |

## Зависимости

Проверено на:

```
python == 3.14
torch == 2.11.0
ultralytics == 8.4.95
opencv-python == 5.0.0
numpy == 2.4.6
lightgbm == 4.6.0
```

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

Возвращаемое значение:

```
{
  "is_anomaly": bool,
  "probability": float,           # 0-1, уверенность ансамбля, что кадр аномальный
  "threshold": float,             # порог решения для этой позиции
  "detected_count": int,
  "expected_count": int,
  "location_confidence": "high" | "low",
  "top3_candidate_locations": [
    {"x": float, "y": float, "row": int, "col": int, "prob_missing": float},
    ...
  ],
  "zone": {"value": "near"|"mid"|"far", "note": str}   # только second_pos, только при низкой уверенности
}
```

### "Тёплый" сервисный режим (без перезагрузки моделей на каждый вызов)

```
python can_anomaly_detector.py --service
```

Загружает обе модели один раз, затем построчно читает из stdin
`photo.jpg position` и построчно выводит в stdout JSON-результат.
Предназначен для долго работающего процесса (например, вызываемого из
управляющего скрипта), а не для запуска нового процесса на каждое
фото - именно загрузка моделей и есть основная часть времени одного
"холодного" вызова.
