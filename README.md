# MNIST Classification: PyTorch vs Classical ML

Сравнение нейросетевых моделей на PyTorch и классических моделей машинного обучения (`scikit-learn` / `LightGBM`) на датасете **MNIST** при едином разбиении на `train / val / test` (`seed=123`).

## Структура проекта

- `config.py`, `data.py`, `train.py`, `utils.py`, `main.py` — пайплайн обучения и оценки моделей на PyTorch.
- `model.py` — архитектуры `MLPModel` и компактная свёрточная сеть `ConvModel` (~17.9k параметров) с блоками `Conv2d -> BatchNorm2d -> ReLU`, `MaxPool2d`, `Dropout2d`, `AdaptiveAvgPool2d((2, 2))` и `Dropout` перед выходным слоем.
- `logreg_sklearn.py` — baseline с `StandardScaler` и `LogisticRegression` на плоских векторах признаков (`28x28 = 784`).
- `tree_models.py` — оценка `RandomForestClassifier` и `LGBMClassifier` на тех же сплитах.
- `requirements.txt` — зависимости окружения.

## Установка

```powershell
pip install -r requirements.txt
```

## Как запускать

### PyTorch (`MLPModel` и `ConvModel`)

```powershell
python main.py --model conv --epochs 15 --lr 1e-3 --optim adamw
python main.py --model mlp
```

### Классический ML (`LogisticRegression`, `RandomForest`, `LightGBM`)

```powershell
python logreg_sklearn.py
python tree_models.py --model rf
python tree_models.py --model lgbm
```

## Результаты экспериментов

| Модель | Параметры | Val Acc | Test Acc | Test Loss | Комментарий |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LogisticRegression** | - | 0.9112 | 0.9208 | 0.3382 | Линейная граница по 784 признакам |
| **RandomForest** | - | 0.9657 | 0.9702 | 0.2519 | Нелинейные сплиты, без 2D-пространства |
| **LightGBM** | - | 0.9705 | 0.9756 | 0.0822 | Лучшая классика и самый низкий `test_loss` среди всех моделей |
| **MLPModel** | 600,330 | 0.9908 | 0.9888 | 0.1214 | Полносвязная сеть поверх вектора 784 |
| **ConvModel** | 17,850 | **0.9920** | **0.9924** | 0.1071 | Лучшая точность при минимуме параметров за счёт 2D-свёрток |

## Выводы

В работе сравнены пять подходов к классификации MNIST - от линейной модели до свёрточной сети. Все модели обучены и оценены на одном и том же сплите (54 000 train / 6 000 val / 10 000 test, seed=123), метрики для нейросетей посчитаны по лучшему чекпоинту (выбран по `val_loss`) в eval-режиме.

Наилучшую точность показала **ConvModel**: `test_acc = 0.9924` при всего **17 850 параметрах** - в 34 раза меньше, чем у MLPModel (600 330). **MLPModel** достигает близкое качество (`test_acc = 0.9888`), но требует на два порядка больше весов и не использует пространственную структуру пикселей.

Среди классических методов лучший результат у **LightGBM** (`test_acc = 0.9756`) - он же даёт **самый низкий `test_loss` среди всех моделей** (`0.0822`), так как бустинг с logloss лучше калибрует вероятности, тогда как у нейросетей loss завышен из-за `label_smoothing=0.1` в функции потерь. RandomForest и LightGBM полностью запоминают train-набор (`train_acc = 1.0`), но сохраняют высокую обобщающую способность за счёт ансамблирования. **LogisticRegression** ожидаемо недообучается (`train_acc 0.946 → test_acc 0.921`) из-за линейности границы.

Итог: для MNIST свёрточная сеть - оптимальный выбор, а градиентный бустинг остаётся сильным конкурентом.