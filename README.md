
# Modeling
Управление тренировкой производится через конфиг файлы в соответсвующей папке.
В папке inference лежит все необходимое для инференса (dvc файлы модели и порогов thresholds)
## Стек и технологии.
```
poetry, dvc, black, wandb

pytorch, pytorch-lightning, albumentations, torchmetrics, iterative-stratification

onnx, onnxruntime
```
## Конфигурация рабочего окружения
```
pip install poetry
make install_c_libs
make install
```
## Где взять данные
- [здесь](https://www.kaggle.com/datasets/nikitarom/planets-dataset)
- либо подключится к dvc и сделать `dvc pull`
## Тренировка модели
Все параметры лежат в конфиге. После тренировки модель конвертируется в ONNX и так же автоматически вычисляются пороги для оптимальный классификации.

```
make train
```
## Конвертация после тренировки ONNX
```
poetry run python -m src.convert path/to/model path/to/converted/model
```
## Предсказание модели
```
poetry run python -m src.onnx_predict path/to/image path/to/onnxmodel
```
## Добавление модели в DVC
1. Инициализация DVC

    В директории проекта пишем команды:
   ```
    make dvc-init_dvc
   ```
   или руками

    ```
	dvc init --no-scm
	dvc remote add --default $(DVC_REMOTE_NAME) "your url here"$(USERNAME)/dvc_files
	dvc remote modify $(DVC_REMOTE_NAME) user $(USERNAME)
	dvc config cache.type hardlink,symlink
   ```
   2. Добавление модели
    ```
    dvc add modelname
    dvc push # отправляем на сервер
    ```
## Тестирование модели
для тестов используется pytest
```
   make test
   make test-coverage
```
## Трекинг модели
WandB
[тык](https://wandb.ai/kekus/soldatov_modeling/)
