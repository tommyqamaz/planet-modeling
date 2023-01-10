# Modeling
Все параметры, необходимые для тренировки (например. размер батча, препроцессинг, колбеки, метрики и т.д.), находятся в файле src/config.py
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
Все параметры лежат в конфиге.

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
    dvc init
   ```
    ```
    dvc remote add --default myremote gdrive://gdrive_folder_id
    ```

    где gdrive_folder_id берется из адреса вашей папки в Google Drive https://drive.google.com/drive/folders/gdrive_folder_id

    ```
    dvc config cache.type hardlink,symlink
    dvc checkout --relink
   ```
   2. Добавление модели
    ```
    dvc add modelname
    dvc push # отправляем на сервер
    ```

## Трекинг модели
WandB
[тык](https://wandb.ai/kekus/soldatov_modeling/)