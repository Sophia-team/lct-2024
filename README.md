## Лидеры цифровой трансформации - 2024
# Предиктивная модель для рекомендации продуктов банка
## Команда `InsightAI`
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Описание
## Настройка окружения
PDM: Python package management tool
```
pdm install
```
Подробнее https://github.com/pdm-project/pdm
### Основные зависимости
- pytorch-lifestream
- py-boost

## Описание решения
Хотели сначала сделать единый мультимодальный эмбединг, но считалось очень долго, поэтому, пока что,делам три эмбединга и конкатим их.  
```mermaid
graph TD;
    train_transactions-->trans_embedder;
    train_dial-->dial_embedder;
    train_geo-->geo_embedder;
    trans_embedder-->train_trans_embeddings
    trans_embedder-->val_trans_embeddings
    trans_embedder-->test_trans_embeddings
    geo_embedder-->train_geo_embeddings
    geo_embedder-->val_geo_embeddings
    geo_embedder-->test_geo_embeddings
    dial_embedder-->train_dial_embeddings
    dial_embedder-->val_dial_embeddings
    dial_embedder-->test_dial_embeddings
    train_trans_embeddings-->train
    train_geo_embeddings-->train
    train_dial_embeddings-->train
    val_trans_embeddings-->val
    val_geo_embeddings-->val
    val_dial_embeddings-->val
    test_trans_embeddings-->test
    test_geo_embeddings-->test
    test_dial_embeddings-->test
```
`train` - эмбединги по клиентам на основе обучающей выборки на каждый месяц  
`val` - эмбединги по клиентам на основе тестовой выборки на каждый месяц  
`test` - эмбединги по клиентам на следующий месяц (2023-01-31), который заливается в лидерборд  

Эмбединги считаются на данных из прошлого, event_date < date_trunc(target_month).  
Затем обучается pyboost на 4 таргетах.  

### Тюнинг pytorch-lifestream
Из выборок берётся sample в 5% клиентов. На этом сэмпле тестируется пайплайн и подбирается размер слоёв, длина цепочки.

### Обучение pytorch-lifestream
На 50% данных (ресурсов и времени мало) обучаем эмбеддер и получаем эмбединги для каждого клиента на каждый доступный месяц. 
`notebooks/embeddings.ipynb` - скрипт получения эмбедингов.  

### Обучение pyboost
С помощью полученных эмбеддинегов строится бустинги.  

## Структура репозитория
```
├── README.md
├── data               <- данные (в гит не загружены, очень большие)
├── models             <- используемые модели (для эмбедингов и для предикшена)
└── notebooks          <- Jupyter notebooks с примерами и исследованиями
    ├── embeddings.ipynb    <- создание эмбеддингов
    ├── drafts              <- черновики
    ├── baseline_core.ipynb      <- бейзлайн модели без эмбеддингов
    └── baseline.ipynb      <- бейзлайн модели
├── pyproject.toml     <- Project configuration file
├── requirements.txt
└── src                     <- Source code
    ├── __init__.py
    └── lct_2024                
        ├── __init__.py          
        └── utils.py        <- используемые скрипты и функции
```
