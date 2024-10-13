# BIOHACK: Восстановление изображений лиц из векторов 🧬

## Описание проекта 🕵️‍♀️

Этот проект посвящен задаче восстановления изображений лиц из векторов, полученных с помощью модели ResNet50@WebFace600K. Для восстановления используется вариационный автоэнкодер (VAE), а качество восстановления оценивается с помощью ArcFace.

## Структура репозитория 📁

```
BIOHACK/
├── src/
│   ├── core/
│   │   ├── dataset.py        # Скрипт для работы с датасетами 📊
│   │   ├── model.py          # Модель для восстановления изображений 🖼️
│   │   └── train_vae.py      # Скрипт для обучения модели 🧠
│   ├── download_celeba.py     # Скрипт для скачивания датасета CelebA 📥
│   ├── download_weights.py     # Скрипт для скачивания весов модели 📥
│   ├── get_results.py         # Скрипт для получения результатов 📈
│   ├── make_dataset.py        # Скрипт для создания обучающей и тестовой выборок ✂️
│   └── utils.py               # Модуль с вспомогательными функциями 🛠️
├── .gitignore                 # Файл для игнорирования ненужных файлов в Git 🚫
├── app.py                     # Основной файл приложения Streamlit 🖥️
├── README.md                  # Этот файл 📖
└── requirements.txt           # Список необходимых библиотек 📦
```

## Установка 🧰

1. Клонируйте репозиторий: `git clone https://github.com/your_username/BIOHACK.git`
2. Создайте виртуальное окружение: `python3 -m venv biovenv`
3. Активируйте виртуальное окружение: 
   - Linux/macOS: `source biovenv/bin/activate`
   - Windows: `biovenv\Scripts\activate`
4. Установите необходимые библиотеки: `pip install -r requirements.txt`
5. Скачайте веса модели ResNet50@WebFace600K: `python src/download_weights.py`

## Использование 🚀

### 1. Подготовка данных 🗄️

* **Вариант 1: Скачивание датасета CelebA:**
   - Запустите скрипт `src/download_celeba.py`. 
   - Датасет будет скачан в папку `data/celeba`.
* **Вариант 2: Использование своих данных:**
   - Поместите изображения в папку `data/images`.
   - Запустите скрипт `src/make_dataset.py` для создания обучающей и тестовой выборок.

### 2. Обучение VAE 🧠

- Запустите скрипт `src/core/train_vae.py`. 
- Обученная модель будет сохранена в директорию `models`.

### 3. 🖼️ Инференс в приложении: Image Restoration Processing

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.0%2B-brightgreen.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)


## 🏁 Запуск приложения

1. Убедитесь, что у вас установлен **Python** и необходимые библиотеки.
2. Поместите модель `w600k_r50.onnx` в папку `models`.
3. Поместите файл с весами `image_restoration_model.pth` в корневую директорию проекта.
4. Запустите приложение с помощью команды:

```bash
streamlit run app.py
```

## 🛠️ Использование

Приложение имеет два основных действия, которые можно выбрать из бокового меню:

### 1. Восстановление изображения из вектора 🖼️➡️🔧

- Загрузите файл с векторами в формате `.npy` или `.json`.
- Полученный вектор будет отображен на экране.
- Если вектор имеет размер `(1, 512)`, нажмите кнопку **"Восстановить изображение"**, чтобы восстановить изображение из вектора.
- Восстановленное изображение будет отображено на странице. 🎊

### 2. Получение вектора из изображения 📷➡️📊

- Загрузите изображение в формате `.jpg` или `.png`.
- Изображение будет преобразовано и подано на вход модели для извлечения вектора.
- Извлечённый вектор и его форма будут показаны на экране.
- Введите имя файла для сохранения вектора (без расширения) и нажмите кнопку **"Скачать вектор"**, чтобы сохранить вектор в формате `.npy`. 💾

## ⚠️ Примечания

- Для корректной работы приложения необходимы предварительно обученные модели.
- Убедитесь, что модель и файл с весами находятся в правильных директориях. 📂

## Пример использования ⌨️

```bash
# Скачать датасет CelebA
python src/download_celeba.py

# Создать обучающую и тестовую выборки
python src/make_dataset.py

# Обучить VAE
python src/core/train_vae.py
```
# Важные ссылки:

## 📺 Скринкаст

Посмотрите скринкаст о том, как использовать приложение: [Скринкаст](https://drive.google.com/file/d/1pQbJIt3QEkiZtNDm8EsR_8jvS6h7W8UY/view?usp=drive_link)

## 📁 Архив с изображениями

Скачайте архив с изображениями тестовой выборки: [Архив изображений](https://drive.google.com/file/d/18mky0aL2xqhlM6MRSC0ucfIf58azHBIZ/view?usp=drive_link)

## 📥 Предобученная модель

Скачайте предобученную модель: [Предобученная модель](https://drive.google.com/file/d/1eAg7T5j6qqrodI459wo_76n_fl_-lOdw/view?usp=drive_link)
