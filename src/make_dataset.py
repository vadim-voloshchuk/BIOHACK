import os
import shutil
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

# Путь к папке с изображениями на диске
images_dir = "data/img_align_celeba/img_align_celeba"

# Путь к папке для сохранения выборок
dataset_dir = "data"

# Путь к скачанным весам ONNX
weights_path = "models/w600k_r50.onnx"

# Размер обучающей выборки (в процентах)
train_size = 0.8

# Создание папок для выборок
os.makedirs(os.path.join(dataset_dir, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "train", "vectors"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "test", "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "test", "vectors"), exist_ok=True)

# Инициализация сессии ONNXRuntime
session = ort.InferenceSession(weights_path)

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Список всех файлов изображений
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

# Разделение на обучающую и тестовую выборки
train_split = int(len(image_files) * train_size)
train_files = image_files[:train_split]
test_files = image_files[train_split:]

# Обработка изображений и извлечение векторов
for i, filename in enumerate(image_files):
    img_path = os.path.join(images_dir, filename)

    # Открытие и преобразование изображения
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).numpy()  # Добавление batch dimension

    # Извлечение вектора через ONNX-модель
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    vector = session.run([output_name], {input_name: img_tensor})[0]

    # Приведение вектора к типу float32
    vector = vector.astype(np.float32)

    # Сохранение изображения и вектора
    if i < train_split:
        shutil.copy(img_path, os.path.join(dataset_dir, "train", "images", filename))
        np.save(os.path.join(dataset_dir, "train", "vectors", filename[:-4] + ".npy"), vector)
    else:
        shutil.copy(img_path, os.path.join(dataset_dir, "test", "images", filename))
        np.save(os.path.join(dataset_dir, "test", "vectors", filename[:-4] + ".npy"), vector)

    print(f"Обработано изображение {i+1}/{len(image_files)}: {filename}")

print("Формирование выборок завершено.")
