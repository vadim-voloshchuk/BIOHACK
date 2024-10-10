import os
import shutil
import zipfile
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from insightface.app import FaceAnalysis

# Путь к папке с изображениями на диске
images_dir = "/path/to/images/on/disk" 

# Путь к папке для сохранения выборок
dataset_dir = "data"

# Путь к скачанным весам ResNet50@WebFace600K
weights_path = "models/buffalo_l.zip"

# Размер обучающей выборки (в процентах)
train_size = 0.8

# Создание папок для выборок
os.makedirs(os.path.join(dataset_dir, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "train", "vectors"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "test", "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "test", "vectors"), exist_ok=True)

# Инициализация FaceAnalysis для извлечения векторов
app = FaceAnalysis(name="buffalo_l", root="models")
app.prepare(ctx_id=0, det_size=(640, 640))

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Размер, ожидаемый ResNet50@WebFace600K
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
    
    # Извлечение вектора с помощью FaceAnalysis
    faces = app.get(Image.open(img_path))
    if faces:
      vector = faces[0].embedding
      vector = vector.astype(np.float32)  # Приведение к типу float32
    else:
      print(f"Не удалось обнаружить лицо на изображении: {filename}")
      continue

    # Сохранение изображения и вектора
    if i < train_split:
        shutil.copy(img_path, os.path.join(dataset_dir, "train", "images", filename))
        np.save(os.path.join(dataset_dir, "train", "vectors", filename[:-4] + ".npy"), vector) 
    else:
        shutil.copy(img_path, os.path.join(dataset_dir, "test", "images", filename))
        np.save(os.path.join(dataset_dir, "test", "vectors", filename[:-4] + ".npy"), vector)

    print(f"Обработано изображение {i+1}/{len(image_files)}: {filename}")

print("Формирование выборок завершено.")