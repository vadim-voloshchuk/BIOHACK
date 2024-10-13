import pickle
import os
import torch
from PIL import Image
import numpy as np
import zipfile
from torch import nn
from scipy.spatial.distance import cosine
from custom_autoenv_level2 import AdvancedImageRestorationModel 
import onnxruntime as ort
import torchvision.transforms as transforms


# Инициализация ONNX-сессии
weights_path = "models/w600k_r50.onnx"
session = ort.InferenceSession(weights_path)


# Путь к pickle-файлу
pickle_path = "embedding_v3.pickle"

# Путь для сохранения сгенерированных изображений
output_dir = "imgs"
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Инициализация модели и загрузка весов (ваша модель уже определена ранее)
model = AdvancedImageRestorationModel().to(device)
model.load_state_dict(torch.load('image_restoration_model.pth'))
model.eval()

# Загрузка pickle файла
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

# Инициализация переменной для суммы косинусных расстояний
total_cosine_distance = 0.0

# Проход по каждому вектору из словаря
for img_name, embeddings in data.items():
    embedding = embeddings['embedding']  # Берем исходный embedding
    embedding_tensor = torch.tensor(embedding).float().unsqueeze(0).to(device)  # Преобразуем в тензор
    
    with torch.no_grad():
        # Восстанавливаем изображение из вектора
        reconstructed_image = model(embedding_tensor)
        reconstructed_image = (reconstructed_image.cpu().numpy().squeeze() * 0.5 + 0.5) * 255
        reconstructed_image = reconstructed_image.astype(np.uint8)
        
        # Создаем изображение и сохраняем в папку imgs
        image = Image.fromarray(reconstructed_image.transpose(1, 2, 0))  # Преобразуем в (height, width, channels)
        image.save(os.path.join(output_dir, img_name))

        # Вычисляем вектор для сгенерированного изображения
        img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)  # Преобразуем обратно в тензор
        generated_vector = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img_tensor.cpu().numpy()})[0]
        
        # Вычисляем косинусное расстояние между исходным и сгенерированным векторами
        cosine_distance = cosine(embedding, generated_vector[0])
        total_cosine_distance += cosine_distance

# Создаем zip архив с изображениями
archive_name = "archive.zip"
with zipfile.ZipFile(archive_name, 'w') as archive:
    for img_name in os.listdir(output_dir):
        archive.write(os.path.join(output_dir, img_name), arcname=os.path.join('imgs', img_name))

print(f"Архив {archive_name} успешно создан.")
print(f"Сумма косинусных расстояний: {total_cosine_distance}")
